"""A real-voice test set for the Ear from DUET's recorded calls (`runs/live-*`): every caller turn cut from what DUET
heard after echo cancelling, a draft transcript by the teacher (TurboCTC), and a page to check each one by ear.

    .venv/Scripts/python ear/realcalls.py cut runs runs/ear-real      (laptop: DUET's VAD and TurboCTC on DirectML)
    .venv/Scripts/python ear/realcalls.py review runs/ear-real        (http://localhost:4310, saves truth.jsonl)
    python prep.py calls ~/ear/data/real ~/ear/prep --calls-kind clean --name real    (spark-1; then --dev real)

`cut`: call.json logs when each caller turn ended, not when it began. Silero VAD (> 0.5) on the echo-cancelled audio
finds speech. While DUET's own channel plays (R > -45 dBFS) and for 0.4 s after, a frame counts as the caller's only
if it is 6 dB above the echo DUET's level predicts (that call's median gap between the two), so DUET's residual
voice is left out. Blocks shorter than 0.2 s are dropped. Speech more than 2 s before the rest of the turn is left
out, and the turn grows by up to 0.6 s through speech running on into DUET's echo at either edge. A turn is the caller's speech after the previous caller turn or backchannel and up to
this one, plus 0.3 s each side. Writes `clean/<id>.wav` (16 kHz, as DUET heard it) and `truth.jsonl` {id, text (the
reference: the teacher's draft until checked), draft, live (what DUET transcribed during the call), speaker, checked,
drop}. `review` keeps the reference fixes and drops in the same truth.jsonl; prep.py reads only id and text, so the
dropped turns are filtered out when the set is copied (`export`).
"""
import argparse
import difflib
import html
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

RATE, CH = 16000, 512
F = CH / RATE


def caller_frames(heard, stereo, rate2):
    from duet.vad import Vad
    v, n = Vad(), len(heard) // CH
    p = np.array([v.step(heard[k * CH:(k + 1) * CH]) for k in range(n)])
    hdb = 10 * np.log10((heard[:n * CH].reshape(n, CH) ** 2).mean(1) + 1e-12)
    m = int(round(F * rate2))
    r = stereo[:, 1]
    nr = min(n, len(r) // m)
    rdb = np.full(n, -200.0)
    rdb[:nr] = 10 * np.log10((r[:nr * m].reshape(nr, m) ** 2).mean(1) + 1e-12)
    voiced, duet = p > 0.5, rdb > -45
    tail = int(0.4 / F)  # DUET's voice rings on in the room for a moment after its channel stops
    near = np.convolve(duet, np.ones(tail + 1), "full")[:n] > 0
    ref = np.maximum.accumulate(np.where(duet, np.arange(n), 0))  # DUET's level now, or when it last played
    level = np.where(near, rdb[ref], -200.0)
    both = voiced & duet
    gain = float(np.median(hdb[both] - rdb[both])) if both.sum() >= 10 else -20.0  # echo level relative to DUET's
    on = voiced & (~near | (hdb > level + gain + 6))
    k = 0  # drop blocks shorter than 0.2 s
    while k < n:
        if on[k]:
            e = k
            while e < n and on[e]:
                e += 1
            if (e - k) * F < 0.2:
                on[k:e] = False
            k = e
        else:
            k += 1
    return on, voiced, gain


def span(on, voiced, lo, hi, gap=2.0, grow=0.0):
    """The turn's frames in [lo, hi): the last run of caller frames with no pause over `gap`, grown through speech
    that runs on into DUET's echo at either edge (a first word said over DUET's last one), by at most `grow`."""
    idx = np.flatnonzero(on[lo:hi]) + lo
    if len(idx) == 0:
        return None
    j = len(idx) - 1
    while j > 0 and (idx[j] - idx[j - 1] - 1) * F <= gap:
        j -= 1
    a, b, g = idx[j], idx[-1], int(grow / F)
    for _ in range(g):
        if a <= lo or not voiced[a - 1]:
            break
        a -= 1
    for _ in range(g):
        if b + 1 >= len(voiced) or not voiced[b + 1]:
            break
        b += 1
    return a, b


def quoted(text):
    m = re.match(r'\s*"(.*?)"\s', text + " ")
    return m.group(1) if m else ""


def cut(args):
    import soundfile as sf
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from duet.asr import Asr
    asr = Asr(directml=True)
    out = Path(args.out)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for call in sorted(Path(args.runs).glob("live-*")):
        heard, rate = sf.read(call / "heard_after_echo_cancel.wav", dtype="float32")
        stereo, rate2 = sf.read(call / "call_mic_L_duet_R.wav", dtype="float32", always_2d=True)
        assert rate == RATE, (call, rate)
        on, voiced, gain = caller_frames(heard, stereo, rate2)
        events = json.loads((call / "call.json").read_text(encoding="utf-8"))["events"]
        start, k = 0.0, 0
        for e in events:
            if e["kind"] == "backchnl":
                start = max(start, e["t"])
            if e["kind"] != "caller":
                continue
            lo, hi = int(start / F), min(len(on), int((e["t"] + 0.1) / F))
            start = e["t"]
            s = span(on, voiced, lo, hi)
            uid = f"{call.name.removeprefix('live-')}_{k:02d}"
            k += 1
            if s is None:
                skipped.append((uid, quoted(e["text"])))
                continue
            a = max(0, int((s[0] * F - 0.3) * RATE))
            b = min(len(heard), int(((s[1] + 1) * F + 0.3) * RATE))
            clip = heard[a:b]
            sf.write(out / "clean" / f"{uid}.wav", clip, RATE)
            draft = asr.long(clip)
            rows.append({"id": uid, "text": draft, "draft": draft, "live": quoted(e["text"]), "speaker": "author",
                         "start": round(a / RATE, 2), "seconds": round(len(clip) / RATE, 2), "checked": False,
                         "drop": False})
        print(f"{call.name}: echo {gain:.0f} dB below DUET's channel, {k} caller turns", flush=True)
    (out / "truth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} clips, {sum(r['seconds'] for r in rows) / 60:.1f} min -> {out}; no caller speech found: "
          f"{skipped}")


def export(args):
    """The checked set for prep.py: copies of the kept clips in clean/ and a truth.jsonl with id and text."""
    import shutil
    src, dst = Path(args.out), Path(args.dest)
    rows = [json.loads(l) for l in (src / "truth.jsonl").read_text(encoding="utf-8").splitlines()]
    keep = [r for r in rows if not r["drop"] and (r["checked"] or args.drafts)]
    (dst / "clean").mkdir(parents=True, exist_ok=True)
    for r in keep:
        shutil.copyfile(src / "clean" / f"{r['id']}.wav", dst / "clean" / f"{r['id']}.wav")
    (dst / "truth.jsonl").write_text("".join(json.dumps({"id": r["id"], "text": r["text"], "speaker": r["speaker"]})
                                             + "\n" for r in keep), encoding="utf-8")
    print(f"{len(keep)} of {len(rows)} clips -> {dst} ({sum(r['checked'] for r in keep)} checked)")


def marked(a, b):
    """a's words, with those not in b highlighted."""
    wa, wb = a.split(), b.split()
    norm = lambda ws: [re.sub(r"[^a-z0-9']", "", w.lower()) for w in ws]  # noqa: E731
    out = []
    for op, i1, i2, _, _ in difflib.SequenceMatcher(None, norm(wa), norm(wb), autojunk=False).get_opcodes():
        seg = html.escape(" ".join(wa[i1:i2]))
        if seg:
            out.append(seg if op == "equal" else f"<mark>{seg}</mark>")
    return " ".join(out)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Ear real-voice check</title><style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1b;--dim:#6b6a66;--line:#e2e0da;--mark:#ffe08a;--ok:#2f7d4f;--drop:#a33}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#1f1f1d;--ink:#ecebe7;--dim:#9a988f;--line:#33322f;
--mark:#6b5410;--ok:#5fbf86;--drop:#e07070}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:2}
h1{font-size:17px;margin:0 0 4px}header p{margin:0;color:var(--dim);font-size:13px}
main{max-width:860px;margin:0 auto;padding:12px 16px 80px}
.c{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
.c.checked{border-left:4px solid var(--ok)}.c.drop{opacity:.55;border-left:4px solid var(--drop)}
.top{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.id{font:12px ui-monospace,monospace;color:var(--dim)}
audio{height:32px;flex:1;min-width:220px}.k{font-size:12px;color:var(--dim);margin:8px 0 2px}
.t{margin:0}mark{background:var(--mark);color:inherit;border-radius:3px;padding:0 2px}
textarea{width:100%;box-sizing:border-box;font:15px/1.4 system-ui,sans-serif;padding:8px;border-radius:8px;
border:1px solid var(--line);background:var(--bg);color:var(--ink);resize:vertical;margin-top:4px}
button{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid var(--line);background:var(--bg);
color:var(--ink);cursor:pointer}button.ok{border-color:var(--ok);color:var(--ok)}button.dr{color:var(--drop)}
.row{display:flex;gap:8px;margin-top:8px;align-items:center}.st{font-size:12px;color:var(--dim);margin-left:auto}
</style></head><body><header><h1>Ear real-voice check: what did the caller say?</h1>
<p>Play each clip and fix the reference so it is exactly the caller's words. Leave out um / uh, keep the draft's way of
writing numbers. <b>Drop</b> clips that are not the caller (DUET's own voice, noise) or are cut off mid-word.
Yellow = where the live transcript and the draft differ. <span id="p"></span></p></header><main>%ROWS%</main>
<script>
const n=document.querySelectorAll('.c').length;
function prog(){const c=document.querySelectorAll('.c.checked').length,d=document.querySelectorAll('.c.drop').length;
document.getElementById('p').textContent=`Done ${c+d} of ${n} (${c} checked, ${d} dropped).`}
async function save(el,checked,drop){const id=el.dataset.id,text=el.querySelector('textarea').value;
const r=await fetch('/save',{method:'POST',body:JSON.stringify({id,text,checked,drop})});
el.querySelector('.st').textContent=r.ok?'saved':'NOT saved';el.classList.toggle('checked',checked&&!drop);
el.classList.toggle('drop',drop);prog();if(checked||drop){const nx=el.nextElementSibling;
if(nx){nx.scrollIntoView({behavior:'smooth',block:'center'});const a=nx.querySelector('audio');a.play().catch(()=>{})}}}
document.querySelectorAll('.c').forEach(el=>{el.querySelector('.ok').onclick=()=>save(el,true,false);
el.querySelector('.dr').onclick=()=>save(el,false,true);
el.querySelector('textarea').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();
save(el,true,false)}})});prog();
</script></body></html>"""


def review(args):
    root = Path(args.out)
    path = root / "truth.jsonl"

    def load():
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]

    def card(r):
        cls = " drop" if r["drop"] else " checked" if r["checked"] else ""
        return (f'<div class="c{cls}" data-id="{r["id"]}"><div class="top"><span class="id">{r["id"]} &middot; '
                f'{r["seconds"]:.1f} s</span><audio controls preload="none" src="/clip/{r["id"]}.wav"></audio></div>'
                f'<div class="k">DUET heard live</div><p class="t">{marked(r["live"], r["draft"]) or "&ndash;"}</p>'
                f'<div class="k">Teacher on this clip (draft)</div><p class="t">{marked(r["draft"], r["live"])}</p>'
                f'<div class="k">Reference</div><textarea rows="2">{html.escape(r["text"])}</textarea>'
                f'<div class="row"><button class="ok">Correct (Enter)</button><button class="dr">Drop</button>'
                f'<span class="st"></span></div></div>')

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(200, PAGE.replace("%ROWS%", "".join(map(card, load()))).encode(), "text/html; charset=utf-8")
            elif m := re.fullmatch(r"/clip/([\w-]+)\.wav", self.path):
                f = root / "clean" / f"{m.group(1)}.wav"
                self.send(200, f.read_bytes(), "audio/wav") if f.exists() else self.send(404, b"", "text/plain")
            else:
                self.send(404, b"", "text/plain")

        def do_POST(self):
            if self.path != "/save":
                return self.send(404, b"", "text/plain")
            d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            rows = load()
            for r in rows:
                if r["id"] == d["id"]:
                    r.update(text=" ".join(str(d["text"]).split()), checked=bool(d["checked"]), drop=bool(d["drop"]))
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            self.send(200, b"ok", "text/plain")

    print(f"http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut")
    c.add_argument("runs")
    c.add_argument("out")
    r = sub.add_parser("review")
    r.add_argument("out")
    r.add_argument("--port", type=int, default=4310)
    e = sub.add_parser("export")
    e.add_argument("out")
    e.add_argument("dest")
    e.add_argument("--drafts", action="store_true", help="also unchecked clips, with the teacher's draft as reference")
    args = ap.parse_args()
    {"cut": cut, "review": review, "export": export}[args.cmd](args)


if __name__ == "__main__":
    main()
