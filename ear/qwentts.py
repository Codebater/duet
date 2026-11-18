"""Rung 1d: our conversational lines in thousands of new voices, rendered with Qwen3-TTS 1.7B (Apache-2.0) on spark-2.

The real-voice test showed the ear fails on voices it has not heard, not on words. So each of the 19,666 lines is
spoken by two designed voices (VoiceDesign: a voice from a description: gender, age, one of ~60 accents, manner, voice
quality, setting) and two cloned ones (Base, from the speaker embedding of one 6-12 s clip of a real VoxPopuli or
LibriSpeech speaker). Held-out voices (100 designed, the cloned speakers the ear never trained on) speak the 593 never-
heard "far" lines as a new-TTS-voice test. The 12 test-call voices are kept out: no description shares a test voice's
gender, age band and accent.

    python qwentts.py refs ~/ear/prep ~/ear/data refs.jsonl                  (spark-1: one reference clip per speaker)
    python qwentts.py voices refs.jsonl voices.json                         (designed + cloned voices, held-out flags)
    python qwentts.py jobs voices.json lines.jsonl far_lines.jsonl jobs/     (train_design, train_clone, test_*)
    python qwentts.py render jobs/train_design.jsonl voices.json out/train   (/opt/voice/venv on spark-2; resumable)

`render` writes `clean/<id>.wav` (16 kHz, 16-bit, with tts.py's lead-in, tail, noise floor and random level) and
appends {id, text, speaker} to `truth_<jobs>.jsonl` batch by batch; the parts are joined into `truth.jsonl` for prep.py.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


GENDERS = ["man", "woman"]
AGES = ["teenage", "in their twenties", "in their thirties", "in their forties", "in their fifties",
        "in their sixties", "in their seventies", "in their eighties"]
NATIVE = ["General American", "Southern US", "New York", "Boston", "Midwestern American", "Californian", "Texan",
          "Canadian", "British RP", "London", "Northern English", "Scottish", "Irish", "Welsh", "Australian",
          "New Zealand", "South African", "Nigerian", "Kenyan", "Ghanaian", "Jamaican", "Indian", "Pakistani",
          "Filipino", "Singaporean"]
SECOND = ["Czech", "Slovak", "Polish", "German", "Dutch", "Swedish", "French", "Spanish", "Mexican", "Brazilian",
          "Portuguese", "Italian", "Russian", "Ukrainian", "Romanian", "Hungarian", "Greek", "Turkish", "Arabic",
          "Persian", "Mandarin Chinese", "Cantonese", "Japanese", "Korean", "Vietnamese", "Thai", "Indonesian"]
EAST = {"Czech", "Slovak", "Polish", "Russian", "Ukrainian", "Romanian", "Hungarian"}
MANNER = ["speaking fast", "slow and careful", "a bit mumbly", "sounding tired", "excited", "hesitant, with pauses",
          "loud and confident", "soft-spoken", "cheerful", "a little annoyed", "relaxed", "nervous", "warm and chatty",
          "flat and bored", "casual"]
QUALITY = ["", "", "a deep voice", "a high voice", "a raspy voice", "a breathy voice", "a slightly nasal voice",
           "a husky voice"]
SETTING = ["on a phone call", "talking into a laptop microphone", "on a video call from home", "calling from a car",
           "in a kitchen", "close to the microphone", "a bit far from the microphone", "chatting with a friend"]
# the 12 test-call voices (frontdesk-bench/data/intents.json) as (gender, age, accent group): never described
TEST = {("man", "in their forties", "Southern US"), ("man", "in their fifties", "Southern US"),
        ("woman", "in their twenties", "American"), ("woman", "in their seventies", "American"),
        ("man", "in their forties", "New York"), ("woman", "in their thirties", "Spanish"),
        ("man", "in their thirties", "Indian"), ("man", "in their twenties", "American"),
        ("woman", "in their fifties", "Midwestern American"), ("man", "in their forties", "American"),
        ("woman", "in their thirties", "British RP"), ("woman", "teenage", "American"),
        ("man", "in their sixties", "East")}
AMERICAN = {"General American", "Californian", "Texan", "Boston", "Midwestern American", "Southern US", "New York"}


def describe(rng):
    while True:
        g, age = rng.choice(GENDERS), rng.choice(AGES)
        native = rng.random() < 0.5
        acc = rng.choice(NATIVE if native else SECOND)
        groups = {acc, "American" if acc in AMERICAN else "", "East" if acc in EAST else ""}
        if any((g, age, a) in TEST for a in groups):
            continue
        pos = "his" if g == "man" else "her"
        who = (f"A teenage {'boy' if g == 'man' else 'girl'}" if age == "teenage"
               else f"A {g} {age.replace('their', pos)}")
        accent = f"with a {acc} accent" if native else f"with a {acc} accent, speaking English as a second language"
        parts = [f"{who} {accent}", rng.choice(MANNER)]
        q = rng.choice(QUALITY)
        if q:
            parts.append(f"with {q}")
        parts.append(rng.choice(SETTING))
        return ", ".join(parts) + ".", {"gender": g, "age": age, "accent": acc}


def refs(args):
    """One clean 6-12 s clip per speaker: the one closest to 9 s among those the teacher hears within 10 % WER."""
    from data import edits, words  # noqa: the ear's own helpers (spark-1)
    prep, data = Path(args.prep), Path(args.data)
    files = {}
    for root in [data / "vox", data / "LibriSpeech"]:
        for p in root.rglob("*"):
            if p.suffix in (".wav", ".flac"):
                files[p.stem] = str(p)
    train_vox = {m["speaker"] for p in prep.glob("vox_[0-9]*.json") for m in json.loads(p.read_text())}
    out = []
    for pre, dataset, held in [("vox", "voxpopuli", False), ("voxdev", "voxpopuli", True),
                               ("voxtest", "voxpopuli", True), ("train", "librispeech", False),
                               ("train360", "librispeech", False), ("train500", "librispeech", False),
                               ("dev", "librispeech", True), ("devother", "librispeech", True)]:
        best = {}
        for p in sorted(prep.glob(f"{pre}_[0-9]*.json")):
            for m in json.loads(p.read_text(encoding="utf-8")):
                if not 6 <= m["seconds"] <= 12 or m["utt"] not in files:
                    continue
                r, h = words(m["truth"]), words(m["teacher"])
                if not r or edits(r, h) / len(r) > 0.10:
                    continue
                k = (dataset, m["speaker"])
                if k not in best or abs(m["seconds"] - 9) < abs(best[k]["seconds"] - 9):
                    best[k] = {"dataset": dataset, "speaker": m["speaker"], "utt": m["utt"], "seconds": m["seconds"],
                               "path": files[m["utt"]], "heldout": held}
        for (dataset, spk), r in best.items():
            if held and dataset == "voxpopuli" and spk in train_vox:
                continue  # VoxPopuli's dev/test speakers overlap its training speakers
            out.append(r)
    seen, uniq = set(), []
    for r in out:
        if (r["dataset"], r["speaker"]) not in seen:
            seen.add((r["dataset"], r["speaker"]))
            uniq.append(r)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in uniq), encoding="utf-8")
    for d in ("voxpopuli", "librispeech"):
        n = [r for r in uniq if r["dataset"] == d]
        print(f"{d}: {sum(not r['heldout'] for r in n)} training speakers, {sum(r['heldout'] for r in n)} held out")


def voices(args):
    rng = np.random.default_rng(args.seed)
    out, seen = {}, set()
    while len(out) < args.designed + args.designed_heldout:
        text, meta = describe(rng)
        if text in seen:
            continue
        seen.add(text)
        out[f"d{len(out):04d}"] = {"kind": "design", "describe": text, **meta,
                                  "heldout": len(out) >= args.designed}
    rows = [json.loads(l) for l in Path(args.refs).read_text(encoding="utf-8").splitlines()]
    train = [r for r in rows if not r["heldout"]]
    vox = [r for r in train if r["dataset"] == "voxpopuli"]
    libri = [r for r in train if r["dataset"] == "librispeech"]
    pick = vox + [libri[i] for i in rng.permutation(len(libri))[:max(0, args.cloned - len(vox))]]
    held = [r for r in rows if r["heldout"]]
    for k, r in enumerate(pick + held):
        out[f"c{k:04d}"] = {"kind": "clone", "ref": Path(r["path"]).name, **r}
    Path(args.out).write_text(json.dumps(out, indent=0), encoding="utf-8")
    kinds = [(v["kind"], v["heldout"]) for v in out.values()]
    print({f"{k} {'held out' if h else 'train'}": kinds.count((k, h)) for k, h in sorted(set(kinds))})


def jobs(args):
    rng = np.random.default_rng(args.seed)
    vs = json.loads(Path(args.voices).read_text(encoding="utf-8"))
    pool = {(k, h): [v for v, m in vs.items() if m["kind"] == k and m["heldout"] == h]
            for k in ("design", "clone") for h in (False, True)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, rows):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"{name}: {len(rows)} clips")

    lines = [json.loads(l) for l in Path(args.lines).read_text(encoding="utf-8").splitlines()]
    for kind, tag in (("design", "d"), ("clone", "c")):
        p = pool[(kind, False)]
        write(f"train_{kind}", [{"id": f"{r['id']}_{tag}{j}", "text": r["text"], "voice": p[v]}
                                for r in lines for j, v in enumerate(rng.choice(len(p), 2, replace=False))])
    far = [json.loads(l) for l in Path(args.test_lines).read_text(encoding="utf-8").splitlines()]
    for kind, tag in (("design", "hd"), ("clone", "hc")):
        p = pool[(kind, True)]
        write(f"test_{kind}", [{"id": f"{r['id']}_{tag}", "text": r["text"], "voice": p[int(rng.integers(len(p)))]}
                               for r in far])


def meminfo(key="MemAvailable"):
    """GB of memory as the kernel counts it; on a Spark that includes the GPU's (unified memory)."""
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 2 ** 20
    return float("inf")


def reclaim(target):
    """CUDA on a Spark counts only free memory, not the page cache, so after big file copies it can fail to start
    while plenty is 'available'. Touching a block of ordinary memory makes the kernel evict cache for it; freeing the
    block leaves that memory free. Only at start, and never within 30 GB of 'available': on unified memory that figure
    overstates what can be evicted, and a mid-run reclaim on 2026-11-06 ran spark-1 out of memory (a teammate's ComfyUI and
    this render were killed)."""
    size = min(target - meminfo("MemFree"), meminfo("MemAvailable") - 30)
    if size > 1:
        block = np.ones(int(size * 2 ** 30), dtype=np.uint8)
        del block


def busy(urls):
    """True while a ComfyUI on this Spark has work queued or running: the render waits for it (it was here first)."""
    import urllib.request
    for url in urls.split(","):
        try:
            q = json.load(urllib.request.urlopen(url, timeout=3))
        except Exception:
            continue  # not running
        if q.get("queue_running") or q.get("queue_pending"):
            return True
    return False


def render(args):
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from scipy.signal import resample_poly

    vs = json.loads(Path(args.voices).read_text(encoding="utf-8"))
    todo = [json.loads(l) for l in Path(args.jobs).read_text(encoding="utf-8").splitlines()]
    out = Path(args.out)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    tpath = out / f"truth_{Path(args.jobs).stem}.jsonl"
    done = {json.loads(l)["id"] for l in tpath.read_text(encoding="utf-8").splitlines()} if tpath.exists() else set()
    todo = sorted([j for j in todo if j["id"] not in done], key=lambda j: len(j["text"]))
    if not todo:
        print(f"{len(done)} done, nothing to render", flush=True)
        return
    kind = vs[todo[0]["voice"]]["kind"]
    reclaim(args.min_free)
    model = Qwen3TTSModel.from_pretrained(f"{args.models}/Qwen3-TTS-12Hz-1.7B-{'VoiceDesign' if kind == 'design' else 'Base'}",
                                          device_map="cuda:0",
                                          dtype=torch.bfloat16, attn_implementation="sdpa")
    prompts = {}
    if kind == "clone":  # one speaker embedding per voice, computed once
        need = sorted({j["voice"] for j in todo})
        for b in range(0, len(need), 64):
            chunk = need[b:b + 64]
            items = model.create_voice_clone_prompt(ref_audio=[str(Path(args.refs) / vs[v]["ref"]) for v in chunk],
                                                    x_vector_only_mode=[True] * len(chunk))
            prompts.update(zip(chunk, items))
    rng = np.random.default_rng(args.seed + len(done))
    t0, secs = time.time(), 0.0
    print(f"{len(done)} done, {len(todo)} to render with {kind}", flush=True)
    groups, cur = [], []  # memory grows with batch x length: cap both the clips and the characters per batch
    for j in todo:
        if cur and (len(cur) >= args.batch or sum(len(c["text"]) for c in cur) + len(j["text"]) > args.chars):
            groups.append(cur)
            cur = []
        cur.append(j)
    groups += [cur] if cur else []
    b = 0
    with open(tpath, "a", encoding="utf-8") as tf:
        for chunk in groups:
            if busy(args.yield_to) or meminfo() < args.min_free:  # the Spark is shared: others first
                print(f"waiting: ComfyUI busy or {meminfo():.0f} GB available", flush=True)
                while busy(args.yield_to) or meminfo() < args.min_free:
                    time.sleep(30)
            torch.manual_seed(args.seed + len(done) + b)
            texts, lang = [j["text"] for j in chunk], ["English"] * len(chunk)
            # A clip that never ends keeps its whole batch generating to the model's 2,048-step cap, and memory grows
            # all the way (it ran spark-1 out of memory twice): cap the steps at twice the slowest pilot voice's time
            # (11 characters a second) plus 2 s, at 12 codec steps a second. Truncated clips fail the teacher filter.
            steps = int(12 * (max(map(len, texts)) / 5.5 + 2))
            if kind == "design":
                wavs, sr = model.generate_voice_design(text=texts, instruct=[vs[j["voice"]]["describe"] for j in chunk],
                                                       language=lang, max_new_tokens=steps)
            else:
                wavs, sr = model.generate_voice_clone(text=texts, language=lang, max_new_tokens=steps,
                                                      voice_clone_prompt=[prompts[j["voice"]] for j in chunk])
            for j, w in zip(chunk, wavs):
                x = resample_poly(np.asarray(w, np.float64), 16000, sr)
                x = np.concatenate([np.zeros(int(rng.uniform(0.05, 0.4) * 16000)), x,
                                    np.zeros(int(rng.uniform(0.1, 0.5) * 16000))])
                x += rng.standard_normal(len(x)) * 10 ** (-75 / 20)  # a faint floor instead of digital silence
                x *= 10 ** (rng.uniform(-26, -14) / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-9)
                sf.write(out / "clean" / f"{j['id']}.wav", np.clip(x, -1, 1).astype(np.float32), 16000,
                         subtype="PCM_16")
                tf.write(json.dumps({"id": j["id"], "text": j["text"], "speaker": j["voice"]}) + "\n")
                secs += len(x) / 16000
            tf.flush()  # no empty_cache: holding a steady block is safer on a shared Spark than freeing and re-taking it
            b += len(chunk)
            el = time.time() - t0
            print(f"{b}/{len(todo)}  {secs / 3600:.2f} h  {el / 60:.0f} min  ({secs / el:.0f}x real time)",
                  flush=True)
    print(f"done: {secs / 3600:.2f} h in {(time.time() - t0) / 60:.0f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("refs")
    a.add_argument("prep")
    a.add_argument("data")
    a.add_argument("out")
    v = sub.add_parser("voices")
    v.add_argument("refs")
    v.add_argument("out")
    v.add_argument("--designed", type=int, default=1000)
    v.add_argument("--designed-heldout", type=int, default=100)
    v.add_argument("--cloned", type=int, default=2000)
    v.add_argument("--seed", type=int, default=0)
    j = sub.add_parser("jobs")
    j.add_argument("voices")
    j.add_argument("lines")
    j.add_argument("test_lines")
    j.add_argument("out")
    j.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("render")
    r.add_argument("jobs")
    r.add_argument("voices")
    r.add_argument("out")
    r.add_argument("--refs", default=str(Path.home() / "ear/qwen/refs"), help="folder with the cloned voices' clips")
    r.add_argument("--models", default="/opt/voice/models", help="where the Qwen3-TTS models are")
    r.add_argument("--batch", type=int, default=96)
    r.add_argument("--chars", type=int, default=4000, help="at most this much text per batch (memory)")
    r.add_argument("--min-free", type=float, default=30, help="GB of free memory to wait for before each batch")
    r.add_argument("--yield-to", default="http://127.0.0.1:8188/queue,http://127.0.0.1:8288/queue",
                   help="ComfyUI queues to wait for before each batch")
    r.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"refs": refs, "voices": voices, "jobs": jobs, "render": render}[args.cmd](args)


if __name__ == "__main__":
    main()
