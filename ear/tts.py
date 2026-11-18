"""Conversational speech for the ear: our own everyday lines, spoken by Supertonic 3 in many mixed voices.

The 310 test calls were rendered with a different engine (Qwen3-TTS VoiceDesign, 12 voices), so the ear cannot score
on them by learning one engine's voices. TurboCTC labels the renders like any other audio (prep.py), and phone.py
makes their phone copies.

    python tts.py lines lines.jsonl --exclude ../../frontdesk-bench/train/evalset/truth.jsonl a.jsonl b.jsonl ...
    python tts.py render lines.jsonl ~/ear/data/tts --supertonic ~/ear/supertonic3      (spark-1, CUDA)

`lines`: every "text" field, duplicates dropped, and anything with word overlap (Jaccard) >= 0.6 with a test call
dropped (frontdesk-bench's contamination guard). `render`: each line twice, in two of 300 voices; a voice is a random
mix of two or three of Supertonic's ten styles (the style space is smooth, so a mix is a new, stable speaker), at a
random speed. Writes `clean/<id>.wav` (16 kHz) and `truth.jsonl` {id, text, speaker}.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_WORD = re.compile(r"[a-z0-9']+")


def words(text):
    return set(_WORD.findall(text.lower().replace("’", "'")))


def lines(args):
    tests = [words(json.loads(l)["text"]) for l in Path(args.exclude).read_text(encoding="utf-8").splitlines()]
    seen, out, dropped = set(), [], {"duplicate": 0, "near_test": 0, "empty": 0}
    for f in args.inputs:
        for l in Path(f).read_text(encoding="utf-8").splitlines():
            text = re.sub(r"\s+", " ", json.loads(l)["text"]).strip()
            key = " ".join(_WORD.findall(text.lower().replace("’", "'")))
            ws = set(key.split())
            if not ws:
                dropped["empty"] += 1
            elif key in seen:
                dropped["duplicate"] += 1
            elif any(len(ws & t) / len(ws | t) >= 0.6 for t in tests):
                dropped["near_test"] += 1
            else:
                seen.add(key)
                out.append({"id": f"l{len(out):06d}", "text": text})
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    print(f"{len(out)} lines -> {args.out}; dropped {dropped}")


def voices(styles, n, rng):
    """n speakers: Dirichlet mixes of 2-3 of the stock styles (ttl [k, 50, 256], dp [k, 8, 16])."""
    ttl, dp = styles
    out = []
    for _ in range(n):
        pick = rng.choice(len(ttl), size=int(rng.integers(2, 4)), replace=False)
        w = rng.dirichlet(np.ones(len(pick)))
        out.append((np.tensordot(w, ttl[pick], 1), np.tensordot(w, dp[pick], 1)))
    return out


def render(args):
    import onnxruntime as ort
    import soundfile as sf
    from scipy.signal import resample_poly
    sys.path.insert(0, args.supertonic)
    import helper  # Supertone's reference inference code, shipped with the model (OpenRAIL-M)

    onnx = str(Path(args.supertonic) / "onnx")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    tts = helper.TextToSpeech(helper.load_cfgs(onnx), helper.load_text_processor(onnx),
                              *helper.load_onnx_all(onnx, so, providers))
    assert tts.dp_ort.get_providers()[0] == "CUDAExecutionProvider", tts.dp_ort.get_providers()
    stock = sorted((Path(args.supertonic) / "voice_styles").glob("*.json"))
    base = helper.load_voice_style([str(p) for p in stock])
    rng = np.random.default_rng(args.seed)
    speakers = voices((base.ttl, base.dp), args.voices, rng)

    rows = [json.loads(l) for l in Path(args.lines).read_text(encoding="utf-8").splitlines()][:args.limit or None]
    jobs = [(f"{r['id']}_{k}", r["text"], int(v)) for r in rows
            for k, v in enumerate(rng.choice(len(speakers), size=args.per_line, replace=False))]
    jobs.sort(key=lambda j: len(j[1]))  # similar lengths per batch: less padding
    out = Path(args.out)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    truth, t0, seconds = [], time.time(), 0.0
    for b in range(0, len(jobs), args.batch):
        chunk = jobs[b:b + args.batch]
        style = helper.Style(np.stack([speakers[v][0] for _, _, v in chunk]).astype(np.float32),
                             np.stack([speakers[v][1] for _, _, v in chunk]).astype(np.float32))
        speed = float(rng.uniform(0.85, 1.25))
        wav, dur = tts.batch([t for _, t, _ in chunk], ["en"] * len(chunk), style, int(rng.choice([6, 8, 10])), speed)
        for (uid, text, v), w, d in zip(chunk, wav, dur):
            x = resample_poly(w[:int(d * tts.sample_rate)].astype(np.float64), 160, 441)
            x = np.concatenate([np.zeros(int(rng.uniform(0.05, 0.4) * 16000)), x,
                                np.zeros(int(rng.uniform(0.1, 0.5) * 16000))])
            x += rng.standard_normal(len(x)) * 10 ** (-75 / 20)  # a faint floor instead of digital silence
            x *= 10 ** (rng.uniform(-26, -14) / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-9)
            sf.write(out / "clean" / f"{uid}.wav", np.clip(x, -1, 1).astype(np.float32), 16000)
            truth.append({"id": uid, "text": text, "speaker": f"v{v:03d}"})
            seconds += len(x) / 16000
        if (b // args.batch) % 50 == 0:
            el = time.time() - t0
            print(f"{b + len(chunk)}/{len(jobs)}  {seconds / 3600:.2f} h  {el:.0f} s  ({seconds / el:.0f}x real time)",
                  flush=True)
    (out / "truth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in truth), encoding="utf-8")
    print(f"done: {len(truth)} clips, {seconds / 3600:.2f} h, {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("lines")
    a.add_argument("out")
    a.add_argument("inputs", nargs="+")
    a.add_argument("--exclude", required=True, help="the test calls' truth.jsonl")
    r = sub.add_parser("render")
    r.add_argument("lines")
    r.add_argument("out")
    r.add_argument("--supertonic", default=str(Path.home() / "ear/supertonic3"))
    r.add_argument("--voices", type=int, default=300)
    r.add_argument("--per-line", type=int, default=2)
    r.add_argument("--batch", type=int, default=32)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    lines(args) if args.cmd == "lines" else render(args)


if __name__ == "__main__":
    main()
