"""Find the mood directions in Supertonic-3's style space, then check they do what they claim on our voice.

    .venv/Scripts/python tools/calibrate_mood_voice.py [--voice orion] [--out models/mood-voice/directions.npz]

1. Render three sentences with every style we have (10 Supertone presets + our 7 Ivy-builder voices), measure the
   median pitch and the pitch range (10th-90th percentile, semitones) with duet.prosody.
2. Ridge-regress the styles on the two measures (z-scored): each coefficient vector is a direction in style space
   per one standard deviation of that measure across voices. `style_ttl` and `style_dp` get their own parts.
3. Verify on --voice: render with each direction at -1, 0, +1 and measure again, plus ASR on each render. A
   direction is kept only if the measure moves the right way at both steps and the words survive.
Writes the npz (pitch_ttl, pitch_dp, range_ttl, range_dp; a direction that fails is written as zeros) and a
REPORT.md next to it. Never overwrites: an existing --out stops the run.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from duet.asr import Asr  # noqa: E402
from duet.prosody import utterance_features  # noqa: E402
from duet.voice import Style, Voice  # noqa: E402

TEXTS = ["Hi there, thanks for calling. How can I help you today?",
         "A complete online store is ready in about a week, with payments and shipping set up.",
         "Of course! You can book a consultation on our website whenever it suits you."]


def measure(voice, style):
    pitch, rng = [], []
    for t in TEXTS:
        base = voice.base
        voice.base = style
        w = resample_poly(voice.synth(t), 1, 3).astype(np.float32)
        voice.base = base
        f = utterance_features(w)
        pitch.append(f["f0_st_abs"])
        rng.append(f["f0_range_st"])
    return float(np.mean(pitch)), float(np.mean(rng))


def words(s):
    return [w for w in "".join(c.lower() if c.isalnum() or c == " " else " " for c in s).split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="orion")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "models/mood-voice/directions.npz")
    ap.add_argument("--ridge", type=float, default=1.0)
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; pick a new --out")
    v = Voice(a.voice)
    names = sorted(v.styles)
    meas = {}
    for n in names:
        meas[n] = measure(v, v.styles[n])
        print(f"{n:8s} pitch {100 * 2 ** (meas[n][0] / 12):5.0f} Hz  range {meas[n][1]:4.1f} st", flush=True)
    X = np.array([meas[n] for n in names])
    mu, sd = X.mean(0), X.std(0)
    Z = (X - mu) / sd
    S = np.stack([np.concatenate([v.styles[n].ttl.ravel(), v.styles[n].dp.ravel()]) for n in names])
    Sc = S - S.mean(0)
    B = np.linalg.solve(Z.T @ Z + a.ridge * np.eye(2), Z.T @ Sc)  # [2, D]: per 1 sd of pitch, of range
    n_ttl = v.base.ttl.size
    dirs = {}
    for k, key in enumerate(("pitch", "range")):
        dirs[f"{key}_ttl"] = B[k, :n_ttl].reshape(v.base.ttl.shape).astype(np.float32)
        dirs[f"{key}_dp"] = B[k, n_ttl:].reshape(v.base.dp.shape).astype(np.float32)

    asr = Asr()
    lines = ["# Mood directions in style space", "", f"Fitted on {len(names)} styles; verified on `{a.voice}`.", "",
             "| direction | step | pitch Hz | range st | words kept |", "|---|---|---|---|---|"]
    keep = {}
    base_style = v.base
    for key, col in (("pitch", 0), ("range", 1)):
        rows = []
        for step in (-1.0, 0.0, 1.0):
            st = Style(base_style.ttl + step * dirs[f"{key}_ttl"], base_style.dp + step * dirs[f"{key}_dp"])
            p, r = measure(v, st)
            v.base = st
            heard = words(asr(resample_poly(v.synth(TEXTS[1]), 1, 3).astype(np.float32)))
            v.base = base_style
            ref = words(TEXTS[1])
            kept = sum(w in heard for w in ref) / len(ref)
            rows.append((p, r, kept))
            lines.append(f"| {key} | {step:+.0f} | {100 * 2 ** (p / 12):.0f} | {r:.1f} | {kept:.0%} |")
            print(f"{key} {step:+.0f}: pitch {100 * 2 ** (p / 12):.0f} Hz range {r:.1f} st words {kept:.0%}", flush=True)
        m = [x[col] for x in rows]
        ok = m[0] < m[1] < m[2] and min(x[2] for x in rows) >= 0.85
        keep[key] = ok
        if not ok:
            dirs[f"{key}_ttl"] *= 0
            dirs[f"{key}_dp"] *= 0
    lines += ["", *[f"- **{k}**: {'kept' if ok else 'REJECTED (not monotonic or words lost); written as zeros'}"
                    for k, ok in keep.items()]]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **dirs)
    (a.out.parent / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(keep, "->", a.out)


if __name__ == "__main__":
    main()
