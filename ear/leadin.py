"""The 310 test calls with a lead-in and a tail, the way the ear hears a caller inside DUET: it runs on the line all the
time, so a turn never starts from an empty state at the caller's first sound. 0.5 s before and after each call: digital
silence for the clean condition; for the phone condition, the same call through frontdesk-bench's own phone line
(`bench/augment.py`, the same random draws, so the speech part matches the official clip) with the line's noise running
on through the padding.

    python leadin.py ~/ear/data/calls ~/ear/data/callslead --augment ~/ear/fdb/augment.py
    python prep.py calls ~/ear/data/callslead ~/ear/prep --name callslead
    python prep.py calls ~/ear/data/callslead ~/ear/prep --calls-kind clean --name callsleadclean
"""
import argparse
import importlib.util
import json
import shutil
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf


def phone_padded(aug, x16, call_id, pad_s):
    """augment.phone step by step, plus pink noise at the line's level before and after the call."""
    rng = np.random.default_rng(zlib.crc32(call_id.encode()))
    x = aug.resample(x16, 16000, 8000)
    x = aug.sosfilt(aug.butter(4, [300, 3400], btype="band", fs=8000, output="sos"), x)
    x = aug.rms_norm(x, -20 + rng.uniform(-4, 4))
    x = aug.mulaw(x)
    noise = aug.pink(len(x), rng)
    snr = rng.uniform(12, 22)
    noise *= np.sqrt(np.mean(x ** 2) / (np.mean(noise ** 2) * 10 ** (snr / 10)))
    pad = int(pad_s * 8000)
    extra = aug.pink(2 * pad, np.random.default_rng(zlib.crc32(f"{call_id}/pad".encode())))
    extra *= np.sqrt(np.mean(noise ** 2) / np.mean(extra ** 2))
    return np.clip(aug.resample(np.concatenate([extra[:pad], x + noise, extra[pad:]]), 8000, 16000), -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calls", help="folder with truth.jsonl, clean/ and phone/")
    ap.add_argument("out")
    ap.add_argument("--augment", required=True, help="frontdesk-bench's bench/augment.py")
    ap.add_argument("--pad", type=float, default=0.5, help="seconds before and after")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("augment", args.augment)
    aug = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aug)

    src, out = Path(args.calls), Path(args.out)
    for kind in ("clean", "phone"):
        (out / kind).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "truth.jsonl", out / "truth.jsonl")
    pad16, gaps = int(args.pad * 16000), []
    for line in (src / "truth.jsonl").read_text(encoding="utf-8").splitlines():
        cid = json.loads(line)["id"]
        x16, sr = sf.read(src / "clean" / f"{cid}.wav", dtype="float64")
        assert sr == 16000, (cid, sr)
        sf.write(out / "clean" / f"{cid}.wav", np.concatenate([np.zeros(pad16), x16, np.zeros(pad16)]), 16000)
        y = phone_padded(aug, x16, cid, args.pad)
        sf.write(out / "phone" / f"{cid}.wav", y, 16000)
        official, _ = sf.read(src / "phone" / f"{cid}.wav", dtype="float64")
        mid = slice(800, len(official) - 800)  # 50 ms in from each end, clear of the resampler's edges
        diff = y[pad16:pad16 + len(official)][mid] - official[mid]
        gaps.append(10 * np.log10(np.sum(official[mid] ** 2) / (np.sum(diff ** 2) + 1e-20)))
    print(f"{len(gaps)} calls -> {out}; speech part vs the official phone clips: worst {min(gaps):.1f} dB, "
          f"median {np.median(gaps):.1f} dB signal-to-difference")


if __name__ == "__main__":
    main()
