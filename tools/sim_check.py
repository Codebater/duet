"""Check a simulated call (runs/sim-*) for the failures the first live call had.

    python tools/sim_check.py runs/sim-long-talker-v4 [more runs...]

- **talked over**: seconds DUET's own sound played while a caller line that had the floor ("after" lines) was being
  said, from 0.3 s into the line to its end. Lines said over DUET ("over") are excluded: there DUET spoke first, and
  how fast it yields is the barge-in test.
- the friend's backchannels ("listening" events) are left out: they are meant to go into the caller's pauses.
- **echo leak**: any run of 5 words from DUET's replies inside a caller turn's transcript (DUET heard itself).
- **latency**: the reply latencies the conductor logged.
Exit code 1 if DUET talked over the caller for more than 1 s in total (one yield takes up to ~0.9 s: the barge-in
rule waits for 3 words or 0.9 s of talk) or any echo leaked.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def check(run):
    run = Path(run)
    call = json.load(open(run / "call.json", encoding="utf-8"))
    stereo, rate = sf.read(run / "call_mic_L_duet_R.wav", dtype="float32")
    out = stereo[:, 1]
    hop = rate // 100
    n = len(out) // hop
    duet_on = 10 * np.log10((out[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12) > -45  # 10 ms frames
    for e in call["events"]:  # the friend's "mm-hm" in a pause is meant to overlap: not talking over them
        if e["kind"] == "listening":
            duet_on[int(e["t"] * 100):int((e["t"] + 0.9) * 100)] = False
    over = 0.0
    for line in call["caller"]:
        if line["when"] != "after" or "seconds" not in line:  # runs before 2026-11-03 have no line lengths
            continue
        a, b = int((line["t"] + 0.3) * 100), int((line["t"] + line["seconds"]) * 100)
        over += float(duet_on[a:b].sum()) / 100
    said = [e["text"].split('"')[1] for e in call["events"] if e["kind"] == "duet"]
    grams = {tuple(w[i:i + 5]) for s in said for w in [words(s)] for i in range(len(w) - 4)}
    leaks = []
    for e in call["events"]:
        if e["kind"] == "caller":
            w = words(e["text"].split('"')[1])
            hit = [g for g in (tuple(w[i:i + 5]) for i in range(len(w) - 4)) if g in grams]
            if hit:
                leaks.append((e["t"], " ".join(hit[0])))
    lat = call["latencies_ms"]
    print(f"{run.name:28s} talked over the caller {over:4.1f} s | echo leaks {len(leaks)}"
          + "".join(f" [{t:.1f} s: \"{g}\"]" for t, g in leaks)
          + f" | latency ms {lat} (median {int(np.median(lat)) if lat else '-'})")
    return over <= 1.0 and not leaks


if __name__ == "__main__":
    ok = all([check(r) for r in sys.argv[1:]])
    sys.exit(0 if ok else 1)
