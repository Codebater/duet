"""Phone or laptop copies of labelled shards: the same clips through a random phone line (`channel.phone`) or as
DUET hears them on a laptop (`channel.laptop`, with `--echo` clips of DUET's voice as the canceller's residual), new
student features, the teacher's labels reused. They stay valid because the line keeps every sample in place, so the
80 ms frames do not move. `train360_007` -> `train360p_007` (phone) or `train360l_007` (laptop).

    python phone.py ~/ear/prep train360 librispeech ~/ear/data/LibriSpeech/train-clean-360
    python phone.py ~/ear/prep tts folder ~/ear/data/tts
    python phone.py ~/ear/prep qwen folder ~/ear/data/qwen/train --line laptop --echo ~/ear/data/tts/clean
"""
import argparse
import json
import shutil
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

from channel import laptop, phone
from features import RATE, features
from prep import FRAME, TAIL, calls_items, librispeech_items

LINE, ECHO = "phone", []
LABELS = ("feat_off", "top_ids", "top_lp", "top_off", "target", "target_off")  # data.KEYS minus the features


def load(item):
    uid, path, rows = item
    pcm, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == RATE, (path, rate)
    pcm = pcm.mean(axis=1)
    pcm = np.pad(pcm, (0, -len(pcm) % FRAME + TAIL * FRAME))  # the line's noise runs on through the silent tail
    rng = np.random.default_rng(zlib.crc32(uid.encode()))
    if LINE == "phone":
        x = phone(pcm, rng)
    else:
        echo = sf.read(ECHO[int(rng.integers(len(ECHO)))], dtype="float32")[0] if ECHO else None
        x = laptop(pcm, rng, echo)
    f = features(x, causal=True).astype(np.float16)
    assert len(f) == rows, (uid, len(f), rows)
    return f


def setup(line, echo):
    global LINE, ECHO
    LINE, ECHO = line, echo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("name", help="shard prefix to copy, e.g. train360")
    ap.add_argument("source", choices=["librispeech", "folder"])
    ap.add_argument("roots", nargs="+", help="where the audio is (several folders if prep.py ran per part)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--line", default="phone", choices=["phone", "laptop"])
    ap.add_argument("--echo", help="laptop: a folder of DUET-voice clips (16 kHz wav) for the echo residual")
    args = ap.parse_args()

    paths = {i[0]: i[2] for root in args.roots for i in (librispeech_items(root) if args.source == "librispeech"
                                                         else calls_items(root, "clean"))}
    stems = sorted(str(p.with_suffix("")) for p in Path(args.data).glob(f"{args.name}_[0-9]*.json"))
    t0, done = time.time(), 0
    echo = sorted(str(p) for p in Path(args.echo).glob("*.wav")) if args.echo else []
    suffix = {"phone": "p", "laptop": "l"}[args.line]
    with Pool(args.workers, initializer=setup, initargs=(args.line, echo)) as pool:
        for stem in stems:
            out = stem.replace(f"/{args.name}_", f"/{args.name}{suffix}_")
            if Path(f"{out}.json").exists():
                continue
            meta = json.loads(Path(f"{stem}.json").read_text(encoding="utf-8"))
            rows = np.diff(np.load(f"{stem}.feat_off.npy"))
            feats = pool.map(load, [(m["utt"], paths[m["utt"]], int(r)) for m, r in zip(meta, rows)], chunksize=8)
            np.save(f"{out}.feats.npy", np.concatenate(feats))
            for key in LABELS:
                shutil.copyfile(f"{stem}.{key}.npy", f"{out}.{key}.npy")
            shutil.copyfile(f"{stem}.json", f"{out}.json")  # last: a shard counts as done once its .json exists
            done += len(meta)
            print(f"  wrote {Path(out).name}: {len(meta)} clips  {done} total  {time.time() - t0:.0f} s", flush=True)
    print(f"done: {done} clips, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
