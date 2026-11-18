"""VoxPopuli EN (European Parliament speech, CC0; Hugging Face facebook/voxpopuli, `en/*.parquet`) into the folder
layout prep.py and phone.py read: `<out>/<split>/clean/<id>.wav` (16 kHz) and `<out>/<split>/truth.jsonl`
{id, text, speaker}. Train, validation and test have different speakers; validation and test stay held out.

    python voxpopuli.py ~/ear/data/vox ~/ear/data/voxpopuli/validation-00000-of-00001.parquet
    python prep.py calls ~/ear/data/vox/validation ~/ear/prep --calls-kind clean --name voxdev

The 30 train parts one at a time as they download (labelling keeps pace), each in its own shard range:
    python voxpopuli.py ~/ear/data/vox/part07 ~/ear/data/voxpopuli/train-00007-of-00030.parquet
    python prep.py calls ~/ear/data/vox/part07/train ~/ear/prep --calls-kind clean --name vox --first-shard 70
    python phone.py ~/ear/prep vox folder ~/ear/data/vox/part*/train
"""
import argparse
import io
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

OUT = None


def convert(path):
    """One parquet file -> wavs; returns its truth rows and seconds of audio."""
    rows, seconds = [], 0.0
    split = Path(path).name.split("-")[0]
    clean = Path(OUT) / split / "clean"
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for r in batch.to_pylist():
            a = r["audio"]
            if a.get("bytes"):
                x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=True)
                x = x.mean(axis=1)
            else:
                x, sr = np.asarray(a["array"], np.float32), a["sampling_rate"]
            assert sr == 16000, (r["audio_id"], sr)
            sf.write(clean / f"{r['audio_id']}.wav", x, 16000, subtype="PCM_16")
            rows.append({"id": r["audio_id"], "text": r["raw_text"] or r["normalized_text"],
                         "speaker": str(r["speaker_id"]), "gold": r.get("is_gold_transcript")})
            seconds += len(x) / 16000
    return split, rows, seconds


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("files", nargs="+", help="downloaded en/*.parquet files; each split's truth.jsonl is rewritten")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    OUT, files = args.out, sorted(args.files)
    for split in {Path(f).name.split("-")[0] for f in files}:
        (Path(OUT) / split / "clean").mkdir(parents=True, exist_ok=True)
    truth, hours = {}, {}
    with Pool(args.workers) as pool:
        for split, rows, seconds in pool.imap_unordered(convert, files):
            truth.setdefault(split, []).extend(rows)
            hours[split] = hours.get(split, 0) + seconds / 3600
            print(f"  {split}: {len(truth[split])} clips, {hours[split]:.1f} h", flush=True)
    for split, rows in truth.items():
        rows.sort(key=lambda r: r["id"])
        (Path(OUT) / split / "truth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(f"{split}: {len(rows)} clips, {hours[split]:.1f} h, {len({r['speaker'] for r in rows})} speakers")


if __name__ == "__main__":
    main()
