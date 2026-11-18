"""Ear step 0: label speech with TurboCTC (the teacher) and store what the student trains on.

Per utterance: the student's causal features [rows, 320] (fp16; audio rounded up to 80 ms, then 320 ms of silence so the
student can answer a little late), the teacher's top-16 CTC log-probs per 80 ms frame, and the teacher's greedy tokens
(the CTC target). Utterances over 10 s are labelled in pieces cut at the quietest 100 ms between 6 and 10 s (as
`duet.asr.Asr.long` does), each cut on an 80 ms boundary so the pieces' teacher frames line up with the whole utterance.

    python prep.py librispeech ~/ear/data/LibriSpeech/train-clean-100 ~/ear/prep --hold-out 8
    python prep.py librispeech ~/ear/data/LibriSpeech/train-other-500 ~/ear/prep --train-name train500 --dev-name devother
    python prep.py calls ~/ear/data/calls ~/ear/prep
    python prep.py calls ~/ear/data/tts ~/ear/prep --calls-kind clean --name tts   (lines rendered by tts.py)
"""
import argparse
import json
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from tokenizers import Tokenizer

from features import RATE, features

FRAME = 1280      # samples per 80 ms CTC frame (4 feature rows)
ROWS = 512        # TurboCTC's static input: 10.24 s
TOPK = 16
TAIL = 4          # frames of silence after each utterance
SHARD = 2000      # utterances per shard
_WORD = re.compile(r"[a-z0-9']+")
BLOCK = 4096      # clips in flight between the CPU workers and the teacher


def cut_points(pcm, piece_s=10.0, min_s=6.0):
    """Piece boundaries (multiples of FRAME) for a clip whose length is a multiple of FRAME."""
    limit, lo_off, step = int(piece_s * RATE), int(min_s * RATE), RATE // 10
    bounds, s = [0], 0
    while len(pcm) - s > limit:
        seg = pcm[s + lo_off:s + limit]
        n = len(seg) // step
        quiet = int(np.argmin((seg[:n * step].reshape(n, step) ** 2).mean(axis=1)))
        cut = round((s + lo_off + quiet * step + step // 2) / FRAME) * FRAME
        s = min(max(cut, s + FRAME), s + limit)
        bounds.append(s)
    bounds.append(len(pcm))
    return bounds


def load(item):
    """CPU side, in a worker: audio -> teacher inputs per piece + student features."""
    uid, spk, path, truth = item
    pcm, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == RATE, (path, rate)
    pcm = pcm.mean(axis=1)
    pcm = np.pad(pcm, (0, -len(pcm) % FRAME))
    bounds = cut_points(pcm)
    teacher = [features(pcm[a:b]) for a, b in zip(bounds, bounds[1:])]  # padded to 512 rows only when run
    student = features(np.concatenate([pcm, np.zeros(TAIL * FRAME, np.float32)]), causal=True).astype(np.float16)
    assert len(student) == (len(pcm) // FRAME + TAIL) * 4, (path, len(student), len(pcm))
    return uid, spk, len(pcm) / RATE, truth, teacher, student


class Teacher:
    def __init__(self, model_dir, batch):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(Path(model_dir) / "turboctc_fp16.onnx"), so,
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert self.session.get_providers()[0] == "CUDAExecutionProvider", self.session.get_providers()
        dim = self.session.get_inputs()[0].shape[0]
        self.batch = batch if not isinstance(dim, int) else dim
        self.tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))

    def __call__(self, pieces):
        """[features [rows <= 512, 320]] -> per piece: top ids [frames, 16], top log-probs, greedy tokens."""
        out = []
        for i in range(0, len(pieces), self.batch):
            chunk = pieces[i:i + self.batch]
            x = np.zeros((len(chunk), ROWS, 320), np.float32)
            mask = np.zeros((len(chunk), ROWS), np.int64)
            for j, f in enumerate(chunk):
                x[j, :len(f)], mask[j, :len(f)] = f, 1
            logits = self.session.run(["logits"], {"input_features": x, "attention_mask": mask})[0]
            for j, n in enumerate(len(f) for f in chunk):
                z = logits[j, :n // 4].astype(np.float32)
                lp = z - z.max(-1, keepdims=True)
                lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
                ids = np.argpartition(-lp, TOPK, axis=-1)[:, :TOPK]
                top = np.take_along_axis(lp, ids, -1)
                order = np.argsort(-top, axis=-1)
                ids, top = np.take_along_axis(ids, order, -1), np.take_along_axis(top, order, -1)
                g = lp.argmax(-1)
                keep = [int(t) for k, t in enumerate(g) if t != 0 and (k == 0 or t != g[k - 1])]
                out.append((ids.astype(np.int16), top.astype(np.float16), keep))
        return out


class ShardWriter:
    def __init__(self, out, name, first=0):
        self.out, self.name, self.n = Path(out), name, first
        self.out.mkdir(parents=True, exist_ok=True)
        self.rows = []

    def add(self, row):
        self.rows.append(row)
        if len(self.rows) >= SHARD:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        feats = [r["feats"] for r in self.rows]
        tops = [r["top_ids"] for r in self.rows]
        tgts = [np.asarray(r["target"], np.int16) for r in self.rows]
        off = lambda xs: np.concatenate([[0], np.cumsum([len(x) for x in xs])]).astype(np.int64)  # noqa: E731
        stem = self.out / f"{self.name}_{self.n:03d}"
        arrays = {"feats": np.concatenate(feats), "feat_off": off(feats), "top_ids": np.concatenate(tops),
                  "top_lp": np.concatenate([r["top_lp"] for r in self.rows]), "top_off": off(tops),
                  "target": np.concatenate(tgts) if tgts else np.zeros(0, np.int16), "target_off": off(tgts)}
        for key, a in arrays.items():  # one .npy per array: training memory-maps the features
            np.save(f"{stem}.{key}.npy", a)
        meta = [{k: r[k] for k in ("utt", "speaker", "seconds", "truth", "teacher")} for r in self.rows]
        stem.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
        print(f"  wrote {stem.name}: {len(self.rows)} utterances", flush=True)
        self.rows, self.n = [], self.n + 1


_SAME = {"i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are", "it's": "it is",
         "that's": "that is", "what's": "what is", "there's": "there is", "he's": "he is", "she's": "she is",
         "how's": "how is", "where's": "where is", "who's": "who is", "here's": "here is", "let's": "let us",
         "don't": "do not", "doesn't": "does not", "didn't": "did not", "can't": "can not", "cannot": "can not",
         "won't": "will not", "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
         "haven't": "have not", "hasn't": "has not", "couldn't": "could not", "wouldn't": "would not",
         "shouldn't": "should not", "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
         "i'd": "i would", "you'd": "you would", "i'll": "i will", "you'll": "you will", "we'll": "we will",
         "it'll": "it will", "that'll": "that will", "favourite": "favorite", "colour": "color", "centre": "center",
         "realise": "realize", "organise": "organize", "haha": "ha ha", "okay": "ok"}


def wer(truth, hyp, same=False):
    """Word error rate (%) of one transcript, as data.wer counts it (no torch here). same=True first writes both the
    same way (contractions spelt out, US spelling): the teacher spells out "I'm" as "I am" for some voices."""
    r, h = (_WORD.findall(t.lower().replace("’", "'")) for t in (truth, hyp))
    if same:
        r, h = (" ".join(_SAME.get(w, w) for w in ws).split() for ws in (r, h))
    d = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, b in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (a != b))
    return 100.0 * d[-1] / max(len(r), 1)


def librispeech_items(root):
    items = []
    for trans in sorted(Path(root).glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            uid, text = line.split(" ", 1)
            items.append((uid, uid.split("-")[0], str(trans.parent / f"{uid}.flac"), text))
    return items


def calls_items(root, kind="phone"):
    truth = [json.loads(line) for line in (Path(root) / "truth.jsonl").read_text(encoding="utf-8").splitlines()]
    return [(t["id"], t.get("speaker", "calls"), str(Path(root) / kind / f"{t['id']}.wav"), t["text"]) for t in truth]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["librispeech", "calls"])
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--teacher", default=str(Path.home() / "ear/teacher"))
    ap.add_argument("--hold-out", type=int, default=8, help="last N speakers become the dev set")
    ap.add_argument("--train-name", default="train", help="shard prefix for the training clips")
    ap.add_argument("--dev-name", default="dev", help="shard prefix for the held-out speakers")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calls-kind", default="phone", choices=["phone", "clean"], help="calls: which audio condition")
    ap.add_argument("--name", help="calls: shard prefix (default calls / callsclean), e.g. tts for rendered lines")
    ap.add_argument("--first-shard", type=int, default=0, help="calls: number of the first shard, to add to a prefix")
    ap.add_argument("--max-wer", type=float, default=0, help="drop clips the teacher hears at more than this WER "
                    "(%%, spelling style aside) against their script: rendered lines where the TTS skipped or added words")
    args = ap.parse_args()

    if args.source == "librispeech":
        items = librispeech_items(args.root)
        speakers = sorted({i[1] for i in items}, key=int)
        dev = set(speakers[len(speakers) - args.hold_out:])  # not [-n:]: n = 0 would hold out everyone
        writers = {"train": ShardWriter(args.out, args.train_name), "dev": ShardWriter(args.out, args.dev_name)}
        split = lambda spk: "dev" if spk in dev else "train"  # noqa: E731
    else:
        items = calls_items(args.root, args.calls_kind)
        name = args.name or ("calls" if args.calls_kind == "phone" else f"calls{args.calls_kind}")
        writers = {"calls": ShardWriter(args.out, name, args.first_shard)}
        split = lambda spk: "calls"  # noqa: E731
    if args.limit:
        items = items[:args.limit]
    teacher = Teacher(args.teacher, args.batch)
    print(f"{len(items)} utterances, teacher batch {teacher.batch}", flush=True)

    t0, done, seconds, group, dropped = time.time(), 0, 0.0, [], []

    def label(group):
        pieces = [p for g in group for p in g[4]]
        labels = iter(teacher(pieces))
        for uid, spk, secs, truth, tpieces, student in group:
            got = [next(labels) for _ in tpieces]
            top_ids = np.concatenate([g[0] for g in got])
            assert len(top_ids) + TAIL == len(student) // 4, (uid, len(top_ids), len(student))
            target = [t for g in got for t in g[2]]
            text = " ".join(t for t in (teacher.tok.decode(g[2], skip_special_tokens=True).strip() for g in got) if t)
            if args.max_wer and wer(truth, text, same=True) > args.max_wer:
                dropped.append(uid)
                continue
            writers[split(spk)].add({"utt": uid, "speaker": spk, "seconds": secs, "truth": truth, "teacher": text,
                                     "feats": student, "top_ids": top_ids,
                                     "top_lp": np.concatenate([g[1] for g in got]), "target": target})

    # imap has no backpressure: the workers outrun the teacher, and on 360 h their finished clips filled 119 GB of
    # RAM (OOM kill). Feeding BLOCK clips at a time caps the backlog at one block.
    with Pool(args.workers) as pool:
        for start in range(0, len(items), BLOCK):
            for row in pool.imap(load, items[start:start + BLOCK], chunksize=8):
                group.append(row)
                done += 1
                seconds += row[2]
                if len(group) >= 64:
                    label(group)
                    group = []
                if done % 1000 == 0:
                    el = time.time() - t0
                    print(f"{done}/{len(items)}  {seconds / 3600:.1f} h audio  {el:.0f} s  "
                          f"({seconds / el:.0f}x real time)", flush=True)
        if group:
            label(group)
    for w in writers.values():
        w.flush()
    print(f"done: {done} utterances, {seconds / 3600:.2f} h, {time.time() - t0:.0f} s"
          + (f"; {len(dropped)} dropped by --max-wer {args.max_wer:g}" if args.max_wer else ""), flush=True)


if __name__ == "__main__":
    main()
