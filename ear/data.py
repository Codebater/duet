"""Loading prep.py's shards, batching by length, greedy CTC decoding and word error rate."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

TAIL = 4  # prep.py: frames of silence after each utterance


KEYS = ("feats", "feat_off", "top_ids", "top_lp", "top_off", "target", "target_off")


def load_shard(stem):
    """One shard: `<stem>.<key>.npy` (features memory-mapped: 960 h is ~115 GB) or the older `<stem>.npz`."""
    if Path(f"{stem}.feats.npy").exists():
        return {k: np.load(f"{stem}.{k}.npy", mmap_mode="r" if k == "feats" else None) for k in KEYS}
    z = np.load(f"{stem}.npz")
    return {k: z[k] for k in KEYS}


class Split:
    """The utterances of one or more shard prefixes (e.g. "train", "train360"), sliced from their shards by offsets."""

    def __init__(self, root, names):
        names = [names] if isinstance(names, str) else list(names)
        stems = sorted(str(p.with_suffix("")) for n in names for p in Path(root).glob(f"{n}_[0-9]*.json"))
        assert stems, f"no {names} shards in {root}"
        self.shards = [load_shard(s) for s in stems]
        self.meta = [m for s in stems for m in json.loads(Path(f"{s}.json").read_text(encoding="utf-8"))]
        self.shard_of = np.concatenate([np.full(len(s["feat_off"]) - 1, k) for k, s in enumerate(self.shards)])
        self.local = np.concatenate([np.arange(len(s["feat_off"]) - 1) for s in self.shards])
        self.rows = np.concatenate([np.diff(s["feat_off"]) for s in self.shards])
        self.frames = np.concatenate([np.diff(s["top_off"]) for s in self.shards])
        assert len(self.meta) == len(self.rows)

    def __len__(self):
        return len(self.meta)

    def hours(self):
        return sum(m["seconds"] for m in self.meta) / 3600

    def item(self, i):
        s, j = self.shards[self.shard_of[i]], self.local[i]
        return (s["feats"][s["feat_off"][j]:s["feat_off"][j + 1]], s["top_ids"][s["top_off"][j]:s["top_off"][j + 1]],
                s["top_lp"][s["top_off"][j]:s["top_off"][j + 1]], s["target"][s["target_off"][j]:s["target_off"][j + 1]])

    def sample(self, every=97, shards=24):
        """Every `every`-th feature row of `shards` evenly spread shards (for the input mean and std). Not all of them:
        the 2,364 h mix is 229 GB of features, more than RAM, and a strided read of it all thrashes the page cache."""
        pick = np.unique(np.linspace(0, len(self.shards) - 1, min(shards, len(self.shards))).round().astype(int))
        return np.concatenate([np.asarray(self.shards[k]["feats"][::every], np.float32) for k in pick])

    def batches(self, max_rows, rng=None):
        """Index lists with (longest rows x count) <= max_rows, similar lengths together; shuffled if rng."""
        order = np.argsort(self.rows, kind="stable")
        if rng is not None:  # shuffle within ~equal lengths, so batches differ each epoch
            order = order[np.argsort(self.rows[order] // 40 * 1_000_000 + rng.integers(0, 1_000_000, len(order)),
                                     kind="stable")]
        out, cur, longest = [], [], 0
        for i in order:
            longest2 = max(longest, self.rows[i])
            if cur and longest2 * (len(cur) + 1) > max_rows:
                out.append(cur)
                cur, longest2 = [], self.rows[i]
            cur.append(int(i))
            longest = longest2
        if cur:
            out.append(cur)
        if rng is not None:
            rng.shuffle(out)
        return out

    def batch(self, idx, device):
        """-> feats [B, rows, 320] f32, frames per item (student), teacher top ids/log-probs [B, Tt, 16] (+ frames),
        flat CTC targets + lengths."""
        rows, tt = self.rows[idx], self.frames[idx]
        feats = np.zeros((len(idx), rows.max(), 320), np.float16)
        top_ids = np.zeros((len(idx), tt.max(), 16), np.int64)
        top_lp = np.full((len(idx), tt.max(), 16), -1e4, np.float32)
        tgt = []
        for j, i in enumerate(idx):
            f, ids, lp, target = self.item(i)
            feats[j, :rows[j]], top_ids[j, :tt[j]], top_lp[j, :tt[j]] = f, ids, lp
            tgt.append(target)
        t = lambda a, dt=None: torch.from_numpy(np.asarray(a)).to(device, non_blocking=True)  # noqa: E731
        return (t(feats).float(), t(rows // 4), t(top_ids), t(top_lp), t(tt),
                t(np.concatenate(tgt).astype(np.int64) if len(tgt) else np.zeros(0, np.int64)),
                t(np.array([len(x) for x in tgt], np.int64)))


def greedy(logits, frames):
    """[B, T, V] logits, valid frames per item -> token lists (CTC collapse, blank 0 dropped)."""
    ids = logits.argmax(-1).cpu().numpy()
    out = []
    for row, n in zip(ids, frames.tolist() if torch.is_tensor(frames) else frames):
        row = row[:n]
        out.append([int(t) for k, t in enumerate(row) if t != 0 and (k == 0 or t != row[k - 1])])
    return out


_WORD = re.compile(r"[a-z0-9']+")


def words(text):
    return _WORD.findall(text.lower().replace("’", "'"))


def edits(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1]


def wer(refs, hyps):
    """Corpus word error rate (%), after lower-casing and dropping punctuation."""
    errors = total = 0
    for r, h in zip(refs, hyps):
        r, h = words(r), words(h)
        errors += edits(r, h)
        total += len(r)
    return 100.0 * errors / max(total, 1)


def to_npy(root):
    """Rewrite older `<stem>.npz` shards as `<stem>.<key>.npy`, so their features are memory-mapped too."""
    for z in sorted(Path(root).glob("*.npz")):
        stem = str(z.with_suffix(""))
        if Path(f"{stem}.feats.npy").exists():
            continue
        arrays = np.load(z)
        for k in KEYS:
            np.save(f"{stem}.{k}.npy", arrays[k])
        print("converted", z.name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["to-npy"])
    ap.add_argument("root")
    args = ap.parse_args()
    to_npy(args.root)
