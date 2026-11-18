"""Where the Ear's word errors are, from evaluate.py's transcripts (`--hyps`): per split, the error rate of reference
words by kind (filler, number, rare in the training text, position in the clip), where the extra words go, and the
most common confusions.

    python errors.py ~/ear/runs/r1b/ear_best.hyps.jsonl ~/ear/prep --train train,train360,train500,tts
"""
import argparse
import collections
import json
from pathlib import Path

from data import words

FILLERS = {"um", "uh", "erm", "hmm", "mm", "ah"}
NUMBERS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
              "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
              "million".split())


def align(ref, hyp):
    """Levenshtein alignment -> per reference word None (right), the word it became, or "" (dropped); and the inserted
    words as (index of the reference word they precede, word)."""
    d = [[i + j if not (i and j) else 0 for j in range(len(hyp) + 1)] for i in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, ins, i, j = [None] * len(ref), [], len(ref), len(hyp)
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out[i - 1] = None if ref[i - 1] == hyp[j - 1] else hyp[j - 1]
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out[i - 1], i = "", i - 1
        else:
            ins.append((i, hyp[j - 1]))
            j -= 1
    return out, ins


def kinds(word, k, n, seen):
    out = ["all"]
    if word in FILLERS:
        out.append("filler")
    if word in NUMBERS or any(c.isdigit() for c in word):
        out.append("number")
    out.append("unseen" if seen[word] == 0 else "rare (<10)" if seen[word] < 10 else "common")
    out.append("1st word" if k == 0 else "2nd word" if k == 1 else "last word" if k == n - 1 else "middle")
    return out


def report(rows, seen):
    stats = collections.defaultdict(lambda: [0, 0])  # kind -> [reference words, errors]
    ins_at = collections.Counter()
    confusions, dropped, inserted = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        ref, hyp = words(r["truth"]), words(r["ear"])
        out, ins = align(ref, hyp)
        for k, (w, got) in enumerate(zip(ref, out)):
            for kind in kinds(w, k, len(ref), seen):
                stats[kind][0] += 1
                stats[kind][1] += got is not None
            if got == "":
                dropped[w] += 1
            elif got is not None:
                confusions[f"{w} -> {got}"] += 1
        for i, w in ins:
            ins_at["start" if i == 0 else "end" if i == len(ref) else "middle"] += 1
            inserted[w] += 1
    errors = stats["all"][1] + sum(ins_at.values())
    lines = [f"{len(rows)} clips, {stats['all'][0]} words, {errors} errors "
             f"(WER {100 * errors / stats['all'][0]:.1f}%: {sum(confusions.values())} wrong, "
             f"{sum(dropped.values())} dropped, {sum(ins_at.values())} extra)",
             f"  {'kind':<12}{'words':>7}{'share':>7}{'wrong':>8}{'of errors':>11}"]
    for kind in ("filler", "number", "unseen", "rare (<10)", "common", "1st word", "2nd word", "middle", "last word"):
        n, e = stats[kind]
        if n:
            lines.append(f"  {kind:<12}{n:>7}{100 * n / stats['all'][0]:>6.1f}%{100 * e / n:>7.1f}%"
                         f"{100 * e / errors:>10.1f}%")
    lines.append("  extra words: " + ", ".join(f"{k} {v} ({100 * v / errors:.1f}% of errors)"
                                               for k, v in ins_at.most_common()))
    for title, c in (("wrong", confusions), ("dropped", dropped), ("extra", inserted)):
        lines.append(f"  most often {title}: " + ", ".join(f"{k} x{v}" for k, v in c.most_common(12)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hyps")
    ap.add_argument("data", help="prep.py's shards, for how often each word occurs in the training text")
    ap.add_argument("--train", default="train,train360,train500,tts", help="shard prefixes the ear trained on")
    args = ap.parse_args()
    seen = collections.Counter()
    for name in args.train.split(","):
        for p in Path(args.data).glob(f"{name}_[0-9]*.json"):
            for m in json.loads(p.read_text(encoding="utf-8")):
                seen.update(words(m["teacher"]))
    rows = [json.loads(line) for line in Path(args.hyps).read_text(encoding="utf-8").splitlines()]
    for split in dict.fromkeys(r["split"] for r in rows):
        print(f"\n== {split}: " + report([r for r in rows if r["split"] == split], seen))


if __name__ == "__main__":
    main()
