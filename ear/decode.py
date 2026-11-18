"""Rung 1c: CTC beam search with a word language model on the Ear's output, to measure how much of its error is
decoding. Greedy decoding takes each 80 ms frame's best token on its own; the beam keeps the best `--width` transcripts
and scores every finished word with an n-gram model (lm.py): alpha x ln P(word | the 3 words before) + beta.

Two language models, neither trained on test text:
- books: the LibriSpeech transcripts of the 960 h the ear trained on;
- mix: books plus our conversational lines (tts.py, already guarded against the test calls), mixed at the weight with
  the lowest perplexity on a tenth of the lines held out.
alpha and beta are picked on the held-out audiobook speakers, then used unchanged on the calls. The best setting on the
calls themselves is printed separately as a ceiling, not a result.

    python decode.py ~/ear/prep ~/ear/runs/r1b/ear_best.pt --lines ~/ear/data/tts_lines.jsonl
"""
import argparse
import heapq
import json
import math
import random
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from data import Split, wer, words
from lm import BOS, EOS, KneserNey, Mix, perplexity
from model import Ear, EarConfig

NEG = -1e30
TOPK = 8
G = {}  # set before the worker pool forks: token table, per-clip top-k log-probs, truths, language models


def lse(a, b):
    if a < b:
        a, b = b, a
    return a if b <= NEG else a + math.log1p(math.exp(b - a))


def token_table(tok):
    """Per token id: its text without the word marker, and whether it starts a word."""
    text, start = [], []
    for i in range(tok.get_vocab_size()):
        p = tok.id_to_token(i) or ""
        if p.startswith("<") and p.endswith(">"):  # special tokens, and byte fallback like <0x41>
            byte = int(p[3:-1], 16) if p.startswith("<0x") else 128
            text.append(chr(byte) if byte < 128 else "")
            start.append(False)
        else:
            text.append(p.lstrip("▁"))
            start.append(p.startswith("▁"))
    return text, start


def text_of(toks):
    return "".join((" " if G["start"][c] else "") + G["text"][c] for c in toks).strip()


def score_words(ctx, part, alpha, beta, lm):
    add = 0.0
    for w in words(part):
        add += alpha * lm.logp(ctx, w) + beta
        ctx = (ctx + (w,))[1 - lm.order:]
    return ctx, add


def greedy(blank, ids, lps):
    out, prev = [], 0
    for t in range(len(blank)):
        c = int(ids[t][0]) if lps[t][0] > blank[t] else 0
        if c and c != prev:
            out.append(c)
        prev = c
    return out


def beam(blank, ids, lps, lm, alpha, beta, width, prune=10.0):
    """CTC prefix beam search. Prefixes are nodes of a tree (parent, token), each with its language-model state:
    (last words, the word being spelled, LM score of the finished words, penalty for the word being spelled). A word is
    scored once the next one starts, so until then a spelling that can no longer become a known word ("thecat", "ation")
    ranks as an unknown word; without that, prefixes that glue words together dodge the LM and crowd out the rest."""
    text, start = G["text"], G["start"]
    parent, token, state, child = [-1], [-1], [((BOS,), "", 0.0, 0.0)], {}
    beams = {0: [0.0, NEG]}  # node -> [log P(ends in blank), log P(ends in a token)]
    for t in range(len(blank)):
        b = float(blank[t])
        floor = max(b, float(lps[t][0])) - prune
        cands = [(int(c), float(l)) for c, l in zip(ids[t], lps[t]) if l > floor]
        nxt = {}
        for p, (pb, pnb) in beams.items():
            tot = lse(pb, pnb)
            e = nxt.setdefault(p, [NEG, NEG])
            e[0] = lse(e[0], tot + b)
            for c, l in cands:
                k = child.get((p, c))
                if k is None:
                    ctx, part, s, _ = state[p]
                    if start[c] and lm is not None:
                        ctx, add = score_words(ctx, part, alpha, beta, lm)
                        part, s = text[c], s + add
                    else:
                        part = text[c] if start[c] else part + text[c]
                    pen = 0.0 if lm is None or not (ws := words(part)) or ws[-1] in lm.prefixes else alpha * lm.unk_logp
                    k = child[(p, c)] = len(parent)
                    parent.append(p)
                    token.append(c)
                    state.append((ctx, part, s, pen))
                e2 = nxt.setdefault(k, [NEG, NEG])
                if c == token[p]:  # the same token again: one token unless a blank came between
                    e[1] = lse(e[1], pnb + l)
                    e2[1] = lse(e2[1], pb + l)
                else:
                    e2[1] = lse(e2[1], tot + l)
        if len(nxt) > width:
            nxt = dict(heapq.nlargest(width, nxt.items(),
                                      key=lambda kv: lse(*kv[1]) + state[kv[0]][2] + state[kv[0]][3]))
        beams = nxt

    def final(k):
        ctx, part, s, _ = state[k]
        if lm is not None:
            ctx, add = score_words(ctx, part, alpha, beta, lm)
            s += add + alpha * lm.logp(ctx, EOS)
        return lse(*beams[k]) + s

    k, out = max(beams, key=final), []
    while k > 0:
        out.append(token[k])
        k = parent[k]
    return out[::-1]


def job(a):
    split, i, lm, alpha, beta, width = a
    blank, ids, lps = G["topk"][split][i]
    return text_of(greedy(blank, ids, lps) if lm == "greedy" else
                   beam(blank, ids, lps, G["lms"].get(lm), alpha, beta, width))


def run(pool, clips, lm, alpha=0.0, beta=0.0, width=16):
    """clips: (split, index) pairs -> WER, transcripts."""
    hyps = pool.map(job, [(s, i, lm, alpha, beta, width) for s, i in clips], chunksize=4)
    return wer([G["truth"][s][i] for s, i in clips], hyps), hyps


@torch.no_grad()
def topk(model, split, device):
    """Per clip: blank log-prob per frame, the TOPK best other tokens and their log-probs."""
    out = []
    for a in range(0, len(split), 64):
        batch = split.batch(list(range(a, min(a + 64, len(split)))), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(batch[0])
        lp = logits.float().log_softmax(-1)
        top = lp[..., 1:].topk(TOPK, dim=-1)
        blank, vals = lp[..., 0].cpu().numpy(), top.values.cpu().numpy()
        ids = (top.indices + 1).cpu().numpy().astype(np.int32)
        out += [(blank[j, :n], ids[j, :n], vals[j, :n]) for j, n in enumerate(batch[1].tolist())]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("ckpt")
    ap.add_argument("--teacher", default=str(Path.home() / "ear/teacher"))
    ap.add_argument("--lines", required=True, help="tts.py's lines.jsonl (the conversational text)")
    ap.add_argument("--books", default="train,train360,train500", help="shard prefixes whose transcripts are the books")
    ap.add_argument("--tune", default="dev,devother", help="held-out splits that pick alpha and beta")
    ap.add_argument("--test", default="dev,devother,callsclean,calls,callsleadclean,callslead")
    ap.add_argument("--tune-n", type=int, default=300, help="clips per tuning split, evenly spread")
    ap.add_argument("--alphas", default="0.3,0.5,0.7,1.0,1.4")
    ap.add_argument("--betas", default="0,1,2,3,4")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--workers", type=int, default=18)
    args = ap.parse_args()
    t0, device = time.time(), "cuda"

    tok = Tokenizer.from_file(str(Path(args.teacher) / "tokenizer.json"))
    G["text"], G["start"] = token_table(tok)
    names = list(dict.fromkeys(args.tune.split(",") + args.test.split(",")))
    splits = {n: Split(args.data, n) for n in names if list(Path(args.data).glob(f"{n}_[0-9]*.json"))}
    G["truth"] = {n: [m["truth"] for m in s.meta] for n, s in splits.items()}
    ck = torch.load(args.ckpt, map_location=device)
    model = Ear(EarConfig(**ck["config"])).to(device).eval()
    model.load_state_dict(ck["model"])
    G["topk"] = {n: topk(model, s, device) for n, s in splits.items()}
    del model
    torch.cuda.empty_cache()
    print(f"log-probs for {sum(len(s) for s in splits.values())} clips ({time.time() - t0:.0f} s)", flush=True)

    held_out = {" ".join(words(t)) for ts in G["truth"].values() for t in ts}
    books = [w for n in args.books.split(",") for p in sorted(Path(args.data).glob(f"{n}_[0-9]*.json"))
             for m in json.loads(p.read_text(encoding="utf-8")) if " ".join(w := words(m["truth"])) not in held_out]
    lines = [w for l in Path(args.lines).read_text(encoding="utf-8").splitlines()
             if " ".join(w := words(json.loads(l)["text"])) not in held_out]
    random.Random(0).shuffle(lines)
    lines_held, lines_train = lines[:len(lines) // 10], lines[len(lines) // 10:]
    lm_books, lm_lines = KneserNey(books, args.order), KneserNey(lines_train, args.order)
    print(f"books: {len(books)} sentences, {sum(map(len, books))} words, {lm_books.size}; lines: {len(lines_train)} "
          f"({sum(map(len, lines_train))} words), {len(lines_held)} held out ({time.time() - t0:.0f} s)", flush=True)
    ppl = {lam: perplexity(Mix([lm_lines, lm_books], [lam, 1 - lam]), lines_held) for lam in np.arange(0.1, 1.0, 0.1)}
    lam = min(ppl, key=ppl.get)
    print("perplexity on the held-out lines: books " + f"{perplexity(lm_books, lines_held):.0f}, mix "
          + ", ".join(f"{k:.1f}: {v:.0f}" for k, v in ppl.items()) + f" -> lines weight {lam:.1f}", flush=True)
    print("4-gram coverage by the books (leak check): " + ", ".join(
        f"{n} {100 * lm_books.coverage([words(t) for t in ts]):.0f} %" for n, ts in G["truth"].items()), flush=True)
    G["lms"] = {"books": lm_books, "mix": Mix([lm_lines, lm_books], [float(lam), 1 - float(lam)])}

    report = {"step": ck["step"], "lines_weight": float(lam), "width": args.width}
    grid = [(a, b) for a in map(float, args.alphas.split(",")) for b in map(float, args.betas.split(","))]
    tune = [(s, int(i)) for s in args.tune.split(",")
            for i in np.unique(np.linspace(0, len(splits[s]) - 1, args.tune_n).round())]
    test = [n for n in args.test.split(",") if n in splits]
    with get_context("fork").Pool(args.workers) as pool:  # workers inherit G, language models included
        rows = {"greedy": (0.0, 0.0), "beam, no LM": (0.0, 0.0)}
        for lm in ("books", "mix"):
            scores = {ab: run(pool, tune, lm, *ab, args.width)[0] for ab in grid}
            rows[f"beam + {lm} LM"] = min(scores, key=scores.get)
            print(f"{lm}: tuned on {args.tune} -> alpha {rows[f'beam + {lm} LM'][0]}, beta "
                  f"{rows[f'beam + {lm} LM'][1]} ({scores[rows[f'beam + {lm} LM']]:.2f} % on the tuning clips, "
                  f"{time.time() - t0:.0f} s)", flush=True)
        hyps_out = []
        for row, (a, b) in rows.items():
            lm = "greedy" if row == "greedy" else None if row == "beam, no LM" else row.split()[2]
            report[row] = {"alpha": a, "beta": b}
            for n in test:
                clips = [(n, i) for i in range(len(splits[n]))]
                report[row][n], hyps = run(pool, clips, lm, a, b, args.width)
                hyps_out += [{"split": f"{n} | {row}", "utt": m["utt"], "truth": m["truth"], "teacher": m["teacher"],
                              "ear": h} for m, h in zip(splits[n].meta, hyps)]  # errors.py reads these per split
            print(f"{row:<16}" + "".join(f"  {n} {report[row][n]:.2f}" for n in test), flush=True)
        for n in [n for n in ("calls", "callslead") if n in splits]:  # tuned on the test itself: a ceiling only
            clips = [(n, i) for i in range(len(splits[n]))]
            scores = {ab: run(pool, clips, "mix", *ab, args.width)[0] for ab in grid}
            best = min(scores, key=scores.get)
            report[f"ceiling {n}"] = {"alpha": best[0], "beta": best[1], "wer": scores[best]}
            print(f"ceiling on {n} (tuned on itself): {scores[best]:.2f} % at alpha {best[0]}, beta {best[1]}",
                  flush=True)
    Path(args.ckpt).with_suffix(".decode.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    Path(args.ckpt).with_suffix(".decode.hyps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in hyps_out),
                                                                encoding="utf-8")
    print(f"done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
