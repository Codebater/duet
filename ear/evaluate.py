"""Score a trained Ear: word error rate on the held-out speakers and the 310 phone calls (vs the truth, next to TurboCTC),
and the streaming check: feeding 160 ms at a time with every layer's state carried (delta layers run step by step)
must give the same words as one pass over the whole clip.

    python evaluate.py ~/ear/prep ~/ear/runs/r0/ear.pt
"""
import argparse
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from data import Split, greedy, wer
from model import Ear, EarConfig
from train import evaluate

CHUNK_ROWS = 8  # 160 ms


@torch.no_grad()
def streaming_check(model, split, tok, device, n):
    """fp32: full-clip pass (fla chunk kernel) vs 160 ms chunks with state (reference recurrence)."""
    model.eval().float()
    full_h, stream_h, refs, worst, t_chunk = [], [], [], 0.0, []
    for i in range(min(n, len(split))):
        feats = split.batch([i], device)[0]
        full, _ = model(feats)
        state, parts = None, []
        for a in range(0, feats.shape[1], CHUNK_ROWS):
            t0 = time.perf_counter()
            logits, state = model(feats[:, a:a + CHUNK_ROWS], state, recurrent=True)
            t_chunk.append(time.perf_counter() - t0)
            parts.append(logits)
        stream = torch.cat(parts, dim=1)
        worst = max(worst, (stream - full).abs().max().item())
        frames = [full.shape[1]]
        full_h.append(tok.decode(greedy(full, frames)[0], skip_special_tokens=True))
        stream_h.append(tok.decode(greedy(stream, frames)[0], skip_special_tokens=True))
        refs.append(split.meta[i]["truth"])
    return {"n": len(refs), "wer_full": wer(refs, full_h), "wer_stream": wer(refs, stream_h),
            "same_text": sum(a == b for a, b in zip(full_h, stream_h)), "max_logit_diff": worst,
            "ms_per_chunk_gpu_python": 1000 * sum(t_chunk) / len(t_chunk)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("ckpt")
    ap.add_argument("--teacher", default=str(Path.home() / "ear/teacher"))
    ap.add_argument("--stream-n", type=int, default=100)
    ap.add_argument("--dev", default="dev", help="comma-separated held-out prefixes; the first gets the streaming check")
    ap.add_argument("--hyps", action="store_true", help="also write every transcript to <ckpt>.hyps.jsonl")
    args = ap.parse_args()
    device = "cuda"
    tok = Tokenizer.from_file(str(Path(args.teacher) / "tokenizer.json"))
    ck = torch.load(args.ckpt, map_location=device)
    model = Ear(EarConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    report = {"step": ck["step"], "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    names = args.dev.split(",")
    rows = []
    for name in names:
        split = Split(args.data, name)
        report[name], hyps = evaluate(model, split, tok, device)
        rows += [{"split": name, "utt": m["utt"], "truth": m["truth"], "teacher": m["teacher"], "ear": h}
                 for m, h in zip(split.meta, hyps)]
    dev = Split(args.data, names[0])
    if list(Path(args.data).glob("calls_[0-9]*.json")):
        calls = Split(args.data, "calls")
        report["calls"], call_hyps = evaluate(model, calls, tok, device)
        report["calls_examples"] = [{"truth": m["truth"], "teacher": m["teacher"], "ear": h}
                                    for m, h in list(zip(calls.meta, call_hyps))[:8]]
        rows += [{"split": "calls", "utt": m["utt"], "truth": m["truth"], "teacher": m["teacher"], "ear": h}
                 for m, h in zip(calls.meta, call_hyps)]
    if args.stream_n:
        report["streaming"] = streaming_check(model, dev, tok, device, args.stream_n)
    print(json.dumps(report, indent=1))
    Path(args.ckpt).with_suffix(".eval.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.hyps:
        Path(args.ckpt).with_suffix(".hyps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                                             encoding="utf-8")


if __name__ == "__main__":
    main()
