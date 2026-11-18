"""int4 weights for the Ear, the way KARST's best measured point did it (hybrid_i4h, research E009b, D5/D6 in
`karst/research/probes/recall_diag.py`): W4A16 round-to-nearest, symmetric (scale = max |w| / 7, levels -8..7), one
scale per group of 128 inputs in each output row (64 when a row does not split into 128s: the first conv), no
retraining.

Same layers as KARST: the delta mixers' qkv, gate and o, the attention layers' qkv and o, both FFN projections and the
CTC head. The per-head decay and write-strength projections (a, b: 6 rows each), norms, biases and the delta layers'
depthwise convs stay as they are. --convs also rounds the two front-end convs (0.8M weights). --only rounds just some
of them (delta, attn, ffn, head, convs) to find what is sensitive; --bits and --group change the rounding.

The checkpoint it writes holds the rounded weights as ordinary floats, so evaluate.py and decode.py score it unchanged;
it prints what the packed model would weigh (int4 weights, fp16 scales, bf16 for the rest).

    python quant.py ~/ear/runs/r1c/ear_best.pt ~/ear/runs/r1c/int4/ear_int4.pt [--convs]
"""
import argparse
import json
import math
from pathlib import Path

import torch

GROUP = 128
MIXERS = {"D": ("mix.qkv", "mix.gate", "mix.o"), "A": ("mix.qkv", "mix.o")}
FFN = ("ffn.gate_up", "ffn.down")


def rtn(w, group=GROUP, bits=4):
    """Round a weight (Linear [out, in], or Conv1d [out, in, k] read as [out, in * k]) to `bits`. Returns the rounded
    copy in w's shape and dtype, and the number of scales."""
    top = 2 ** (bits - 1) - 1
    shape, o = w.shape, w.shape[0]
    w2 = w.float().reshape(o, -1)
    i = w2.shape[1]
    g = math.gcd(i, group)
    wg = w2.view(o, i // g, g)
    s = (wg.abs().amax(-1, keepdim=True) / top).clamp(min=1e-8)
    return ((wg / s).round().clamp(-top - 1, top) * s).view(shape).to(w.dtype), s.numel()


def targets(config, parts):
    """parts: a set of delta, attn, ffn, head, convs."""
    names = []
    for n, kind in enumerate(config["pattern"]):
        if ("delta" if kind == "D" else "attn") in parts:
            names += [f"blocks.{n}.{p}.weight" for p in MIXERS[kind]]
        if "ffn" in parts:
            names += [f"blocks.{n}.{p}.weight" for p in FFN]
    if "head" in parts:
        names.append("head.weight")
    if "convs" in parts:
        names += ["sub.c1.weight", "sub.c2.weight"]
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--convs", action="store_true", help="also round the two front-end convs")
    ap.add_argument("--only", help="comma-separated subset of delta,attn,ffn,head,convs (default: all but convs)")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=GROUP)
    args = ap.parse_args()
    parts = set(args.only.split(",")) if args.only else {"delta", "attn", "ffn", "head"}
    if args.convs:
        parts.add("convs")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    names = targets(ck["config"], parts)
    missing = [n for n in names if n not in sd]
    assert not missing, missing
    q_params = scales = 0
    err = {}
    for n in names:
        w = sd[n]
        sd[n], k = rtn(w, args.group, args.bits)
        q_params += w.numel()
        scales += k
        err[n] = ((sd[n].float() - w.float()).norm() / w.float().norm()).item()
    params = sum(v.numel() for k, v in sd.items() if k not in ("mean", "std"))
    rest = params - q_params
    packed = q_params * args.bits / 8 + scales * 2 + rest * 2
    report = {"from": args.ckpt, "step": ck["step"], "bits": args.bits, "group": args.group, "parts": sorted(parts),
              "params_M": params / 1e6, "int4_params_M": q_params / 1e6, "bf16_params_M": rest / 1e6,
              "packed_MB": packed / 1e6, "bf16_MB": params * 2 / 1e6,
              "rel_err_mean": sum(err.values()) / len(err), "rel_err_max": max(err.values()),
              "worst": max(err, key=err.get)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({**ck, "model": sd, "int4": report}, args.out)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
