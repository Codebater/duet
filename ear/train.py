"""Train the Ear from TurboCTC's labels: CTC on the teacher's transcript + KL to its top-16 posteriors per 80 ms frame.

The student hears causally, so it may answer `--delay` frames after the teacher: student frame j+delay is matched to
teacher frame j (prep.py adds 4 frames of silence after each utterance for that).

    python train.py ~/ear/prep ~/ear/runs/r0 --minutes 120

A run that was stopped goes on from its last eval when the same command is run again with --resume (`resume.pt`: weights,
optimizer, data position, random state). A run from before resume.pt goes on from `ear.pt` with a new optimizer.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from data import Split, greedy, wer
from model import Ear, EarConfig


def losses(model, batch, delay, spec=None):
    feats, frames, top_ids, top_lp, tt, target, target_len = batch
    if spec is not None:
        feats = spec(feats, model)
    logits, _ = model(feats)
    lp = logits.float().log_softmax(-1)
    ctc = F.ctc_loss(lp.transpose(0, 1), target, frames, target_len, blank=0, reduction="mean", zero_infinity=True)
    tmax = top_ids.shape[1]
    student = lp[:, delay:delay + tmax]
    p = top_lp.softmax(-1)  # the teacher's top 16, renormalised
    ce = -(p * student.gather(-1, top_ids)).sum(-1)
    valid = torch.arange(tmax, device=ce.device)[None] < tt[:, None]
    kl = (ce * valid).sum() / valid.sum()
    return ctc, kl, logits


class SpecAugment:
    """Two frequency masks (same mel bins in all four 80-wide blocks of a row) and time masks over ~5 % of rows."""

    def __init__(self, rng, freq=15, time_w=20):
        self.g, self.freq, self.time_w = rng, freq, time_w

    def __call__(self, feats, model):
        b, rows, _ = feats.shape
        x = feats.view(b, rows, 4, 80).clone()
        mean = model.mean.view(4, 80)
        for i in range(b):
            for _ in range(2):
                w = int(self.g.integers(0, self.freq + 1))
                f0 = int(self.g.integers(0, 80 - w + 1))
                x[i, :, :, f0:f0 + w] = mean[:, f0:f0 + w]
            for _ in range(max(1, rows // 400)):
                w = int(self.g.integers(0, self.time_w + 1))
                t0 = int(self.g.integers(0, max(1, rows - w)))
                x[i, t0:t0 + w] = mean
        return x.view(b, rows, 320)


def save(obj, path):
    """torch.save through a temporary file, so a run stopped mid-write never leaves a broken checkpoint."""
    tmp = Path(f"{path}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, split, tok, device, max_rows=60000, limit=0):
    model.eval()
    hyps, idx_all = [], list(range(len(split)))[:limit or None]
    for idx in [idx_all[i:i + 64] for i in range(0, len(idx_all), 64)]:
        batch = split.batch(idx, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(batch[0])
        hyps += [tok.decode(t, skip_special_tokens=True).strip() for t in greedy(logits, batch[1])]
    model.train()
    meta = [split.meta[i] for i in idx_all]
    return {"wer_truth": wer([m["truth"] for m in meta], hyps),
            "teacher_wer_truth": wer([m["truth"] for m in meta], [m["teacher"] for m in meta]),
            "wer_teacher": wer([m["teacher"] for m in meta], hyps), "n": len(meta)}, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--teacher", default=str(Path.home() / "ear/teacher"))
    ap.add_argument("--train", default="train", help="comma-separated shard prefixes to train on; tts*3 = three times")
    ap.add_argument("--dev", default="dev", help="comma-separated held-out prefixes; the first picks the best step")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--pattern", default="DDDADD")
    ap.add_argument("--heads", type=int, default=4, help="64 wide each: d / 64")
    ap.add_argument("--ffn", type=int, default=640)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--minutes", type=float, default=120)
    ap.add_argument("--steps", type=int, default=0, help="fixed step count (else from --minutes after a timing run)")
    ap.add_argument("--max-rows", type=int, default=64000, help="feature rows per batch (20 ms each)")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--kl", type=float, default=0.5)
    ap.add_argument("--delay", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--snapshot-every", type=int, default=0, help="also keep ear_<step>.pt every N steps (for analysis)")
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="go on from the run in OUT (resume.pt, else ear.pt), if any")
    ap.add_argument("--init", help="a new run from this checkpoint's weights (input statistics included): new "
                                   "optimizer, warm-up, step 0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(Path(args.teacher) / "tokenizer.json"))

    t0 = time.time()
    names = [n for spec in args.train.split(",") for n in [spec.split("*")[0]] * int((spec.split("*") + [1])[1])]
    train = Split(args.data, names)
    devs = {name: Split(args.data, name) for name in args.dev.split(",")}
    if list(Path(args.data).glob("calls_[0-9]*.json")):
        devs["calls"] = Split(args.data, "calls")
    dev_name = args.dev.split(",")[0]
    print(f"train {len(train)} utts {train.hours():.1f} h; "
          + ", ".join(f"{n} {len(s)} utts {s.hours():.2f} h" for n, s in devs.items())
          + f"  (loaded in {time.time() - t0:.0f} s)", flush=True)

    cfg = EarConfig(d=args.d, pattern=args.pattern, heads=args.heads, ffn=args.ffn, window=args.window)
    model = Ear(cfg).to(device)
    sample = train.sample()
    model.mean.copy_(torch.from_numpy(sample.mean(0)))
    model.std.copy_(torch.from_numpy(sample.std(0) + 1e-3))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ear: {n_params / 1e6:.2f}M params  {cfg}", flush=True)

    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)]
    rest = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and not getattr(p, "_no_weight_decay", False))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01}, {"params": rest, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.98), fused=True)
    spec = None if args.no_spec else SpecAugment(rng)

    total = args.steps or 10 ** 9  # set after 50 timed steps when --minutes decides
    step, epoch, skip, log, best, minutes, rewarm = 0, 0, 0, [], None, 0.0, None
    if args.resume and (out / "resume.pt").exists():
        st = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["rng"]
        torch.set_rng_state(st["torch_rng"])
        torch.cuda.set_rng_state(st["cuda_rng"])
        step, epoch, skip, total = st["step"], st["epoch"] - 1, st["batch"], st["total"]
        log, best, minutes = st["log"], st["best"], st["minutes"]
        print(f"resumed at step {step} (epoch {epoch + 1}, batch {skip})", flush=True)
    elif args.resume and (out / "ear.pt").exists():  # a run from before resume.pt: its weights, a new optimizer
        assert args.steps, "resuming from ear.pt needs --steps"
        prev = torch.load(out / "ear.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(prev["model"])
        step = rewarm = prev["step"]
        if (out / "log.json").exists():
            log = [r for r in json.loads((out / "log.json").read_text(encoding="utf-8")) if r["step"] <= step]
        best = min((r[dev_name]["wer_truth"] for r in log), default=None)
        epoch, minutes = (log[-1]["epoch"], log[-1]["minutes"]) if log else (0, 0.0)
        print(f"resumed at step {step} from ear.pt: new optimizer, warmup again, new data order", flush=True)
    elif args.init:
        prev = torch.load(args.init, map_location="cpu", weights_only=False)
        assert prev["config"] == model.config(), (prev["config"], model.config())
        model.load_state_dict(prev["model"])
        print(f"initialised from {args.init} (its step {prev['step']})", flush=True)

    def ck():
        return {"model": model.state_dict(), "config": model.config(), "step": step, "delay": args.delay}

    t_start = time.time() - minutes * 60
    while step < total:
        epoch += 1
        batches = train.batches(args.max_rows, np.random.default_rng([args.seed, epoch]))  # rebuilt as is on resume
        for k, idx in enumerate(batches[skip:], skip):
            if step >= total:
                break
            lr = args.lr * min(1.0, (step + 1) / args.warmup)
            if step >= args.warmup and total < 10 ** 9:
                lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / (total - args.warmup))))
            if rewarm is not None:  # the new optimizer's moments start from zero
                lr *= min(1.0, (step - rewarm + 1) / args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = train.batch(idx, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctc, kl, _ = losses(model, batch, args.delay, spec)
            loss = ctc + args.kl * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step == 50 and not args.steps:
                per = (time.time() - t_start) / 50
                total = max(args.warmup * 2, int(args.minutes * 60 / per * 0.93))  # leave time for evals
                print(f"{per * 1000:.0f} ms/step -> {total} steps for {args.minutes:.0f} min", flush=True)
            if step % 50 == 0:
                print(f"step {step} ep {epoch}  ctc {ctc.item():.3f}  kl {kl.item():.3f}  gn {gn.item():.2f}  "
                      f"lr {lr:.2e}  {(time.time() - t_start) / 60:.1f} min", flush=True)
            if args.snapshot_every and step % args.snapshot_every == 0:
                save(ck(), out / f"ear_{step}.pt")
            if step % args.eval_every == 0 or step == total:
                row = {"step": step, "epoch": epoch, "minutes": (time.time() - t_start) / 60,
                       "ctc": ctc.item(), "kl": kl.item()}
                hyps = {}
                for name, split in devs.items():
                    row[name], hyps[name] = evaluate(model, split, tok, device)
                log.append(row)
                print("EVAL " + json.dumps(row), flush=True)
                for m, h in list(zip(devs[dev_name].meta, hyps[dev_name]))[:3]:
                    print(f"  truth  : {m['truth']}\n  teacher: {m['teacher']}\n  ear    : {h}", flush=True)
                save(ck(), out / "ear.pt")
                if best is None or row[dev_name]["wer_truth"] < best:
                    best = row[dev_name]["wer_truth"]
                    save(ck(), out / "ear_best.pt")
                    print(f"  best so far: {dev_name} WER {best:.2f} at step {step}", flush=True)
                (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
                save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": epoch,
                      "batch": k + 1, "total": total, "log": log, "best": best, "minutes": row["minutes"],
                      "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state()}, out / "resume.pt")
        skip = 0
    print(f"done: {step} steps in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
