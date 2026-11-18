"""Train the text reflex: bge-small-en-v1.5 (MIT, 33M) + five heads read from one CLS vector, then export one ONNX.

    ../frontdesk-bench/.venv/Scripts/python tools/train_text_reflex.py [--out models/text-reflex] [--epochs 4]
        [--friend]   # two more heads for the friend: mode (7) and feeling (5), from tools/make_friend_data.py

Run it with frontdesk-bench's venv (torch 2.9 ROCm + transformers 5.20). Heads, all on normalize(CLS) * 20 like
Reflexx's intent head: intent (26, softmax), act (11, softmax), flags (greeting, request: sigmoid), sentiment
(neg/neu/pos, softmax), complete (sigmoid). Rows mask the heads they have no label for.

Tests (nothing here is trained on): Reflexx's 310 benchmark calls (truth + phone ASR) for intent, the hand-written
`data/text/test_acts.jsonl` for act / greeting / request / sentiment, and the 310 calls cut at a random word for
completeness. Writes <out>/text_reflex.onnx, tokenizer.json, meta.json, REPORT.md. Never overwrites: an existing
<out> stops the run.
"""
import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
EVALSET = ROOT.parent / "frontdesk-bench" / "train" / "evalset"
ACTS = ["request", "greeting", "smalltalk", "backchannel", "stop", "repeat", "continue", "thanks", "goodbye",
        "affirm", "deny"]
SENTS = ["neg", "neu", "pos"]
MODES = ["share_good", "share_bad", "share_story", "chat", "about_you", "ask_info", "ask_advice"]
FEELINGS = ["very_negative", "negative", "neutral", "positive", "very_positive"]
FEELING_SENT = {"very_negative": "neg", "negative": "neg", "neutral": "neu", "positive": "pos", "very_positive": "pos"}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def norm(t):
    return re.sub(r"[\s.,!?;:]+$", "", re.sub(r"\s+", " ", t).strip())


class TextReflex(torch.nn.Module):
    def __init__(self, encoder, n_intents, friend=False):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.intent = torch.nn.Linear(h, n_intents)
        self.act = torch.nn.Linear(h, len(ACTS))
        self.flags = torch.nn.Linear(h, 2)
        self.sentiment = torch.nn.Linear(h, len(SENTS))
        self.complete = torch.nn.Linear(h, 1)
        self.friend = friend
        if friend:
            self.mode = torch.nn.Linear(h, len(MODES))
            self.feeling = torch.nn.Linear(h, len(FEELINGS))

    def logits(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        z = torch.nn.functional.normalize(cls, dim=-1) * 20.0
        out = (self.intent(z), self.act(z), self.flags(z), self.sentiment(z), self.complete(z))
        return out + (self.mode(z), self.feeling(z)) if self.friend else out

    def forward(self, input_ids, attention_mask):
        i, a, f, s, c, *rest = self.logits(input_ids, attention_mask)
        out = (torch.softmax(i, -1), torch.softmax(a, -1), torch.sigmoid(f), torch.softmax(s, -1),
               torch.sigmoid(c)[:, 0])
        return out + tuple(torch.softmax(x, -1) for x in rest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "models" / "text-reflex"))
    ap.add_argument("--base", default=str(ROOT / "models" / "bge-small-en-v1.5"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--friend", action="store_true", help="add the friend's mode and feeling heads")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} exists; pick a new --out (runs are never overwritten)")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    intents = json.load(open(EVALSET / "labels.json"))
    ii, ai, si, mi, fi = ({l: i for i, l in enumerate(x)} for x in (intents, ACTS, SENTS, MODES, FEELINGS))
    train, val = load(ROOT / "data/text/train.jsonl"), load(ROOT / "data/text/val.jsonl")
    if a.friend:  # everyday talk: mode + feeling and its sentiment (tools/make_friend_data.py, both sets)
        rng = random.Random(1)

        def friend_rows(split):
            rows = []
            for name in ("friend", "friend_plain"):
                path = ROOT / f"data/text/{name}_{split}.jsonl"
                for k, r in enumerate(load(path) if path.exists() else []):
                    text = r["text"].replace("’", "'").replace("‘", "'")  # the recogniser's apostrophe
                    # out of scope for the front desk on a third of them only (all of them cost v5 2.5 points of
                    # intent); no "finished" label (all-finished cost v5 15 points of cut-sentence detection), but
                    # a cut copy of every second one as unfinished
                    rows.append({"text": text, "mode": r["mode"], "feeling": r["feeling"],
                                 "sentiment": FEELING_SENT[r["feeling"]],
                                 "intent": "out_of_scope" if k % 3 == 0 else None})
                    w = text.split()
                    if k % 2 == 0 and len(w) >= 5:
                        rows.append({"text": " ".join(w[:rng.randint(2, int(len(w) * 0.7))]), "complete": 0})
            return rows
        train += friend_rows("train")
        val += friend_rows("val")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    model = TextReflex(AutoModel.from_pretrained(a.base), len(intents), friend=a.friend).to(dev)

    def encode(texts):
        t = tok([norm(x) for x in texts], padding=True, truncation=True, max_length=64, return_tensors="pt")
        return t["input_ids"].to(dev), t["attention_mask"].to(dev)

    def target(rows, key, index):
        y = torch.tensor([index[r[key]] if r.get(key) is not None else -100 for r in rows])
        return y.to(dev)

    def binary(rows, key):
        y = torch.tensor([float(r[key]) if r.get(key) is not None else -1.0 for r in rows])
        return y.to(dev)

    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def bce(logit, y):
        m = y >= 0
        return torch.nn.functional.binary_cross_entropy_with_logits(logit[m], y[m]) if m.any() else logit.sum() * 0

    def loss_of(rows):
        li, la, lf, ls, lc, *lfriend = model.logits(*encode([r["text"] for r in rows]))
        loss = 0.0
        heads = [(li, target(rows, "intent", ii), 1.0), (la, target(rows, "act", ai), 1.0),
                 (ls, target(rows, "sentiment", si), 1.0)]
        if lfriend:
            heads += [(lfriend[0], target(rows, "mode", mi), 1.0), (lfriend[1], target(rows, "feeling", fi), 1.0)]
        for logit, y, w in heads:
            if (y >= 0).any():
                loss = loss + w * ce(logit, y)
        loss = loss + 0.5 * bce(lf[:, 0], binary(rows, "greeting")) + 0.5 * bce(lf[:, 1], binary(rows, "request"))
        return loss + 0.5 * bce(lc[:, 0], binary(rows, "complete"))

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(train) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for epoch in range(a.epochs):
        model.train()
        order = np.random.permutation(len(train))
        for i in range(0, len(train), a.batch):
            loss = loss_of([train[j] for j in order[i:i + a.batch]])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            vl = np.mean([loss_of(val[i:i + 128]).item() for i in range(0, len(val), 128)])
        print(f"epoch {epoch + 1}: train loss {loss.item():.3f} val loss {vl:.3f} ({time.time() - t0:.0f}s)", flush=True)

    @torch.no_grad()
    def predict(texts):
        model.eval()
        outs = [model(*encode(texts[i:i + 128])) for i in range(0, len(texts), 128)]
        return [torch.cat([o[k] for o in outs]).float().cpu().numpy() for k in range(len(outs[0]))]

    report = {}
    for src in ("truth", "asr_phone"):
        rows = load(EVALSET / f"{src}.jsonl")
        p = predict([r["text"] for r in rows])
        report[f"intent_{src}"] = float(np.mean([intents[int(x.argmax())] == r["label"] for r, x in zip(rows, p[0])]))
        report[f"request_flag_{src}"] = float(np.mean(p[2][:, 1] > 0.5))
        report[f"complete_{src}"] = float(np.mean(p[4] > 0.5))
        rng = random.Random(5)
        cut = []
        for r in rows:
            w = r["text"].split()
            if len(w) >= 5:
                cut.append(" ".join(w[:rng.randint(2, int(len(w) * 0.8))]))
        report[f"incomplete_detected_{src}_cut"] = float(np.mean(predict(cut)[4] < 0.5))
    test = load(ROOT / "data/text/test_acts.jsonl")
    p = predict([r["text"] for r in test])
    acted = [(r, x) for r, x in zip(test, p[1]) if r.get("act")]
    report["act_test"] = float(np.mean([ACTS[int(x.argmax())] == r["act"] for r, x in acted]))
    report["act_test_n"] = len(acted)
    report["act_test_errors"] = [f"{r['text']!r}: {r['act']} -> {ACTS[int(x.argmax())]}" for r, x in acted
                                 if ACTS[int(x.argmax())] != r["act"]]
    report["greeting_flag_test"] = float(np.mean([(f[0] > 0.5) == bool(r["greeting"]) for r, f in zip(test, p[2])]))
    report["request_flag_test"] = float(np.mean([(f[1] > 0.5) == bool(r["request"]) for r, f in zip(test, p[2])]))
    report["sentiment_test"] = float(np.mean([SENTS[int(x.argmax())] == r["sentiment"] for r, x in zip(test, p[3])]))
    report["sentiment_test_errors"] = [f"{r['text']!r}: {r['sentiment']} -> {SENTS[int(x.argmax())]}"
                                       for r, x in zip(test, p[3]) if SENTS[int(x.argmax())] != r["sentiment"]]
    with_intent = [(r, x) for r, x in zip(test, p[0]) if r.get("intent")]
    report["intent_test_acts"] = float(np.mean([intents[int(x.argmax())] == r["intent"] for r, x in with_intent]))
    if a.friend:  # the hand-written friend test set (never generated)
        ft = load(ROOT / "data/text/friend_test.jsonl")
        q = predict([r["text"] for r in ft])
        pm = [MODES[int(x.argmax())] for x in q[5]]
        pf = [FEELINGS[int(x.argmax())] for x in q[6]]
        report["friend_mode_test"] = float(np.mean([m == r["mode"] for m, r in zip(pm, ft)]))
        report["friend_feeling_test"] = float(np.mean([f == r["feeling"] for f, r in zip(pf, ft)]))
        report["friend_feeling_direction_test"] = float(np.mean([FEELING_SENT[f] == FEELING_SENT[r["feeling"]]
                                                                 for f, r in zip(pf, ft)]))
        report["friend_test_n"] = len(ft)
        report["friend_errors"] = [f"{r['text']!r}: {r['mode']}/{r['feeling']} -> {m}/{f}"
                                   for m, f, r in zip(pm, pf, ft) if m != r["mode"] or f != r["feeling"]]
    print(json.dumps(report, indent=1))

    out.mkdir(parents=True)
    model.eval().cpu()
    sample = tok(["Hi, can I move my consultation to Friday", "mm-hm"], padding=True, return_tensors="pt")
    path = out / "text_reflex.onnx"
    names = ["intent", "act", "flags", "sentiment", "complete"] + (["mode", "feeling"] if a.friend else [])
    torch.onnx.export(model, (sample["input_ids"], sample["attention_mask"]), path, opset_version=17, dynamo=False,
                      input_names=["input_ids", "attention_mask"], output_names=names,
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    **{n: {0: "batch"} for n in names}})
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    enc = tok([norm(r["text"]) for r in test], padding=True, truncation=True, max_length=64, return_tensors="np")
    got = sess.run(names, {"input_ids": enc["input_ids"].astype(np.int64),
                           "attention_mask": enc["attention_mask"].astype(np.int64)})
    with torch.no_grad():
        ref = model(torch.tensor(enc["input_ids"]), torch.tensor(enc["attention_mask"]))
    report["onnx_max_abs_diff"] = float(max(np.abs(g - r.numpy()).max() for g, r in zip(got, ref)))
    shutil.copy(Path(a.base) / "tokenizer.json", out / "tokenizer.json")
    meta = {"intents": intents, "acts": ACTS, "sentiments": SENTS, "base": "BAAI/bge-small-en-v1.5",
            "max_length": 64, "report": report}
    if a.friend:
        meta |= {"modes": MODES, "feelings": FEELINGS}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    lines = ["# Text reflex", "", f"bge-small-en-v1.5 + {len(names)} heads, {a.epochs} epochs, lr {a.lr}, "
             f"{len(train)} training rows. ONNX {path.stat().st_size / 1e6:.0f} MB, max |diff| vs torch "
             f"{report['onnx_max_abs_diff']:.1e}.", "", "| test | value |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in report.items() if isinstance(v, float)]
    lines += ["", "Act errors:", *[f"- {e}" for e in report["act_test_errors"]], "", "Sentiment errors:",
              *[f"- {e}" for e in report["sentiment_test_errors"]]]
    if a.friend:
        lines += ["", "Friend test errors (mode/feeling wanted -> got):", *[f"- {e}" for e in report["friend_errors"]]]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
