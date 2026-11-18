"""Replay one of our calls through the friend with different ears: what DUET answers when it gets the caller's turns
as they were said, as TurboCTC heard them (the same cut clip, offline) and as the Reflexx Ear heard them. Each version
gets its own fresh friend, so the conversation can drift the way it would live. Python's and the model's sampling use
a fixed seed, but the model on the GPU still varies its wording a little for the same words (2026-11-06: same text,
different first reply), so compare the decisions (mode/move), not the wording. Tone is neutral (no audio), and a turn
the ear heard as nothing is skipped, as DUET would not answer it.

    .venv/Scripts/python tools/ear_replay.py 20261103-230111 runs/ear-real/r1c_hyps.jsonl --split "real | beam + mix LM"

Writes runs/ear-real/replay_<call>.md (or --out) next to the turns (private: his voice's words, vault only).
"""
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from duet.friend import FriendBrain  # noqa: E402
from duet.llm import Llm  # noqa: E402
from duet.prosody import ToneReading  # noqa: E402
from duet.understand import TextReflex  # noqa: E402

SEED = 7


class SeededLlm(Llm):
    def _post(self, messages, stream, max_tokens, temperature, fmt=None):
        body = {"model": self.model, "messages": messages, "stream": stream, "keep_alive": self.keep_alive,
                "options": {"temperature": temperature, "num_predict": max_tokens, "top_p": 0.9,
                            "repeat_penalty": 1.1, "repeat_last_n": 256, "seed": SEED}}
        if fmt:
            body["format"] = fmt
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=60)


def say(reply):
    reply.sentences.extend(reply.stream or [])
    return " ".join(reply.sentences)


def run(llm, lines):
    """lines: what DUET heard, per caller turn ('' = nothing). Returns the opening and one row per turn."""
    random.seed(SEED)
    brain = FriendBrain(llm, memory_path=None, seed=SEED)
    reflex = TextReflex(brain.reflex_dir)
    calm = ToneReading(0.0, 0.0, "")
    r = brain.opening()
    opening = say(r)
    brain.heard(r, len(r.sentences), False)
    rows = []
    for text in lines:
        if not text.strip():
            rows.append({"heard": "", "decision": "", "reply": "(heard nothing: no answer)"})
            continue
        reading = reflex.read(text)
        reply = brain.caller_turn(reading, calm)
        out = say(reply)
        d = getattr(brain, "last_decision", {})
        rows.append({"heard": text, "decision": f"{d.get('mode')}/{d.get('move')}, act {reading.act}", "reply": out})
        brain.heard(reply, len(reply.sentences), False)
        if reply.end_call:
            break
    if hasattr(brain, "close"):
        brain.close()
    return opening, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("call", help="call id, e.g. 20261103-230111 (the turns in runs/ear-real/truth.jsonl)")
    ap.add_argument("hyps", help="the ear's transcripts of the real turns (evaluate/decode --hyps rows)")
    ap.add_argument("--split", default="real | beam + mix LM")
    ap.add_argument("--ear", default="Reflexx Ear r1c", help="name in the report")
    ap.add_argument("--truth", default=str(ROOT / "runs/ear-real/truth.jsonl"))
    ap.add_argument("--out", help="report path (default runs/ear-real/replay_<call>.md)")
    args = ap.parse_args()
    turns = [t for t in map(json.loads, open(args.truth, encoding="utf-8"))
             if t["id"].rsplit("_", 1)[0] == args.call and not t.get("drop")]
    ear = {h["utt"]: h["ear"] for h in map(json.loads, open(args.hyps, encoding="utf-8")) if h["split"] == args.split}
    missing = [t["id"] for t in turns if t["id"] not in ear]
    assert not missing, f"no ear transcript for {missing}"
    versions = {"as said": [t["text"] for t in turns], "TurboCTC": [t["draft"] for t in turns],
                args.ear: [ear[t["id"]] for t in turns]}
    llm = SeededLlm()
    out = {name: run(llm, lines) for name, lines in versions.items()}
    md = [f"# Call {args.call} replayed through the friend: {len(turns)} caller turns", "",
          f"Ear transcripts: {args.split}. Same seed for every version; tone neutral.", ""]
    for name, (opening, _) in out.items():
        md.append(f"- opening ({name}): {opening}")
    for i, t in enumerate(turns):
        md += ["", f"## {i + 1}. {t['id']}", ""]
        for name, (_, rows) in out.items():
            if i >= len(rows):
                md.append(f"- **{name}**: (call already ended)")
                continue
            r = rows[i]
            md.append(f"- **{name}** heard: \"{r['heard']}\" [{r['decision']}]")
            md.append(f"  - DUET: {r['reply']}")
    path = Path(args.out) if args.out else ROOT / "runs/ear-real" / f"replay_{args.call}.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwritten to {path}")


if __name__ == "__main__":
    main()
