"""The moments you marked with keys during a live call, each with what DUET had decided and the call around it.

    python tools/review_call.py [runs/live-...]        (default: the newest live call)

Keys during `python -m duet live`: g good, b bad, t wrong tone or energy, l should have listened, c should have talked
back, x missed the context, m misheard me, i interrupted me, s too slow, n note.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else max((ROOT / "runs").glob("live-*"),
                                                          key=lambda p: p.stat().st_mtime)
    call = json.load(open(run / "call.json", encoding="utf-8"))
    events, marks = call["events"], call.get("feedback", [])
    turns = [e for e in events if e["kind"] in ("caller", "duet", "said", "listening", "cut", "barge-in")]
    print(f"{run.name}: {sum(e['kind'] == 'caller' for e in events)} turns of yours, {len(marks)} marked moments, "
          f"reply latencies {call['latencies_ms']} ms\n")
    for m in marks:
        d = m.get("decision") or {}
        print(f"[{m['t']:6.1f} s] {m['label'].upper()}")
        if d:
            print(f"  you said     \"{d.get('text', '')}\"")
            print(f"  DUET decided mode {d.get('mode')}, move {d.get('move')}, voice {d.get('register')}, "
                  f"opener {d.get('opener')!r}, mood {d.get('mood')}")
        if m.get("speaking"):
            print(f"  it was saying \"{m['speaking']}\"")
        elif m.get("last_reply"):
            print(f"  it had said  \"{m['last_reply']}\"")
        near = [e for e in turns if m["t"] - 12 <= e["t"] <= m["t"] + 1]
        for e in near:
            print(f"    {e['t']:6.1f} {e['kind']:9s} {e['text'][:150]}")
        print()


if __name__ == "__main__":
    main()
