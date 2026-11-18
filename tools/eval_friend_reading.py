"""How well the friend reads what someone wants (mode) and how strongly they feel (feeling), on the hand-written test
set data/text/friend_test.jsonl (68 lines, incl. the friend's first live call): the rules (text reflex v4 + feeling
words + patterns) against the learned heads (text reflex v5 with --friend).

    .venv/Scripts/python tools/eval_friend_reading.py models/text-reflex-v4 [models/text-reflex-v5]
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from duet.friend import feeling, mode_of  # noqa: E402
from duet.understand import TextReflex  # noqa: E402

SEVEN = {("listen", "good_news"): "share_good", ("listen", "nice"): "share_good", ("listen", "bad_news"): "share_bad",
         ("listen", "story"): "share_story", ("listen", "teaser"): "share_story", ("chat", "chat"): "chat",
         ("chat", "yes_no"): "chat", ("chat", "about_you"): "about_you", ("help", "answer"): "ask_info",
         ("help", "ideas"): "ask_advice", ("ritual", "how_are_you"): "about_you", ("ritual", "hello"): "chat",
         ("ritual", "thanks"): "chat", ("ritual", "bye"): "chat"}
DIRECTION = {"very_negative": "neg", "negative": "neg", "neutral": "neu", "positive": "pos", "very_positive": "pos"}


def five(s):
    """{neg, neu, pos} -> the five feeling levels (the rules can't tell very negative from negative)."""
    if s["neg"] > 0.85:
        return "very_negative"
    if s["neg"] > 0.5:
        return "negative"
    if s["pos"] > 0.5:
        return "very_positive"
    if s["pos"] >= 0.35:
        return "positive"
    return "neutral"


def main():
    rows = [json.loads(line) for line in open(ROOT / "data/text/friend_test.jsonl", encoding="utf-8")]
    for path in sys.argv[1:]:
        reflex = TextReflex(Path(path))
        modes, feels, wrong = [], [], Counter()
        for r in rows:
            reading = reflex.read(r["text"])
            reading.sentiment = feeling(reading, r["text"])
            m = SEVEN.get(mode_of(reading, r["text"]), "chat")
            f = five(reading.sentiment)
            modes.append(m == r["mode"])
            feels.append((f == r["feeling"], DIRECTION[f] == DIRECTION[r["feeling"]]))
            if m != r["mode"]:
                wrong[f"{r['mode']} -> {m}"] += 1
        n = len(rows)
        how = "learned heads" if reflex.modes else "rules"
        print(f"{Path(path).name} ({how}): mode {sum(modes)}/{n} ({100 * sum(modes) / n:.0f} %), feeling level "
              f"{sum(a for a, _ in feels)}/{n}, feeling direction {sum(b for _, b in feels)}/{n} | most common mode "
              f"errors: {wrong.most_common(4)}")


if __name__ == "__main__":
    main()
