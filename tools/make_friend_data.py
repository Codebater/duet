"""Training sentences for the friend's mode and feeling heads, written by the local language model.

    .venv/Scripts/python tools/make_friend_data.py [--per-combo 240]

Each row is something a person might say out loud to a friend on a call, with two labels:
- **mode**, what they want from this moment: share_good (good news), share_bad (something hard or annoying),
  share_story (telling what they're doing or what happened, no strong feeling), chat (remarks, opinions,
  reactions, banter), about_you (asking the friend about themselves), ask_info (a question they want answered),
  ask_advice (asking for ideas or advice about their own situation);
- **feeling**, how it feels in the words: very_negative, negative, neutral, positive, very_positive.
Topics rotate (work, family, food, movies, health, ...) and some rows are written the way a speech recogniser hears
them (lower case, fillers, no punctuation). Writes data/text/friend_train.jsonl and friend_val.jsonl (90/10), never
over an existing file. The hand-written test set is data/text/friend_test.jsonl (not generated).
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from duet.llm import Llm  # noqa: E402

MODES = ["share_good", "share_bad", "share_story", "chat", "about_you", "ask_info", "ask_advice"]
FEELINGS = ["very_negative", "negative", "neutral", "positive", "very_positive"]
COMBOS = {  # (mode, feeling): what to ask the model for
    ("share_good", "very_positive"): "telling a close friend big, exciting good news that just happened to them",
    ("share_good", "positive"): "telling a friend something small and nice: a good meal, a fun evening, enjoying a "
                                "show, a sunny day, a little win; pleased, not over the moon",
    ("share_bad", "very_negative"): "telling a close friend something really hard that is happening to them: a loss, "
                                    "a breakup, being fired, a sick relative, feeling very alone or scared",
    ("share_bad", "negative"): "venting to a friend about something annoying or a bad day: traffic, a rude "
                               "coworker, being tired, a delay, a small disappointment",
    ("share_story", "neutral"): "telling a friend, matter-of-fact, what they're doing right now or what happened, "
                                "with no strong feeling",
    ("chat", "neutral"): "casual chat remarks to a friend: an opinion, agreeing, a thought, a reaction to what the "
                         "friend said; not telling a story and not asking for anything",
    ("chat", "positive"): "playful, happy chat remarks to a friend: a joke, teasing, enthusiasm about a shared topic",
    ("chat", "negative"): "grumpy or skeptical chat remarks to a friend: disagreeing, being unimpressed, mild "
                          "complaints about the topic",
    ("about_you", "neutral"): "asking the friend about themselves: their opinion, taste, favourites, how they are, "
                              "what they're up to",
    ("about_you", "positive"): "warmly asking the friend about themselves or complimenting them, curious and happy",
    ("ask_info", "neutral"): "asking the friend a factual question they want answered: a fact, how something works, "
                             "what a word means, a date or time, a quick explanation",
    ("ask_advice", "neutral"): "asking the friend for ideas or advice about their own situation or a decision",
    ("ask_advice", "negative"): "upset and asking the friend what to do about a problem they're struggling with",
}
# --set plain (text-reflex-v6): v5 read "We're having a baby" and "I passed my driving test" as neutral stories,
# because nearly every generated piece of good news carried feeling words; these state events plainly
PLAIN_COMBOS = {
    ("share_good", "very_positive"): "telling a close friend big good news as a plain statement of fact, with no "
                                     "feeling words at all (no 'excited', 'happy', 'amazing', no exclamation marks): "
                                     "a baby on the way, a passed exam, a new job, an accepted offer, an engagement",
    ("share_bad", "very_negative"): "telling a close friend something very sad or hard as a plain statement of fact, "
                                    "with no feeling words: a death, a diagnosis, a breakup, losing a job",
    ("share_good", "positive"): "mentioning something small and nice that happened, plainly, with mild words "
                                "('pretty good', 'nice'): a finished book, a good walk, a tasty dinner",
    ("chat", "negative"): "a mild negative opinion in passing, without strong words: something was kind of boring, "
                          "not their thing, overrated, meh",
    ("chat", "positive"): "a mild positive opinion in passing: they like an idea, a food, a song, a place",
}
TOPICS = ["work", "a new job", "school or studying", "family", "a partner", "friends", "their kids", "a pet",
          "food and cooking", "a movie or series", "music", "sports", "health or the body", "money", "moving house",
          "travel", "the weather", "a hobby", "their phone or computer", "a party", "the weekend", "sleep", "a car",
          "shopping", "a neighbour", "a game", "their city", "plans for tonight", "a birthday", "the news"]


def asr_style(t, rng):
    """Some rows as a speech recogniser writes them."""
    if rng.random() < 0.35:
        t = t.lower()
    if rng.random() < 0.3:
        t = re.sub(r"[,;:]", "", t)
    if rng.random() < 0.15:
        t = rng.choice(["um ", "uh ", "so ", "like ", "yeah "]) + t
    return t


def norm(t):
    return re.sub(r"[\s.,!?;:]+$", "", re.sub(r"\s+", " ", t).strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-combo", type=int, default=240)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--set", choices=["main", "plain"], default="main")
    a = ap.parse_args()
    out = ROOT / "data" / "text"
    tag = "friend" if a.set == "main" else "friend_plain"
    combos = COMBOS if a.set == "main" else PLAIN_COMBOS
    tr, va = out / f"{tag}_train.jsonl", out / f"{tag}_val.jsonl"
    if tr.exists() or va.exists():
        sys.exit(f"{tr} exists; never overwritten (rename it to regenerate)")
    llm = Llm()
    rng = random.Random(17)
    rows, seen, t0 = [], set(), time.time()
    for (mode, feeling), what in combos.items():
        got, tries = 0, 0
        while got < a.per_combo and tries < a.per_combo // a.batch * 4:
            tries += 1
            topic = rng.choice(TOPICS)
            prompt = (f"Write {a.batch} different things a person might say out loud on a phone call to a close friend, "
                      f"{what}. Topic: {topic}. Everyday spoken English, 4 to 25 words each, varied wording and "
                      f"length, first person, no names of famous people. Answer as JSON: "
                      f'{{"lines": ["...", ...]}}')
            try:
                txt = llm.complete([{"role": "user", "content": prompt}], max_tokens=600, temperature=0.95,
                                   fmt="json")
                lines = json.loads(txt).get("lines", [])
            except Exception as e:  # a bad batch is skipped
                print("skip:", type(e).__name__, e)
                continue
            for line in lines:
                if not isinstance(line, str):
                    continue
                t = norm(asr_style(line.strip(), rng))
                key = t.lower()
                if 3 <= len(t.split()) <= 30 and key not in seen:
                    seen.add(key)
                    rows.append({"text": t, "mode": mode, "feeling": feeling})
                    got += 1
        print(f"{mode:11s} {feeling:13s} {got:4d} rows ({time.time() - t0:.0f} s)", flush=True)
    rng.shuffle(rows)
    k = len(rows) // 10
    out.mkdir(parents=True, exist_ok=True)
    for path, part in ((va, rows[:k]), (tr, rows[k:])):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows) - k} train, {k} val rows")


if __name__ == "__main__":
    main()
