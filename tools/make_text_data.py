"""Training data for the text reflex (one bge-small trunk, five heads), written to data/text/{train,val}.jsonl.

Sources:
- Reflexx's 4,839 + 522 synthetic front-desk calls (frontdesk-bench/train/data, Gemma-4 generated, 26 intents).
  The 310 benchmark calls stay out: they are the test set, as in the paper.
- Dialogue acts written here as templates with slots (greetings, backchannels, stop, repeat, continue, thanks,
  goodbye, affirm, deny, small talk), alone and combined with calls ("Hi! ..." / "... thanks").
- Sentiment: calls wrapped in negative or positive clauses, plus standalone reactions. A call counts as neutral only
  when it has no word from either lexicon; otherwise its sentiment is left unlabelled (masked in the loss).
- Completeness: calls cut at a random word (incomplete) vs whole utterances (complete).

Every row: text, intent (or null), act (or null), greeting, request (0/1 or null), sentiment (neg/neu/pos or null),
complete (0/1). Trailing punctuation is stripped here and at runtime, so "complete" can't be read off a full stop.
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTDESK = ROOT.parent / "frontdesk-bench" / "train" / "data"
OUT = ROOT / "data" / "text"
rng = random.Random(11)

ACTS = ["request", "greeting", "smalltalk", "backchannel", "stop", "repeat", "continue", "thanks", "goodbye",
        "affirm", "deny"]
NAMES = ["Mark", "Jenny", "Carlos", "Priya", "Tom", "Aisha", "Daniel", "Mei", "Luca", "Sarah", "Kofi", "Olga"]
FIRMS = ["Bloom Bakery", "Rivera Plumbing", "Northside Dental", "Kite Studio", "Atlas Fitness", "Green Leaf Cafe"]

GREET = ["hi", "hello", "hey", "hi there", "hello there", "hey there", "good morning", "good afternoon",
         "good evening", "morning", "hiya", "howdy", "yeah hi", "oh hi", "hello hello", "hey hey", "hi hi",
         "hello, is this GRAPHIQ", "hi, is this GRAPHIQ", "hello, am I speaking to GRAPHIQ",
         "hi, am I through to the right place", "hey Sona", "hello Sona", "hi, it's {name}", "hello, this is {name}",
         "hi, {name} here", "hey, it's {name} from {firm}", "good morning, this is {name} from {firm}",
         "hello? anyone there", "hello, can you hear me", "hi, hello", "greetings", "yo", "hi good morning",
         "hey good afternoon", "hello, how are you", "hi, how are you doing", "hey, how's it going",
         "hello, how are you today", "hi, hope you're well", "good morning, how are you"]
SMALLTALK = ["how are you", "how's your day going", "are you a robot", "is this an AI", "am I talking to a real person",
             "what's your name", "who am I speaking with", "you sound nice", "nice weather today, isn't it",
             "how's business", "busy day", "are you real", "where are you based", "do you like your job",
             "what time is it there", "is this a recording", "you're funny", "that was quick", "you talk fast",
             "why are you so happy", "what's the weather like"]
# The date, the time, the caller's name, "can you hear me" and "talk to me" are answered by the brain's pattern layer
# whatever act is read, so they are not taught here: as small talk they pulled short requests ("do you build apps")
# with them (text-reflex-v2, 2026-11-03).
# Statements about the caller with no request in them, and feelings: small talk, not a question to route. Nothing
# that usually leads into a request ("I run a small bakery") is here.
STATEMENTS = ["I'm just very busy", "we're only two people", "I've been working all night", "it's been a long day",
              "I'm testing something", "I'm just trying this out", "I'm building a speech model",
              "my computer is talking", "I'm in the car right now", "I just got back from a meeting",
              "we work really efficiently", "I'm a bit tired today", "I was just thinking about it",
              "I'm just curious", "I'm doing fine", "I'm good", "not much, just working", "I'm excited",
              "I'm very excited", "I'm happy", "I'm a little nervous", "funny, I was just thinking that"]
BACKCHANNEL = ["mm-hm", "mhm", "uh-huh", "yeah", "yep", "right", "okay", "ok", "sure", "got it", "i see", "uh huh",
               "mm", "alright", "cool", "makes sense", "oh nice", "ah okay", "oh okay", "yeah yeah", "right right",
               "okay okay", "gotcha", "interesting", "oh really", "true", "fair enough", "ah", "oh", "hmm",
               "nice", "great", "perfect", "i understand", "sure sure", "yeah okay", "mm okay", "right okay"]
STOP = ["wait", "hold on", "stop", "sorry, wait", "hang on", "hang on a second", "one sec", "one second",
        "no no no", "excuse me", "can I stop you there", "sorry to interrupt", "actually wait", "hold on hold on",
        "let me stop you there", "wait wait", "sorry, sorry", "stop stop", "hold up", "just a moment",
        "wait a second", "okay stop", "sorry, can I just jump in", "can I interrupt", "hold on a sec",
        "no wait", "um wait", "hang on hang on", "sorry hold on", "let me interrupt you"]
REPEAT = ["sorry, what", "can you repeat that", "say that again", "pardon", "what did you say", "I didn't catch that",
          "come again", "could you say that again", "sorry, I missed that", "repeat that please", "what was that",
          "can you say that again more slowly", "sorry, you cut out", "I didn't hear you", "huh", "what",
          "say again", "could you repeat the last part", "sorry, can you go over that again", "you lost me"]
CONTINUE = ["go on", "sorry, continue", "please go ahead", "carry on", "you were saying", "go ahead",
            "sorry, go on", "keep going", "please continue", "finish what you were saying", "no, go on",
            "sorry about that, continue", "okay go on", "please carry on", "continue please", "and then"]
THANKS = ["thank you", "thanks", "thanks so much", "great, thanks", "perfect, thank you", "awesome thanks",
          "appreciate it", "thank you very much", "thanks a lot", "cheers", "much appreciated",
          "that's helpful, thanks", "thanks for your help", "brilliant, thank you", "lovely, thanks"]
GOODBYE = ["bye", "goodbye", "have a good day", "talk to you later", "that's all, bye", "thanks, bye",
           "okay bye bye", "bye bye", "see you", "take care", "have a nice day", "catch you later",
           "that's everything, goodbye", "alright, bye now", "good night", "speak soon"]
AFFIRM = ["yes please", "yes", "yeah that works", "sounds good", "correct", "exactly", "that's right", "absolutely",
          "definitely", "for sure", "yes, that's it", "that would be great", "please do", "let's do that",
          "yes, Thursday works", "that works for me", "sure, go ahead and book it", "yeah, sounds perfect",
          "yes I'd like that", "of course"]
DENY = ["no thanks", "no", "not really", "no, that's not it", "nope", "no thank you", "not now", "I don't think so",
        "no, that's wrong", "not at the moment", "no, that doesn't work", "nah", "no, I'm good", "maybe not",
        "no, I meant something else", "that's not what I asked"]
NEG = ["This is so frustrating.", "I'm really annoyed.", "Honestly, I'm pretty upset about this.",
       "I've been waiting forever and nobody calls me back.", "This is the third time I'm calling!",
       "I'm not happy with how this went.", "Ugh, nothing works.", "I'm kind of stressed out.",
       "I'm really disappointed.", "This is ridiculous.", "I'm fed up.", "I'm losing customers because of this!",
       "Nobody is answering my emails.", "I'm so tired of this.", "This has been a nightmare.",
       "I'm angry, to be honest.", "I'm worried.", "I'm at my wit's end.", "This is unacceptable.",
       "I feel ignored.", "It's been a terrible week.", "I'm really let down.", "This keeps happening!",
       "I'm sick of waiting.", "Sorry, I'm just really stressed."]
POS = ["I'm so excited about this!", "You guys did an amazing job on my friend's site.", "Love your work!",
       "Great news!", "I'm really happy.", "This is awesome.", "Thanks so much, you've been super helpful!",
       "I've heard great things about you.", "Your portfolio is gorgeous.", "I can't wait to get started!",
       "My friend loves her new site.", "You've been wonderful.", "This is exactly what I needed!",
       "That's fantastic.", "I'm thrilled.", "What a great service.", "You made my day.",
       "Everything has been perfect so far.", "I'm really looking forward to this.", "That's brilliant!"]
REACT_NEG = ["that's terrible", "oh no", "ugh", "seriously", "that's annoying", "that's not good", "great, just great",
             "this is a mess", "I hate this", "that sucks", "that's awful", "really? again"]
REACT_POS = ["awesome", "that's great", "wonderful", "amazing", "love it", "fantastic", "yay", "perfect, love that",
             "that's so cool", "excellent", "brilliant", "oh that's lovely"]
NEG_WORDS = {"frustrat", "annoy", "upset", "angry", "terrible", "awful", "ridiculous", "disappoint", "nightmare",
             "worried", "stress", "unacceptable", "hate", "broken", "losing", "lost", "complain", "furious", "fed up",
             "nobody", "again", "still not", "doesn't work", "not working", "down", "problem", "issue", "error",
             "wrong", "refund", "cancel"}
POS_WORDS = {"excit", "amazing", "love", "great", "awesome", "happy", "fantastic", "wonderful", "thrilled",
             "brilliant", "perfect", "gorgeous", "beautiful", "can't wait"}
GREET_RE = re.compile(r"^\s*(?:(?:yeah|oh|um|uh|so|okay)[, ]+)?(?:hi|hello|hey|hiya|howdy|greetings|"
                      r"good (?:morning|afternoon|evening))(?: there| guys| folks| everyone| sona)?(?:[,.!?]+\s*|\s+)"
                      r"(?=\S)", re.I)


def fill(t):
    return t.format(name=rng.choice(NAMES), firm=rng.choice(FIRMS))


def norm(t):
    """What the model sees, here and at runtime: trailing punctuation off, sometimes commas off (ASR style)."""
    t = re.sub(r"\s+", " ", t).strip()
    if rng.random() < 0.3:
        t = t.replace(",", "")
    return re.sub(r"[\s.,!?;:]+$", "", t)


def cap(t):
    return t[:1].upper() + t[1:] if rng.random() < 0.7 else t


def row(text, intent=None, act=None, greeting=None, request=None, sentiment=None, complete=1):
    return {"text": norm(text), "intent": intent, "act": act, "greeting": greeting, "request": request,
            "sentiment": sentiment, "complete": complete}


def sentiment_of(text):
    low = text.lower()
    neg = any(w in low for w in NEG_WORDS)
    pos = any(w in low for w in POS_WORDS)
    if not neg and not pos:
        return "neu"
    return None  # unknown: masked


def lexicon_sentiment(text):
    """For short statements: one side's words alone decide ("I'm very excited" is positive); both sides: masked."""
    low = text.lower()
    neg = any(w in low for w in NEG_WORDS)
    pos = any(w in low for w in POS_WORDS)
    return "neu" if not (neg or pos) else ("pos" if pos and not neg else "neg" if neg and not pos else None)


def strip_greeting(text):
    m = GREET_RE.match(text)
    return (text[m.end():].strip(), True) if m else (text, False)


def from_call(c):
    out = []
    text, intent = c["text"], c["label"]
    body, had = strip_greeting(text)
    body = cap(body) if body else text
    out.append(row(text, intent, "request", int(had), 1, sentiment_of(text)))
    if had:
        out.append(row(body, intent, "request", 0, 1, sentiment_of(body)))
    else:
        g = cap(fill(rng.choice(GREET)))
        out.append(row(f"{g}{rng.choice([', ', '. ', '! '])}{body}", intent, "request", 1, 1, sentiment_of(body)))
    r = rng.random()
    if r < 0.18:
        clause = rng.choice(NEG)
        t = f"{clause} {body}" if rng.random() < 0.6 else f"{body} {clause}"
        out.append(row(t, intent, "request", 0, 1, "neg"))
    elif r < 0.33:
        clause = rng.choice(POS)
        t = f"{clause} {body}" if rng.random() < 0.6 else f"{body} {clause}"
        out.append(row(t, intent, "request", 0, 1, "pos"))
    words = text.split()
    if len(words) >= 5 and rng.random() < 0.55:
        cut = rng.randint(2, max(2, int(len(words) * 0.85)))
        part = " ".join(words[:cut])
        trail = rng.choice(["", "", " um", " uh", " and", " so", " like"])
        out.append(row(part + trail, None, None, int(had) if cut > 2 else None, None, None, 0))
    return out


def standalone():
    out = []
    pools = {"greeting": GREET, "smalltalk": SMALLTALK + STATEMENTS, "backchannel": BACKCHANNEL, "stop": STOP,
             "repeat": REPEAT, "continue": CONTINUE, "thanks": THANKS, "goodbye": GOODBYE, "affirm": AFFIRM,
             "deny": DENY}
    reps = {"greeting": 10, "smalltalk": 10, "backchannel": 8, "stop": 10, "repeat": 10, "continue": 10,
            "thanks": 10, "goodbye": 10, "affirm": 10, "deny": 10}
    for act, pool in pools.items():
        for _ in range(reps[act]):
            for t in pool:
                statement = t in STATEMENTS
                t = cap(fill(t))
                sent = ("pos" if act == "thanks" else lexicon_sentiment(t) if statement else
                        sentiment_of(t) if act == "smalltalk" else "neu")
                if act == "greeting" and rng.random() < 0.25:
                    t = f"{t}{rng.choice(['. ', ', '])}{cap(rng.choice(SMALLTALK))}"
                # small talk is out of scope for the intent head too: unlabelled, chit-chat-shaped questions drifted
                # into random intents ("Who won the game last night?" -> partnership in text-reflex-v3)
                out.append(row(t, "out_of_scope" if act == "smalltalk" else None, act, int(act == "greeting"), 0,
                               sent))
    # greeting + small talk, small talk + greeting
    for _ in range(300):
        out.append(row(f"{cap(fill(rng.choice(GREET)))}, {rng.choice(SMALLTALK)}", None, "greeting", 1, 0, "neu"))
    # thanks + goodbye, stop + repeat, etc.: the last act wins
    combos = [(THANKS, GOODBYE, "goodbye"), (STOP, REPEAT, "repeat"), (BACKCHANNEL, CONTINUE, "continue"),
              (STOP, DENY, "deny"), (BACKCHANNEL, THANKS, "thanks"), (BACKCHANNEL, AFFIRM, "affirm")]
    for a, b, act in combos:
        for _ in range(120):
            out.append(row(f"{cap(rng.choice(a))}, {rng.choice(b)}", None, act, 0, 0,
                           "pos" if act == "thanks" else "neu"))
    for k in range(25):
        for t in REACT_NEG:
            out.append(row(cap(t), None, None, 0, 0, "neg"))
        for t in REACT_POS:
            out.append(row(cap(t), None, None, 0, 0, "pos"))
        for t in NEG:  # a complaint on its own may be a request ("nobody answers my emails"): act left unlabelled
            out.append(row(t, None, None, 0, 0, "neg"))
        # a good feeling said on its own is small talk (a thank-you among them is thanks); before 2026-11-03 these
        # had no act label, and the only positive acts the model saw were thank-yous ("I'm very excited" -> thanks)
        for t in POS:
            act = ("thanks" if re.search(r"\bthank", t, re.I) else "smalltalk") if k < 10 else None
            out.append(row(t, "out_of_scope" if act == "smalltalk" else None, act, 0, 0, "pos"))
    # cut-offs of short acts and greetings ("hi I wanted to", "so um", "and the")
    stems = ["I wanted to", "I was wondering if", "can you tell me about the", "so um", "and the", "I'd like to",
             "my website is", "the thing is", "what about", "could you", "I need a", "is it possible to",
             "we're looking for", "how much would", "do you guys", "so basically", "I'm calling because"]
    for _ in range(40):
        for s in stems:
            g = f"{cap(fill(rng.choice(GREET)))}, " if rng.random() < 0.4 else ""
            out.append(row(g + s, None, None, int(bool(g)), None, None, 0))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        calls = [json.loads(l) for l in open(FRONTDESK / f"{split}.jsonl", encoding="utf-8")]
        rows = [r for c in calls for r in from_call(c)]
        extra = standalone()
        rng.shuffle(extra)
        rows += extra if split == "train" else extra[:len(extra) // 8]
        rng.shuffle(rows)
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        acts = {a: sum(r["act"] == a for r in rows) for a in ACTS}
        sents = {s: sum(r["sentiment"] == s for r in rows) for s in ("neg", "neu", "pos")}
        print(split, len(rows), "rows | acts", acts, "| sentiment", sents,
              "| incomplete", sum(r["complete"] == 0 for r in rows))


if __name__ == "__main__":
    main()
