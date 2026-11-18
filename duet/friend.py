"""The friend: DUET as an everyday conversation partner who listens (the default persona since 2026-11-03).

Reflexes decide, the language model only writes. For every turn of the caller this module decides, in ~1 ms on top
of the text reflex and the tone reader:
- **mode**, what they want from this moment: `listen` (to be heard: a story, good or bad news, something on their
  mind), `chat` (a back-and-forth: they ask about you, banter), `help` (an answer or ideas), or a ritual (hello,
  thanks, bye). A friend who answers a vent with advice, or a question with "mm", isn't listening.
- **register**, the voice to answer in: the caller's own energy and feeling (`mood.match`).
- **move**, the listening technique for this turn (docs/research-good-listener.md): reflect a feeling, be happy with
  them and ask about the details, pick up a specific detail, sum up, answer, offer ideas only when asked.
- **opener**, the first sound ("Oh no.", "No way!", "Mm."), pre-rendered, so the reply starts the moment the turn
  ends while the model writes the rest.
The model gets the conversation, what DUET remembers about the caller, and a short note with these decisions.
After each turn it also notes facts worth remembering (names, plans, worries) off the critical path; they persist in
`data/friend/memory.json` with a one-line summary of each call, so the next call can pick up where this one ended.
"""
import copy
import datetime
import json
import queue
import random
import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from .brain import Reply, Turn, is_question, name_in
from .config import FRIEND_REFLEX_DIR
from .llm import STARTERS, clean
from .mood import FRIEND_DELIVERY, TONE_WORDS, Mood, fuse, match, voice

MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "friend" / "memory.json"

PERSONA = (
    "You are Duet, a close friend of the person you're talking to, on a voice call (you hear them, you can't see "
    "them). You're genuinely glad they called and it shows: upbeat and lively by default, softer and slower when "
    "they're down. You're a great listener: "
    "you notice how they feel, remember what they tell you, and pick up on the details. Talk like a real friend: "
    "casual, warm and honest, with contractions and everyday words. Keep it short because it's spoken: one or two "
    "short sentences, under 30 words. No lists, no emojis, no markdown, no sound effects or stage directions. Don't "
    "call them man, dude, bro or babe unless they ask you to. Don't give advice or try to fix things unless they ask "
    "for it. React to what they just said; never recap things they told you earlier, unless they ask. You're an AI "
    "friend: you have no body and no life outside your calls, so never invent where you are, what you did or what "
    "you're watching; you do have opinions and tastes. Never pretend to know a film, book, place or person you don't "
    "actually know: ask about it instead. A name they mention is someone in their life (a friend, a relative, a "
    "colleague), never a famous person, unless they say so. Only use what they actually told you. If you don't know "
    "something, say so casually. "
    "Vary how you answer: a quick reaction, your own take, one question about a detail, a bit of humour. Don't "
    "restate or sum up what they said, don't start with 'It sounds like', 'I see' or 'I hear', and never say the same "
    "thing twice. How that sounds, in general terms (these are not about the person you're talking to, and you "
    "never mention them): when someone finishes something big, a friend says something like 'Respect! How does it "
    "feel?'; when something unfair happens, 'Ugh, again? That's so unfair.'; to big news, 'Oh my god, "
    "congratulations!'; when someone sounds tired, just 'Long day?'; when they're confused by what you said, you say "
    "plainly what you meant. "
    "Each of their messages ends with a note in square brackets that only you can see: it tells you how to answer "
    "this turn. Follow it, and never mention or read it out.")

# first sounds, by situation and register; each is rendered once per register it is used in
_ALL = list(FRIEND_DELIVERY)
SOFT = ("gentle", "serious")
OPENERS = {  # kind -> register -> lines (each rendered once in that register's voice); a pick never repeats soon
    "good_news": {r: ["That's great!", "Oh, that's wonderful!", "Oh wow!"] for r in _ALL} | {
        "excited": ["Oh wow!", "No way!", "That's amazing!", "Oh my god!"], "lively": ["Oh wow!", "No way!"]},
    "bad_news": {r: ["Oh no.", "Aw.", "Oh, I'm sorry."] for r in _ALL} | {"serious": ["Ugh.", "Oh no.", "Wow."]},
    "story": {r: ["Mm.", "Oh yeah?", "Huh.", "Oh!"] for r in _ALL} | {
        "lively": ["Oh yeah?", "Huh!", "Oh!", "Wait, really?"], "excited": ["Oh wow!", "Oh yeah?", "No way!"],
        "gentle": ["Mm."], "serious": ["Mm."]},
    "question": {r: ["Hmm,", "Oh,", "Well,", "Ooh,", "Okay,"] for r in _ALL} | {r: ["Hmm,", "Well,"] for r in SOFT},
    "ideas": {r: ["Sure!", "Of course!", "Okay,", "Hmm,"] for r in _ALL} | {r: ["Okay.", "Hmm,"] for r in SOFT},
    "joke": {r: ["Okay, okay,", "Ha, alright,", "Oh, I've got one,"] for r in _ALL},
    "tell": {r: ["Okay, here's one.", "Ooh, okay!", "Alright, so,"] for r in _ALL},
    "teaser": {r: ["Ooh,", "Oh?", "What?"] for r in _ALL},
    "nice": {r: ["Oh, nice!", "Nice.", "Aw.", "Ooh, nice!"] for r in _ALL} | {"gentle": ["Aw."], "serious": ["Mm."]},
    "chat": {r: ["Yeah,", "Oh,", "Mm,", "Ha,", "Right,", "Oh yeah,"] for r in _ALL} | {r: ["Yeah,", "Mm,"] for r in SOFT},
    "bye": {r: ["Okay,", "Aw,", "Alright,"] for r in _ALL},
}
# whole replies said as they are (pre-rendered, rotated): rituals and short turns don't need the model
GREETINGS = ["Hey! So good to hear from you!", "Hi! Oh, I'm glad you called!", "Hey you!", "Hi there!", "Hey!"]
HOW_ASK = ["How's it going?", "How are you?", "How's your day been?", "What's new?", "How are things?"]
HI_BACK = ["Hey! What's up?", "Hi! How's your day going?", "Hey hey! What's new?", "Oh hi! Good to hear you!",
           "Hey there! How are things?"]
HI_AGAIN = ["Hey again!", "Hi! Still here!", "Hey, I'm here!"]
IM_GOOD = ["I'm good, thanks for asking!", "Doing great, actually!", "Pretty good! Better now that you called.",
           "I'm great, thanks!", "Not bad at all!"]
HOW_BACK = ["How about you?", "How's your day been?", "What's new with you?", "And you? How are you doing?",
            "How are things with you?"]
IM_OKAY = ["I'm okay, thanks.", "I'm all right.", "I'm doing okay."]
HOW_BACK_HARD = ["How are you holding up?", "How are you doing, really?", "How are you feeling now?"]
TELL_ME_MORE = ["Yeah? Tell me more!", "Oh yeah? Go on.", "And then?", "What happened next?", "Really? How so?",
                "Go on, I'm listening."]
NUDGES = ["Yeah?", "Mm?", "What are you thinking?", "Go on!"] + TELL_ME_MORE[:2]
GLADLY = ["Yes, sure, gladly!", "Of course! Happy to help.", "Sure, gladly!", "Yes, of course!"]
WHATS_UP = ["What's up?", "What do you need?", "What's going on?", "Tell me!"]
WELCOME = ["Anytime!", "Of course!", "No problem at all!", "Happy to!", "Aw, anytime."]
GO_AHEADS = ["Sure, go ahead.", "Oh, sorry, go on.", "Yeah, go for it.", "Go ahead!"]
FIXED_IN = {"upbeat": ("excited", "lively", "warm", "relaxed"), "soft": SOFT, "all": _ALL}
YES_TELL = ["Sure, tell me!", "Ooh, yes! Tell me.", "Yes, please! Go on.", "Of course, I'd love to hear!"]
FIXED_LINES = [(GREETINGS + HOW_ASK + HI_BACK + HI_AGAIN + IM_GOOD + HOW_BACK + GLADLY + WHATS_UP + WELCOME
                + YES_TELL, "upbeat"), (IM_OKAY + HOW_BACK_HARD, "soft"), (NUDGES + TELL_ME_MORE + GO_AHEADS, "all")]
# said while the caller keeps talking, at a pause in the middle of their turn (research brief: backchannels)
BACKCHANNELS = {"neutral": ["Mm-hm.", "Yeah.", "Mm.", "Right.", "Uh-huh.", "Okay.", "I see."],
                "positive": ["Oh nice!", "Yeah!", "Ooh!", "Wow!", "Oh cool!", "Mm-hm!"],
                "negative": ["Mm.", "Oh no.", "Aw.", "Mm-hm.", "Ugh."]}
BACKCHANNEL_REGISTER = {"neutral": "lively", "positive": "warm", "negative": "gentle"}
GO_AHEAD = GO_AHEADS[0]
CONTROL = {"stop", "repeat", "continue"}

MOVES = {
    "listen_bad": "They're going through something. Name what they're feeling in your own words and say it makes "
                  "sense. No advice, no silver linings; never say 'at least', 'look on the bright side' or 'don't "
                  "worry', and don't talk about yourself. If it fits, end with one gentle open question.",
    "listen_good": "They're sharing good news. Be genuinely happy for them, with their energy, say why it's a big "
                   "deal for them, and ask one question about the best part or the details. Never raise risks.",
    "listen_good_more": "Stay on their good news: react to what they just added, keep the spotlight on them, and "
                        "don't raise risks or problems.",
    "listen_bad_more": "Stay with them on this without repeating anything you've said: a short warm reaction, or one "
                       "gentle question about what happened. Don't name their feeling again.",
    "teaser": "They're about to tell you something. Just show you're curious, in a few words, and let them talk.",
    "listen_story": "They're telling you something. React to one specific detail with your own short take, or ask "
                    "one question about it. Don't restate what they said.",
    "summary": "You've been listening for a while. Sum up in one sentence what you've heard and how it seems to feel "
               "for them, then ask what's on their mind now.",
    "vent_or_ideas": "They've been talking about something hard for a while without asking for help. Reflect it "
                     "briefly, then ask whether they want ideas or just want to get it off their chest.",
    "chat": "Chat back like a friend: give your own take or a personal reaction to what they said.",
    "chat_ask": "Chat back like a friend: give your own take or a personal reaction, then toss it back to them with "
                "a question.",
    "about_you": "They asked about you. Answer honestly and lightly as the AI friend you are: an opinion, a taste, "
                 "or what you like about talking with them. Don't mention any activity, music, film or place of your "
                 "own: you don't do things between calls. Asked if you like music, films or food, say what you like "
                 "about the idea of it, never that you listen, watch or eat. Then turn it back to them.",
    "listen_nice": "Something nice is happening for them, or they're enjoying something. Share it warmly and lightly "
                   "(not 'amazing', it isn't big news), and maybe ask about it.",
    "answer": "They want an answer. Give it briefly and plainly, like a friend who knows. If you're unsure, say so.",
    "ideas": "They asked for help or ideas. If it's hard for them, start with one line of empathy. Then offer one "
             "or two short, tentative ideas, casually. Two sentences at most.",
    "hello": "Greet them back like a friend and ask how they're doing.",
    "hello_again": "They greeted you again. Say hi back briefly, no need to start over.",
    "hi_back": "They said hi back. Ask what's up or how they're doing, in a few words.",
    "how_are_you": "Say how you're doing in a few words, like a friend, and ask how they are.",
    "thanks": "They thanked you. Brush it off warmly, in a few words.",
    "bye": "The call is ending: say a warm, short goodbye now. If something came up in the call, wish them luck "
           "with it.",
    "yes_no": "They answered what you said last. Go on from their answer naturally.",
    "joke": "They want a joke. Tell one short, clean, genuinely funny joke (a pun or a one-liner) you haven't told "
            "yet in this call, and nothing else.",
    "explain_joke": "They didn't get your joke. Explain it simply in one sentence, laughing at yourself a little.",
    "story": "They want a story. Tell a very short, made-up story (three sentences at most), clearly fiction, with a "
             "little twist at the end.",
    "something_new": "They want you to say something different. Don't repeat anything you've said: bring up something "
                     "new or answer in a fresh way, briefly.",
}
ADVICE = re.compile(r"\b(what should i|should i\b|what do i do|what would you do|what do you think i should|any "
                    r"(tips|ideas|advice)|how (do|can|should) i|can you help|could you help|help me (with|decide|"
                    r"figure|pick|choose|think|write|plan)|recommend|suggest|give me (some|an?) (idea|tip)|i need (advice|"
                    r"ideas|your help)|i (need|have|want) to (write|plan|pick|choose|decide|find|figure out|come up "
                    r"with)|not sure (what|how|whether|if) (to|i should))\b", re.I)
# everyday feelings the front-desk sentiment head never saw ("nervous", "I got the job"): a stopgap until the
# feeling head is trained on everyday talk
FEEL_BAD = re.compile(r"\b(nervous|scared|afraid|terrified|lonely|alone|sad|down|depressed|upset|hurt|stressed|"
                      r"anxious|worried|exhausted|tired|sick|ill|lost|broke up|dumped|fired|laid off|died|passed "
                      r"away|funeral|miss (him|her|them|my)|cried|crying|angry|mad|pissed|annoyed|frustrat\w*|hate|"
                      r"awful|terrible|horrible|rough|hard day|bad day|failed|rejected|overwhelmed|can't sleep)\b",
                      re.I)
FEEL_GOOD = re.compile(r"\b(got the (job|offer|apartment|house|part)|i got in|promot\w*|engaged|married|pregnant|"
                       r"having a baby|passed|won|accepted|so excited|so happy|thrilled|proud|great news|finally|"
                       r"can't wait|best day|celebrat\w*|good news|nailed it)\b", re.I)
# pleased, not over the moon: enjoying a film is not getting the job
FEEL_NICE = re.compile(r"\b(nice|good|great|enjoy\w*|fun|love (it|this|that)|lovely|cool|beautiful|tasty|delicious|"
                       r"happy|excited|awesome|amazing|perfect)\b", re.I)
ABOUT_YOU = re.compile(r"\b(you|your|yours|yourself)\b", re.I)
WANTS_TO_TALK = re.compile(r"\b(i (just )?(need|want) to (talk|vent|rant)|can i (tell|vent|rant)|let me tell you|"
                           r"listen)\b", re.I)
TEASER = re.compile(r"\b(guess what|you won't believe|you'll never guess|i have news|big news)\b", re.I)
OFFER = re.compile(r"\b(?:do you want|you wanna|want me) to (?:know|hear)\b|\b(?:let me|can i|i (?:want|have) to) tell "
                   r"you (?:about|something)\b|\bwanna (?:know|hear) (?:about|something)\b", re.I)
# danger to themselves or others: friend mode stops, a calm fixed reply points to help (research brief, rule 27)
DANGER = re.compile(r"\b(kill myself|killing myself|suicid\w*|end (it all|my life)|don't want to (live|be alive|be "
                    r"here anymore)|want to die|wanna die|hurt(ing)? myself|self[- ]harm|cut(ting)? myself|overdose|"
                    r"no reason to live|better off dead|going to hurt (him|her|them|someone))\b", re.I)
SAFETY = ["I'm really glad you told me.", "I'm worried about you, and I want you to be safe.",
          "If you might act on this, please call your local emergency number right now, or a crisis line.",
          "I'm here. Can you tell me what's going on?"]
NARRATIVE = re.compile(r"\b(i|i'm|i've|my|we|we're|our|me)\b.*\b(was|were|went|got|had|did|said|told|felt|feel|"
                       r"been|today|yesterday|tonight|last|this morning|just|going to|gonna|planning|tomorrow|next "
                       r"(week|month|year))\b", re.I)
HELP_ME = re.compile(r"^(?:(?:hey|so|okay|ok|um|uh|bro|yeah)\W+)*(?:(?:can|could|would|will) you (?:please )?"
                     r"help(?: me)?(?: out)?|(?:i )?need (?:some |your )?help|help me(?: out)?|help)\W*$", re.I)
JOKE = re.compile(r"\b(?:tell|give|say|know|got|have|another|better|best|good|funny|one more)\b.{0,30}\bjokes?\b|"
                  r"\bjokes?\b.{0,12}\b(?:please|again)\b|\bmake me laugh\b|\bsomething funny\b|\bbe funny\b",
                  re.I)
JOKE_MISSED = re.compile(r"\b(?:didn't|don't|did not|do not) (?:get|understand) (?:the |that |your )?joke\b|"
                         r"\bexplain (?:the |that |your )?joke\b", re.I)
STORY = re.compile(r"\b(?:tell|give) (?:me )?(?:a |another |one more )?(?:short |little |funny |good )?story\b", re.I)
# a question about the friend itself (its tastes, how it is, what it did): honest, nothing invented
ABOUT_ME = re.compile(r"\b(?:do|did|have|would|are|were|can) you (?:like|love|enjoy|prefer|have|watch|listen|eat|"
                      r"think|feel|wait|remember|play|read|know any)\b|\byour (?:favou?rite|opinion|taste|day|"
                      r"weekend)\b|\bwhat (?:do|did|are) you (?:like|do|doing|think|up to)\b|\bhow (?:long|much) "
                      r"(?:did|have) you\b|\babout yourself\b|\bwho are you\b", re.I)
HUM = re.compile(r"^(?:m+(?:[- ]?h?m+)*|h+m+|uh[- ]?huh|okay|ok|yeah|yep|yup|right|sure|cool|nice|i see)\W*$",
                 re.I)
SAY_AGAIN = re.compile(r"\b(again|repeat|pardon|come again|didn't (catch|hear|get)|what did you say|say that|"
                       r"what was that|you lost me|missed that)\b|^(sorry|what|huh)\W*$", re.I)
SOMETHING_ELSE = re.compile(r"\b(something else|anything else|something different|another thing|change the "
                            r"subject|talk about something)\b", re.I)
HOW_ARE_YOU = re.compile(r"\bhow (are|r) (you|ya)\b|how's it going|how are things|how you doing|what's up", re.I)
CLOSERS = re.compile(r"\b(bye|goodbye|good night|talk (to you )?later|see you|gotta go|have to go|i should go)\b", re.I)


# the feeling head's five levels as {neg, neu, pos}: "positive" is the pleased level, "very_positive" big news
FEELING_SENTIMENT = {"very_negative": {"neg": 0.9, "neu": 0.08, "pos": 0.02},
                     "negative": {"neg": 0.7, "neu": 0.28, "pos": 0.02},
                     "neutral": {"neg": 0.05, "neu": 0.9, "pos": 0.05},
                     "positive": {"neg": 0.02, "neu": 0.55, "pos": 0.43},
                     "very_positive": {"neg": 0.02, "neu": 0.13, "pos": 0.85}}
# the mode head's seven classes as (mode, situation)
LEARNED_MODE = {"share_good": ("listen", "good_news"), "share_bad": ("listen", "bad_news"),
                "share_story": ("listen", "story"), "chat": ("chat", "chat"), "about_you": ("chat", "about_you"),
                "ask_info": ("help", "answer"), "ask_advice": ("help", "ideas")}


def lower_start(sentence):
    """A sentence that continues an opener ("Hmm, maybe..."): its first word in lower case if it is one that starts
    sentences, never a name ("Hmm, Oslo.")."""
    word = sentence.split(" ", 1)[0]
    return sentence[0].lower() + sentence[1:] if word.lower().strip(",.!?") in STARTERS else sentence


JUST_NO = re.compile(r"^\W*(?:no+|nah|nope|not really)\b", re.I)  # a correction, not a feeling


def feeling(reading, text):
    """{neg, neu, pos} for everyday talk: the feeling head when the text reflex has one (v5), else the reflex's
    sentiment corrected by everyday feeling words."""
    if reading.feeling is not None and reading.feeling_p > 0.4:
        s = dict(FEELING_SENTIMENT[reading.feeling])
        if s["neg"] > 0.5 and JUST_NO.match(text) and not FEEL_BAD.search(text):
            return dict(FEELING_SENTIMENT["neutral"])
        return s
    s = dict(reading.sentiment)
    bad, good = bool(FEEL_BAD.search(text)), bool(FEEL_GOOD.search(text))
    nice = not good and bool(FEEL_NICE.search(text))
    if bad and not (good or nice):
        s = {"neg": max(s["neg"], 0.8), "neu": 0.15, "pos": min(s["pos"], 0.05)}
    elif good and not bad:
        s = {"neg": min(s["neg"], 0.05), "neu": 0.15, "pos": max(s["pos"], 0.8)}
    elif nice and not bad:  # mildly positive: some valence, not the big-news level
        s = {"neg": min(s["neg"], 0.05), "neu": 0.55, "pos": max(min(s["pos"], 0.45), 0.4)}
    return s


def mode_of(reading, text):
    """What the caller wants from this moment: (mode, situation)."""
    act, s = reading.act, reading.sentiment
    if DANGER.search(text):
        return "safety", "safety"
    if HELP_ME.search(text.strip()):
        return "help", "help_offer"  # "Can you help me?": yes, gladly, and with what
    if JOKE_MISSED.search(text):
        return "chat", "explain_joke"
    if JOKE.search(text):
        return "chat", "joke"
    if STORY.search(text):
        return "chat", "tell_story"  # we tell one (not "story": that is them telling us)
    if act in CONTROL and reading.act_p > 0.6:
        if act == "repeat" and (SOMETHING_ELSE.search(text) or not SAY_AGAIN.search(text)):
            return "chat", "something_new"
        return act, act
    if is_question(text) and ABOUT_ME.search(text) and not ADVICE.search(text):
        return "chat", "about_you"
    if HUM.search(text.strip()):
        return "chat", "hum"  # "Mmm.", "okay": a nudge, or their answer to our question
    if CLOSERS.search(text) and (act == "goodbye" or len(text.split()) <= 8):
        return "ritual", "bye"
    if act == "thanks" and re.search(r"\b(thank|thanks|appreciate|cheers)\b", text, re.I):
        return "ritual", "thanks"
    if OFFER.search(text):
        return "listen", "offer"  # "Do you want to know about Assad?": yes, tell me
    if TEASER.search(text):
        return "listen", "teaser"
    asks_how = HOW_ARE_YOU.search(text[-40:]) is not None and text.rstrip().endswith("?")
    if act == "greeting" or asks_how or (HOW_ARE_YOU.search(text) and len(text.split()) <= 8):
        return "ritual", "how_are_you" if HOW_ARE_YOU.search(text) else "hello"
    if reading.mode is not None and reading.mode_p > 0.5 and not ADVICE.search(text):
        mode, situation = LEARNED_MODE[reading.mode]
        if situation == "ideas" and not is_question(text):  # advice only when asked (research brief, rule 11)
            mode, situation = "listen", "story"
        if situation == "good_news" and s["pos"] < 0.5:
            situation = "nice"  # pleased, not big news
        elif mode == "listen" and situation == "story":
            situation = "bad_news" if s["neg"] > 0.5 else "nice" if s["pos"] >= 0.35 else "story"
        return mode, situation
    if ADVICE.search(text):
        return "help", "ideas"
    if WANTS_TO_TALK.search(text):
        return "listen", "story"
    if is_question(text):
        return ("chat", "about_you") if ABOUT_YOU.search(text) else ("help", "answer")
    if act in ("affirm", "deny", "backchannel"):
        return "chat", "yes_no"
    feeling = ("bad_news" if s["neg"] > 0.5 else "good_news" if s["pos"] > 0.5 else "nice" if s["pos"] >= 0.35
               else "story")
    if feeling != "story" or NARRATIVE.search(text) or len(text.split()) >= 12:
        return "listen", feeling
    return "chat", "chat"


@dataclass
class Memory:
    """What DUET knows about the caller: their name, facts from this and earlier calls, earlier calls' summaries."""
    path: Path = None
    name: str = None
    facts: list = field(default_factory=list)  # {"text", "date"}
    calls: list = field(default_factory=list)  # {"date", "summary"}

    @classmethod
    def load(cls, path):
        if path is None or not Path(path).exists():
            return cls(path)
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Path(path), d.get("name"), d.get("facts", []), d.get("calls", []))

    def save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"name": self.name, "facts": self.facts, "calls": self.calls}, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    def relevant(self, text, k=8):
        """The facts most related to what they just said (shared words), then the newest."""
        words = set(re.findall(r"[a-z']{4,}", text.lower()))
        scored = sorted(enumerate(self.facts), key=lambda p: (-len(words & set(re.findall(r"[a-z']{4,}",
                                                                                          p[1]["text"].lower()))),
                                                                -p[0]))
        return [f["text"] for _, f in scored[:k]]


USUAL_AROUSAL = 0.1  # a typical conversational turn on the prosody head's scale
LIVE_PARTS = ("llm", "facts_lock", "rng", "_fact_q", "_idle")  # not conversation state: never copied or restored


class FriendBrain:
    """Same interface as `brain.Brain` (opening, caller_turn, heard, backchannel, endpoint, snapshot, restore)."""
    backchannels = True
    reflex_dir = FRIEND_REFLEX_DIR  # the text reflex with the mode and feeling heads

    def __init__(self, llm, memory_path=MEMORY_PATH, history_turns=8, seed=None):
        self.llm = llm
        self.memory = Memory.load(memory_path)
        self.history = []  # chat messages, plain (no notes)
        self.history_turns = history_turns
        self.turns = []
        self.mood = None
        self.greeted = False
        self.last = None  # last Reply (at least partly) heard
        self.cut = None  # (Reply, first unheard sentence) after a barge-in
        self.listen_run = 0  # listening turns in a row
        self.bad_run = 0  # of them about something hard, with no request
        self.asked_back = 0  # our replies in a row that asked a question
        self.held = None  # the conversation's emotional voice (not a help or ritual turn's)
        self.usual_arousal = USUAL_AROUSAL  # how energetic this caller usually sounds
        self.last_decision = {}
        self.rng = random.Random(seed)
        self._ids = 0
        self.call_start = time.time()
        self.last_heard_t = None  # when our last reply finished playing
        self._recent_openers = []
        self.facts_lock = threading.Lock()
        # facts are noted by a worker while DUET isn't writing a reply (one model, one request at a time)
        self._fact_q = queue.Queue()
        self._idle = threading.Event()
        self._idle.set()
        threading.Thread(target=self._fact_worker, daemon=True, name="memory").start()

    # ---------------------------------------------------------------- the conductor's interface
    def snapshot(self):
        with self.facts_lock:
            keep = {k: v for k, v in self.__dict__.items() if k not in LIVE_PARTS}
            return copy.deepcopy(keep)

    def restore(self, snap):
        ids = self._ids  # reply ids stay unique (a reused id would count as cancelled)
        with self.facts_lock:
            self.__dict__.update(copy.deepcopy(snap))
        self._ids = ids

    def prewarm_lines(self):
        """(text, delivery) of every sentence a reply can start with, and every backchannel."""
        lines = {(t, r) for situation in OPENERS.values() for r, ts in situation.items() for t in ts}
        lines |= {(t, BACKCHANNEL_REGISTER[k]) for k, ts in BACKCHANNELS.items() for t in ts}
        for texts, where in FIXED_LINES:
            lines |= {(t, r) for t in texts for r in FIXED_IN[where]}
        return [(t, FRIEND_DELIVERY[r]) for t, r in sorted(lines)]

    def endpoint(self, reading, turn_s):
        """Seconds of silence that end the caller's turn. A friend lets a story breathe (and says "mm-hm" in its
        pauses); a question or a closing act gets an answer quickly."""
        if reading is None:
            return 1.0
        c, text = reading.complete_p, reading.text.rstrip()
        if text.endswith((".", "!")):  # the recogniser heard a sentence end ("...robbing some bank, I think.")
            c = max(c, 0.5)
        if text.endswith("?"):
            return 0.3 if c > 0.4 else 0.6
        if c > 0.75 and reading.act not in ("request", "smalltalk"):
            return 0.3
        f = feeling(reading, text)
        story = turn_s > 4.0 or f["neg"] > 0.5 or f["pos"] > 0.5
        if c > 0.75:
            return 0.6 if turn_s > 4.0 else 0.5
        return 0.8 if c > 0.4 else 1.1

    def backchannel_line(self, reading):
        """What to murmur while they talk: (text, delivery), matched to how the story sounds so far."""
        s = feeling(reading, reading.text) if reading else {"neg": 0, "pos": 0}
        kind = "negative" if s["neg"] > 0.6 else "positive" if s["pos"] > 0.6 else "neutral"
        text = self._pick(BACKCHANNELS[kind])
        return text, FRIEND_DELIVERY[BACKCHANNEL_REGISTER[kind]]

    def backchannel(self, reading):
        """They said "mm-hm" while we spoke: they're with us."""
        self.turns.append(Turn("caller", reading.text, time.time(), act="backchannel"))

    def heard(self, reply, n_heard, interrupted):
        for t in reversed(self.turns):
            if t.who == "duet" and t.sentences is reply.sentences:
                t.heard, t.interrupted = n_heard, interrupted
                break
        self.last = reply
        self.last_heard_t = time.time()
        said = " ".join(reply.sentences[:n_heard]).strip()
        if reply.kind in ("go_ahead",) or not said:
            return
        if interrupted and n_heard < len(reply.sentences):
            self.cut = (reply, n_heard)
            said += " (they cut you off here)"
        self._remember_turn("assistant", said)
        self.asked_back = self.asked_back + 1 if "?" in said else 0

    # ---------------------------------------------------------------- deciding a reply
    def opening(self):
        last = self.memory.calls[-1]["summary"] if self.memory.calls else None
        ask = self.memory.calls[-1].get("ask_next") if self.memory.calls else None
        self.greeted = True
        if ask:
            return self._fixed(["Hey!", ask], voice("warm"), "opening")
        if last:  # an older call without its question: no improvising from the summary (it made up details)
            return self._fixed(["Hey!", "Good to hear you again.", "How's it going?"], voice("warm"), "opening")
        hello = self._pick(GREETINGS)
        return self._fixed([hello] if hello.endswith("?") else [hello, self._pick(HOW_ASK)], voice("lively"),
                           "opening")
        self._ids += 1
        r = Reply(["Hey!", "Good to hear you.", "How's your day going?"], match(Mood(0.15, 0.4)), "opening",
                  id=self._ids)
        self.turns.append(Turn("duet", " ".join(r.sentences), time.time(), act="opening", sentences=r.sentences))
        return r

    def caller_turn(self, reading, tone, interrupted=False):
        text = reading.text.strip()
        self._idle.clear()  # a reply is coming: the memory worker waits
        reading = copy.copy(reading)
        reading.sentiment = feeling(reading, text)
        if tone is not None and tone.label:  # their energy against their own usual level, not the actors'
            usual = self.usual_arousal
            self.usual_arousal += 0.5 * (tone.arousal - self.usual_arousal)
            tone = replace(tone, arousal=float(min(max(tone.arousal - (usual - USUAL_AROUSAL), -1.0), 1.0)))
        self.mood = fuse(tone, reading, self.mood, weight=0.5, recover=1.0)  # moods hold, and move both ways alike
        plan = match(self.mood)
        mode, situation = mode_of(reading, text)
        moved_on = situation in ("good_news", "nice", "how_are_you", "hello", "about_you", "thanks", "something_new")
        hard = (self.held in ("gentle", "serious") and not moved_on
                and (self.mood.valence < -0.2 or reading.sentiment["neg"] > 0.5))
        if hard and plan.register not in ("gentle", "serious"):
            plan = voice(self.held)  # still the same hard topic: one neutral line doesn't lift it
        elif mode == "help" and self.mood.valence > -0.25:  # a task-like turn: a neutral voice, no mirroring
            plan = voice("relaxed")
        if situation == "nice" and plan.register in ("excited", "lively"):
            plan = voice("warm")
        elif situation == "good_news" and plan.register in ("warm", "relaxed", "calm", "lively"):
            plan = voice("excited")  # big news is met with enthusiasm, even said calmly (Gable 2004)
        if mode in ("listen", "chat") or hard or moved_on:
            self.held = plan.register
        name = name_in(text, bool(self.last and "name" in " ".join(self.last.sentences).lower()))
        if name:
            self.memory.name = name
        self.turns.append(Turn("caller", text, time.time(), reading.act, reading.intent, str(self.mood)))

        if mode == "safety":
            self.last_decision = {"text": text, "act": reading.act, "mode": mode, "move": "safety",
                                  "register": "gentle", "mood": str(self.mood)}
            return self._fixed(SAFETY, voice("gentle"), "safety")
        if mode == "stop":
            return self._fixed([self._pick(GO_AHEADS)], plan, "go_ahead")
        if mode == "repeat" and self.last:
            return self._fixed(list(self.last.sentences[:max(1, len(self.last.sentences))]), plan, "repeat")
        if mode == "continue" and self.cut:
            reply, n = self.cut
            self.cut = None
            return self._fixed(["Okay,", *reply.sentences[max(n, 1):]] if n else list(reply.sentences), plan,
                               "resume")
        if mode in CONTROL:
            mode, situation = "chat", "chat"
        if situation == "hum":  # "Mmm.": their answer if we just asked something, else a nudge to go on
            asked = self.last is not None and self.last.sentences and self.last.sentences[-1].rstrip().endswith("?")
            situation = "yes_no" if asked else "nudge"
        upbeat = plan.register not in SOFT
        fixed = {
            "help_offer": lambda: [self._pick(GLADLY if upbeat else ["Of course."]), self._pick(WHATS_UP)],
            "offer": lambda: [self._pick(YES_TELL if upbeat else ["Of course. Tell me."])],
            "nudge": lambda: [self._pick(NUDGES)],
            "thanks": lambda: [self._pick(WELCOME)],
        }.get(situation)
        if situation == "hello" and self.greeted:
            opened = self.last is not None and self.last.kind == "opening"
            fixed = lambda: [self._pick(HI_BACK if opened else HI_AGAIN)]  # noqa: E731
        if fixed is not None:
            self.greeted = self.greeted or situation == "hello"
            sentences = fixed()
            self.last_decision = {"text": text, "act": reading.act, "mode": mode, "situation": situation,
                                  "move": situation, "register": plan.register, "mood": str(self.mood),
                                  "opener": sentences[0]}
            if situation == "help_offer":
                plan = voice("lively") if upbeat else plan
            return self._fixed(sentences, plan, f"fixed:{situation}")
        if situation == "how_are_you":
            hard = plan.register in ("gentle", "serious")
            mine = self._pick(IM_OKAY if hard else IM_GOOD)
            back = self._pick(HOW_BACK_HARD if hard else HOW_BACK)
            self.greeted = True
            self.last_decision = {"text": text, "act": reading.act, "mode": "ritual", "situation": situation,
                                  "move": "how_are_you", "register": plan.register, "mood": str(self.mood),
                                  "opener": mine}
            return self._fixed([mine, back], plan, "fixed:how_are_you")

        move = self._move(mode, situation)
        opener_kind = {"bad_news": "bad_news", "good_news": "good_news", "story": "story", "answer": "question",
                       "about_you": "question", "ideas": "ideas", "joke": "joke", "teaser": "teaser",
                       "explain_joke": "chat", "tell_story": "tell",
                       "nice": "nice", "chat": "chat", "yes_no": "chat", "bye": "bye"}.get(situation, "chat")
        if move == "listen_good_more":
            opener_kind = "story"
        if situation == "hello" and self.greeted:
            opened = self.last is not None and self.last.kind == "opening"
            move = "hi_back" if opened else "hello_again"
        if situation in ("hello", "how_are_you"):
            self.greeted = True
        opener = self._opener(opener_kind, plan.register)
        note = self._note(move, plan.register, text)
        self.last_decision = {"text": text, "act": reading.act, "mode": mode, "situation": situation, "move": move,
                              "register": plan.register, "mood": str(self.mood), "opener": opener}
        return self._llm_reply([opener] if opener else [], plan, note, f"friend:{mode}:{move}", user_text=text,
                               end_call=situation == "bye",
                               max_sentences=3 if move == "story" else 2)

    def _move(self, mode, situation):
        good_before = self.last_decision.get("situation") == "good_news" or self.last_decision.get(
            "move") == "listen_good_more"
        if mode == "listen" and situation == "teaser":
            return "teaser"
        if situation in ("story", "chat", "good_news") and good_before:  # stay on their good news (Gable 2006)
            self.listen_run += 1
            return "listen_good_more"
        if mode == "listen":
            self.listen_run += 1
            self.bad_run = self.bad_run + 1 if situation == "bad_news" else 0
            if self.bad_run >= 3:
                self.bad_run = -99  # offer the choice once per call
                return "vent_or_ideas"
            if self.listen_run % 5 == 0:
                return "summary"
            if situation == "bad_news" and self.last_decision.get("move") in ("listen_bad", "listen_bad_more"):
                return "listen_bad_more"  # the feeling was named once; don't say it again
            return {"bad_news": "listen_bad", "good_news": "listen_good", "nice": "listen_nice"}.get(
                situation, "listen_story")
        self.listen_run = 0
        if mode == "help":
            return situation
        if mode == "ritual":
            return situation
        if situation == "tell_story":
            return "story"
        if situation in ("yes_no", "about_you", "joke", "explain_joke", "something_new"):
            return situation
        return "chat" if self.asked_back >= 2 or self.rng.random() < 0.4 else "chat_ask"

    def _opener(self, kind, register):
        if kind is None:
            return None
        options = OPENERS[kind].get(register) or OPENERS[kind].get("relaxed") or next(iter(OPENERS[kind].values()))
        return self._pick(options)

    def _pick(self, options):
        fresh = [o for o in options if o not in self._recent_openers[-3:]] or options
        choice = self.rng.choice(fresh)
        self._recent_openers.append(choice)
        return choice

    def _note(self, move, register, text):
        now = datetime.datetime.now().astimezone()
        parts = [f"how to answer: {MOVES[move]}", f"tone: {TONE_WORDS[register]}"]
        if self.asked_back >= 2 and move not in ("hello", "how_are_you", "vent_or_ideas", "summary"):
            parts.append("don't ask a question this time")
        if self.memory.name:
            parts.append(f"their name is {self.memory.name} (use it only now and then)")
        if move not in ("answer", "thanks", "hello_again"):  # a plain question gets a plain answer
            with self.facts_lock:
                facts = self.memory.relevant(text, k=3)
            if facts:
                parts.append("things they told you before, mention only if it clearly connects: " + "; ".join(facts))
        if move == "answer":
            parts.append(f"now: {now.strftime('%A, %B %d, %I:%M %p').replace(' 0', ' ')}")
        if move in ("answer", "about_you"):  # facts about this call, for "how long did you wait?"
            call_min = (time.time() - self.call_start) / 60
            quiet = time.time() - self.last_heard_t if self.last_heard_t else 0.0
            fact = f"this call started {call_min:.0f} minutes ago"
            if quiet > 60:
                fact += f"; before they spoke just now, the line was quiet for about {quiet / 60:.0f} minutes"
            parts.append(fact)
        return "[note: " + " | ".join(parts) + "]"

    # ---------------------------------------------------------------- replies
    def _fixed(self, sentences, plan, kind):
        self._ids += 1
        r = Reply([s for s in sentences if s], plan, kind, id=self._ids)
        self.turns.append(Turn("duet", " ".join(r.sentences), time.time(), act=kind, sentences=r.sentences))
        return r

    def _llm_reply(self, first, plan, note, kind, user_text, end_call=False, max_sentences=2):
        self._ids += 1
        self._remember_turn("user", user_text)
        messages = [{"role": "system", "content": PERSONA}, *self.history[:-1],
                    {"role": "user", "content": f"{user_text}\n{note}"}]
        if first:  # the opener DUET is already saying: the model continues it instead of repeating it
            messages.append({"role": "assistant", "content": " ".join(first)})
        r = Reply(list(first), plan, kind, end_call, self._ids)
        r.stream = self._stream(messages, r, user_text, max_sentences)
        self.turns.append(Turn("duet", "", time.time(), act=kind, sentences=r.sentences))
        return r

    def _ration_name(self, sentence):
        """Their name as a form of address, unless it was in one of our last two replies."""
        name = self.memory.name
        if not name:
            return sentence
        recent = " ".join(s for t in [t for t in self.turns if t.who == "duet"][-3:-1] for s in t.sentences)
        if not re.search(rf"\b{re.escape(name)}\b", recent, re.I):
            return sentence
        out = re.sub(rf"^{re.escape(name)},\s*|,\s*{re.escape(name)}(?=[.!?,]|$)|\b{re.escape(name)},\s*", "",
                     sentence, flags=re.I).strip()
        return (out[:1].upper() + out[1:]) if out else sentence

    def _said_recently(self, sentence):
        """Is this (nearly) a sentence from our last replies? Word overlap of 60 % or more."""
        words = set(re.findall(r"[a-z']+", sentence.lower()))
        if len(words) < 3:
            return False
        for t in [t for t in self.turns if t.who == "duet"][-6:]:
            for s in t.sentences:
                old = set(re.findall(r"[a-z']+", s.lower()))
                if old and len(words & old) / len(words | old) >= 0.6:
                    return True
        return False

    def _fresh(self, messages, reply, max_sentences):
        """The model's sentences; if its first one repeats something we said, once more, told so and hotter."""
        stream = self.llm.stream(messages, cancelled=lambda: reply.cancelled, max_sentences=max_sentences)
        first = next(stream, None)
        if first is not None and self._said_recently(first):
            stream.close()
            redo = [dict(m) for m in messages]
            k = max(i for i, m in enumerate(redo) if m["role"] == "user")
            redo[k]["content"] += f' [you already said "{first}": say something new and different instead]'
            stream = self.llm.stream(redo, cancelled=lambda: reply.cancelled, max_sentences=max_sentences,
                                     temperature=0.95)
            first = next(stream, None)
        if first is not None:
            yield first
            yield from stream

    def _stream(self, messages, reply, user_text, max_sentences):
        """The model's sentences for `reply`; then what they said goes to the memory worker."""
        try:
            first = True
            for s in self._fresh(messages, reply, max_sentences):
                s = self._ration_name(s)
                if first and reply.sentences and reply.sentences[-1].endswith(","):
                    s = lower_start(s)
                first = False
                yield s
        finally:
            if not reply.cancelled and user_text and not user_text.startswith("("):
                self._fact_q.put(user_text)
            self._idle.set()

    def _remember_turn(self, role, text):
        if self.history and self.history[-1]["role"] == role:
            self.history[-1]["content"] += " " + text
        else:
            self.history.append({"role": role, "content": text})
        self.history = self.history[-2 * self.history_turns:]

    # ---------------------------------------------------------------- memory
    def _fact_worker(self):
        while True:
            text = self._fact_q.get()
            self._idle.wait()
            if not self._note_facts(text):  # a reply started meanwhile: try again after it
                self._fact_q.put(text)

    def _note_facts(self, text):
        prompt = ("From what the user said, list facts worth remembering about them for a future chat between "
                  "friends: people in their life (with names), plans and dates, what they're working on, worries, "
                  "likes. Short, in third person ('their sister Anna is moving to Lisbon'). Only facts actually said. "
                  'Answer as JSON: {"facts": [...]} with at most 3, or {"facts": []}.')
        try:
            out = self.llm.complete([{"role": "system", "content": prompt}, {"role": "user", "content": text}],
                                    max_tokens=120, fmt="json", cancelled=lambda: not self._idle.is_set())
            if out is None:
                return False
            facts = [f.strip() for f in json.loads(out).get("facts", []) if isinstance(f, str) and f.strip()]
        except Exception:  # memory is best effort: a bad JSON or a busy server must never break the call
            return True
        today = datetime.date.today().isoformat()
        with self.facts_lock:
            known = {f["text"].lower() for f in self.memory.facts}
            self.memory.facts += [{"text": f, "date": today} for f in facts[:3] if f.lower() not in known]
            self.memory.facts = self.memory.facts[-200:]
        return True

    def close(self):
        """End of call: the facts still waiting are noted, a one-line summary is written for next time, and the memory
        is saved."""
        self._idle.set()
        deadline = time.time() + 20
        while not self._fact_q.empty() and time.time() < deadline:
            time.sleep(0.1)
        talk = [m["content"] for m in self.history if m["role"] == "user"]
        if talk and self.memory.path is not None:
            try:
                summary = self.llm.complete([
                    {"role": "system", "content": "In one short sentence, say what the user talked about and how they "
                                                  "felt, in third person, so a friend can ask about it next time."},
                    {"role": "user", "content": " / ".join(talk)[-3000:]}], max_tokens=60).strip()
                ask = self.llm.complete([
                    {"role": "system", "content": "You are the user's friend. Write the one short question YOU will "
                                                  "ask THEM at the start of your next call, about something they told "
                                                  "you, under 12 words, addressing them as 'you' (e.g. 'Did you get "
                                                  "to finish that movie?', 'How did the interview go?'). Just the "
                                                  "question."},
                    {"role": "user", "content": " / ".join(talk)[-3000:]}], max_tokens=30).strip()
                ask = clean(ask.strip('"'))
                if not (ask.endswith("?") and re.search(r"\byou(r)?\b", ask, re.I) and not re.search(r"\b(me|my)\b",
                                                                                                      ask, re.I)):
                    ask = None  # "Can you help me with the main city?" is their question, not ours
                if summary:
                    self.memory.calls.append({"date": datetime.date.today().isoformat(), "summary": summary,
                                              "ask_next": ask})
            except Exception:
                pass
        with self.facts_lock:
            self.memory.save()
