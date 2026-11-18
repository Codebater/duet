"""The brain: the call's context, and what to say next.

The context is explicit state, not a prompt (Reflexx's "state as data"): every turn with its text, act, intent and
mood; whether greetings were exchanged; the last reply and how much of it the caller actually heard; a reply cut
off by the caller (so "go on" resumes at the right sentence); and the question DUET last asked (so a "yeah" answers
it). Greeting and intent are separate decisions: "Hi" alone gets a greeting back and an invitation; "Hi, how much is
a store?" gets a short greeting back and the answer; "How much is a store?" gets the answer.
"""
import copy
import datetime
import re
import time
from dataclasses import dataclass, field, replace
from zoneinfo import ZoneInfo

from . import replies as R
from .mood import REGISTER_DELIVERY, Mood, fuse, respond
from .voice import sentences as split

OFFICE = ZoneInfo("America/New_York")  # Sona answers from Fort Lauderdale

QUESTION = re.compile(r"^(?:what|what's|whats|who|who's|where|when|why|how|which|do|does|did|can|could|would|will|"
                      r"is|are|am|was|were|should|have|has)\b", re.I)
DATE_Q = re.compile(r"\bwhat(?:'s| is)? (?:the |today'?s )?(?:date|day)\b|\bwhich day\b|\bwhat(?:'s| is) today\b|"
                    r"\btoday'?s date\b", re.I)
TIME_Q = re.compile(r"\bwhat time\b|\btime is it\b|\bwhat'?s the time\b", re.I)
NAME_Q = re.compile(r"\b(?:know|remember|what'?s|what is|say) my name\b|\bwho am i\b|\bremember me\b|"
                    r"\bknow who i am\b", re.I)
HEAR_Q = re.compile(r"\bcan you hear me\b|\bare you (?:still )?there\b|\bis (?:any|some)one there\b", re.I)
TALK_Q = re.compile(r"\b(?:talk|speak)(?:ing)? (?:to|with) me\b|\bsay something\b|\bwhat (?:else )?can you do\b|"
                    r"\btell me about yourself\b|\bwhat do you do\b", re.I)
NAME_SAID = re.compile(r"\b(?:my name is|my name'?s|call me|name is)\s+([A-Za-z][A-Za-z'-]+)", re.I)
NAME_INTRO = re.compile(r"\bthis is\s+([A-Z][A-Za-z'-]+)")  # "Hi, this is Mark": only a capitalised word
NAME_REPLY = re.compile(r"^(?:(?:it'?s|it is|i'?m|i am|this is|my name is|my name'?s|call me)\s+)?([A-Za-z][A-Za-z'-]+)"
                        r"[.!?]*$", re.I)  # the answer to "what's your name?": "Alex", "It's Alex"
THANKS_WORD = re.compile(r"\b(?:thank|thanks|thx|appreciate|appreciated|cheers|grateful)\b", re.I)
NOT_NAMES = {"a", "an", "the", "not", "very", "just", "so", "really", "sorry", "fine", "good", "here", "okay", "ok",
             "yes", "no", "yeah", "well", "great", "busy", "excited", "happy", "calling", "looking", "sure", "um", "uh",
             "hi", "hello", "hey", "graphiq", "graphic", "sona", "nobody", "nothing", "me", "you", "it"}


def is_question(text):
    t = text.strip()
    return t.endswith("?") or bool(QUESTION.match(t))


def name_in(text, asked):
    """A caller's name if they just gave it (any time with "my name is", only right after we asked otherwise)."""
    t = text.strip()
    for pattern, ok in ((NAME_SAID, True), (NAME_INTRO, True), (NAME_REPLY, asked and len(t.split()) <= 4)):
        m = pattern.search(t) if ok else None
        if m and m.group(1).lower() not in NOT_NAMES:
            return m.group(1)[:1].upper() + m.group(1)[1:]
    return None


@dataclass
class Turn:
    who: str  # caller | duet
    text: str
    t: float
    act: str = ""
    intent: str = ""
    mood: str = ""
    sentences: list = field(default_factory=list)
    heard: int = 0  # duet: sentences played in full
    interrupted: bool = False


@dataclass
class Reply:
    sentences: list  # said in order; a streamed reply appends to it as the model writes
    plan: object  # mood.Plan
    kind: str
    end_call: bool = False
    id: int = 0
    stream: object = None  # more sentences to come (friend.FriendBrain: the language model), or None
    cancelled: bool = False  # set by the conductor: stop writing and saying it

    def __deepcopy__(self, memo):
        """A copy for a brain snapshot: everything but the live stream (a generator can't be copied, and a restored
        brain never resumes an old reply)."""
        r = copy.copy(self)
        memo[id(self)] = r
        r.sentences = copy.deepcopy(self.sentences, memo)  # memo keeps it the same list as its Turn's copy
        r.plan = copy.deepcopy(self.plan, memo)
        r.stream = None
        return r


class Brain:
    def __init__(self):
        self.turns = []
        self.mood = None
        self.lift_turns = 0
        self.greeted = False  # the caller's greeting has been answered
        self.last = None  # last Reply that was (at least partly) spoken
        self.cut = None  # (Reply, first unheard sentence) after a barge-in
        self.asked = None  # what our last question offered: "email", or ("clarify", intent)
        self.sour = 0  # turns left in which the caller counts as recovering from frustration
        self.name = None  # the caller's name, once they say it
        self._ids = 0

    backchannels = False  # the front desk doesn't murmur while the caller talks

    # ---------------------------------------------------------------- the conductor's interface
    def snapshot(self):
        return copy.deepcopy(self.__dict__)

    def restore(self, snap):
        ids = self._ids  # reply ids stay unique (a reused id would count as cancelled)
        self.__dict__.update(copy.deepcopy(snap))
        self._ids = ids

    def prewarm_lines(self):
        """(text, delivery) of every sentence a reply can start with: the openers, and each answer's first line."""
        lines = [(t, REGISTER_DELIVERY[r]) for t, r in R.OPENERS]
        lines += [(split(answer)[0], d) for answer in R.ANSWERS.values() for d in REGISTER_DELIVERY.values()]
        return lines

    def endpoint(self, reading, turn_s):
        """Seconds of silence that end the caller's turn: a question (the recogniser writes its "?") or a short
        closing act hands the turn over; a finished statement may have more coming (the long-talker simulation lost
        the floor at every sentence pause at 0.25 s); unfinished words wait longest."""
        if reading is None:
            return 0.9
        c = reading.complete_p
        if reading.text.rstrip().endswith("?"):
            return 0.25 if c > 0.4 else 0.55
        if c > 0.75:
            return 0.5 if reading.act in ("request", "smalltalk") else 0.25
        return 0.55 if c > 0.4 else 0.9

    def vary(self, options):
        """The first of these lines that wasn't said in our last two replies (the first live call heard
        "Ha, fair enough! What can I do for you today?" four times)."""
        recent = [s for t in [t for t in self.turns if t.who == "duet"][-2:] for s in t.sentences]
        fresh = [o for o in options if o not in recent]
        return fresh[0] if fresh else options[len(self.turns) % len(options)]

    def now(self):
        return datetime.datetime.now(OFFICE)

    def _reply(self, sentences, plan, kind, end_call=False):
        self._ids += 1
        r = Reply([x for s in sentences if s for x in split(s)], plan, kind, end_call, self._ids)
        self.turns.append(Turn("duet", " ".join(r.sentences), time.time(), act=kind, mood=plan.register,
                               sentences=r.sentences))
        return r

    def opening(self):
        return self._reply([R.OPENING, R.OPENING_INVITE], respond(Mood(0.1, 0.2)), "opening")

    def heard(self, reply, n_heard, interrupted):
        """The conductor reports how much of a reply was played before it ended or was cut."""
        for t in reversed(self.turns):
            if t.who == "duet" and t.sentences == reply.sentences:
                t.heard, t.interrupted = n_heard, interrupted
                break
        self.last = reply
        if interrupted and n_heard < len(reply.sentences) and reply.kind.startswith(("answer", "resume", "repeat")):
            self.cut = (reply, n_heard)  # kept through "Sure, go ahead" until resumed or replaced by a new answer

    def backchannel(self, reading):
        """The caller said "mm-hm" while we spoke: they're with us. Noted, nothing to say."""
        self.turns.append(Turn("caller", reading.text, time.time(), act="backchannel"))

    def caller_turn(self, reading, tone, interrupted=False):
        self.mood = fuse(tone, reading, self.mood)
        self.lift_turns = self.lift_turns + 1 if self.mood.valence > -0.1 else 0
        self.sour = 2 if self.mood.valence < -0.25 else max(self.sour - 1, 0)
        plan = respond(self.mood, self.lift_turns, recovering=self.sour > 0 and self.mood.valence >= -0.25)
        r, act, text = plan.register, reading.act, reading.text
        self.turns.append(Turn("caller", text, time.time(), act, reading.intent, str(self.mood)))
        low = text.lower()
        how_are_you = re.search(r"\bhow (are|r) (you|ya)\b|how's it going|how are things|how you doing", low)
        asked, self.asked = self.asked, None
        invite = self.vary(R.INVITES[r])
        if act == "thanks" and not THANKS_WORD.search(low):  # "I'm very excited" is not a thank-you
            act = "smalltalk"

        # the caller's name, whenever they give it
        name = name_in(text, asked == "name")
        if name:
            self.name = name
            if not (act == "request" or reading.has_request):
                self.greeted = self.greeted or reading.has_greeting
                return self._reply([R.NICE_TO_MEET.format(name=name), invite], plan, "name")

        # questions any front desk answers, whatever act the reflex read: the clock, the caller's name, the line,
        # what Sona can do (decided by pattern; the facts come from the clock and the call, never generated)
        if act not in ("stop", "goodbye", "continue"):
            if DATE_Q.search(low):
                return self._reply([R.LOOKUP[r], R.spoken_date(self.now())], plan, "fact:date")
            if TIME_Q.search(low):
                return self._reply([R.LOOKUP[r], R.spoken_time(self.now())], plan, "fact:time")
            if NAME_Q.search(low):
                if self.name:
                    return self._reply([R.NAME_KNOWN.format(name=self.name), invite], plan, "fact:name")
                self.asked = "name"
                return self._reply([self.vary(R.NAME_ASK)], plan, "ask:name")
            if HEAR_Q.search(low):
                return self._reply([R.HEAR_YOU, invite], plan, "hear")
            if TALK_Q.search(low):
                return self._reply([R.ACK[r], self.vary(R.CAN_DO)], plan, "can_do")

        if act == "request" or reading.has_request:
            first = [R.AFTER_CUT[r] if interrupted else R.ACK[r]]
            if reading.sentiment["neg"] > 0.5:  # a complaint: empathy before anything else
                first = [R.EMPATHY[r]]
            if reading.has_greeting and not self.greeted:
                first = [R.GREET_BACK[r]] + (first if reading.sentiment["neg"] > 0.5 else [])
            self.greeted = self.greeted or reading.has_greeting
            self.cut = None  # a new question replaces whatever was cut off
            if reading.intent_p < 0.45 and reading.intent != "out_of_scope":
                self.asked = ("clarify", reading.intent)
                return self._reply([*first, R.CLARIFY.format(topic=R.TOPICS[reading.intent])], plan, "clarify")
            body = split(R.ANSWERS[reading.intent])
            if body[-1].endswith("?") and "email" in body[-1]:
                self.asked = "email"
            return self._reply([*first, *body], plan, f"answer:{reading.intent}")
        if act == "greeting":
            if self.greeted:  # hello is said once a call; a second greeting gets "hi again" or the how-are-you
                first = R.HOW_ARE_YOU[r] if how_are_you else R.GREET_AGAIN[r]
                return self._reply([first, invite], plan, "greet:again")
            self.greeted = True
            return self._reply([R.GREET_BACK[r], R.HOW_ARE_YOU[r] if how_are_you else "", invite], plan, "greet")
        if act == "smalltalk":
            if re.search(r"\b(robot|a\.?i\.?|bot|computer|real person|human|machine)\b", low) and is_question(text):
                return self._reply(["Hmm.", R.BOT, invite], plan, "smalltalk:bot")
            if how_are_you:
                return self._reply([R.HOW_ARE_YOU[r], invite], plan, "smalltalk:how")
            if is_question(text):  # a question we have no answer for: say so, and say what we can do
                return self._reply([self.vary(R.CANT_ANSWER), self.vary(R.CAN_DO)], plan, "smalltalk:cant")
            # a statement: meet its feeling
            s = reading.sentiment
            first = (R.GLAD[r] if s["pos"] > 0.5 else R.EMPATHY[r] if s["neg"] > 0.5 else R.NOTED[r])
            return self._reply([first, invite], plan, "smalltalk")
        if act == "repeat" and self.last:
            again = self.last.sentences[1:] if len(self.last.sentences) > 1 else self.last.sentences
            plan.delivery = replace(plan.delivery, speed=round(plan.delivery.speed - 0.08, 3))
            return self._reply([R.RESUME, *again], plan, "repeat")
        if act == "continue":
            if self.cut:
                reply, n = self.cut
                self.cut = None
                rest = reply.sentences[max(n, 1):]
                return self._reply([R.RESUME, *rest], plan, "resume")
            return self._reply([R.GO_AHEAD[r]], plan, "go_ahead")
        if act == "stop":
            return self._reply([R.GO_AHEAD[r]], plan, "yield")
        if act == "thanks":
            return self._reply([R.WELCOME[r], R.ANYTHING_ELSE], plan, "welcome")
        if act == "goodbye":
            return self._reply([R.GOODBYE], plan, "goodbye", end_call=True)
        if act in ("affirm", "backchannel") and asked:
            if isinstance(asked, tuple):
                return self._reply([R.ACK[r], *split(R.ANSWERS[asked[1]])], plan, f"answer:{asked[1]}")
            return self._reply([R.AFFIRMED], plan, "affirmed")
        if act == "deny" and isinstance(asked, tuple):
            return self._reply(["Hmm.", R.NOT_SURE], plan, "clarify:more")
        if act == "deny":
            return self._reply([R.DENIED], plan, "denied")
        if act in ("affirm", "backchannel"):  # "okay", "bro": nothing was asked, so leave them room
            if self.last and self.last.kind.startswith("answer"):
                return self._reply([R.GREAT, R.ANYTHING_ELSE], plan, "ack")
            return self._reply([self.vary(R.LISTENING), invite], plan, "ack")
        return self._reply(["Hmm.", R.NOT_SURE], plan, "unsure")
