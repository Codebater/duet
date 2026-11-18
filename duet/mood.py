"""Mood: what the caller's tone and words add up to, and how DUET should sound in return.

The caller's mood is two numbers: arousal (calm -1 .. +1 excited), mostly from the tone of voice (`prosody`), and
valence (negative -1 .. +1 positive), mostly from the words (`understand`) with a small hint from the tone. A slow
average over the call keeps one odd turn from swinging it.

The reply's mood follows one rule: **meet the caller where they are, then lift**.
- positive or neutral caller: match their energy (same arousal), a little warmer (valence + lift); the bright
  register only when the words are positive too;
- frustrated (negative, aroused): don't mirror anger; calm and steady, slower, warm and brief;
- low (negative, quiet): gentle and warm, a touch slower, a small lift in energy;
and the lift grows a little each turn while the caller follows it ("enhance the mood").

The reply's mood becomes a `Delivery` (pace, style lift along the measured voice directions, level) plus a
register the replies are worded in: "bright", "warm", "calm", "steady".
"""
from dataclasses import dataclass

from .voice import Delivery


@dataclass
class Mood:
    arousal: float = 0.0
    valence: float = 0.0

    @property
    def label(self):
        a, v = self.arousal, self.valence
        if v < -0.25:
            return "frustrated" if a > 0.15 else "low"
        if v > 0.25:
            return "excited" if a > 0.25 else "content"
        return "energetic" if a > 0.35 else ("calm" if a < -0.35 else "neutral")

    def __str__(self):
        return f"{self.label} (arousal {self.arousal:+.2f}, valence {self.valence:+.2f})"


def fuse(tone, reading, previous=None, weight=0.6, recover=0.5):
    """This turn's mood from the tone (`ToneReading`) and the words (`Reading`), smoothed with the call so far.
    `recover`: how much slower the mood rises than it falls (the front desk: a complaint isn't undone by one
    "thanks"; the friend: 1.0, it meets joy at once)."""
    words = reading.valence if reading else 0.0
    valence = 0.8 * words + 0.2 * (tone.valence_hint if tone else 0.0)
    arousal = tone.arousal if tone else 0.0
    if reading and reading.sentiment["neg"] > 0.6:
        arousal = max(arousal, 0.2)  # complaints are rarely calm, even when said quietly
    now = Mood(arousal, valence)
    if previous is None:
        return now
    # moods sour fast and recover slowly: one polite "thanks" doesn't undo a complaint
    wv = weight if now.valence < previous.valence else weight * recover
    return Mood(previous.arousal + weight * (now.arousal - previous.arousal),
                previous.valence + wv * (now.valence - previous.valence))


@dataclass
class Plan:
    target: Mood
    register: str  # bright | warm | calm | steady
    delivery: Delivery

    @property
    def opener_delivery(self):
        """The register's fixed delivery, so openers can be rendered once and cached."""
        return REGISTER_DELIVERY[self.register]


def _delivery(target, register):
    # pitch lift: measured on Orion (models/mood-voice/REPORT.md) as a threshold, ~0 below 0.3, +1 st at 0.6, +4 st
    # at 0.7, so it is capped at 0.65; the range direction failed its check and is zero, so pace and level carry
    # arousal.
    lift = min(max(0.75 * target.valence + 0.3 * target.arousal, -0.3), 0.65)
    return Delivery(speed=round(1.03 + 0.12 * target.arousal, 3), range_lift=round(0.8 * target.arousal, 3),
                    pitch_lift=round(lift, 3), gain_db=round(1.5 * target.arousal, 2), label=register)


REGISTER_DELIVERY = {"bright": _delivery(Mood(0.45, 0.6), "bright"), "warm": _delivery(Mood(0.1, 0.4), "warm"),
                     "calm": _delivery(Mood(-0.2, 0.25), "calm"), "steady": _delivery(Mood(-0.3, 0.2), "steady")}


# ------------------------------------------------------------------ the friend: match, don't lift

# A friend answers in the caller's own energy and feeling (accommodation): seven registers on the arousal x valence
# plane, each a fixed delivery so its openers and backchannels can be rendered once. Joy is met in full
# (active-constructive responding); distress is met in seriousness, never in anger, and never cheered up.
FRIEND_REGISTERS = {
    "excited": Mood(0.55, 0.65), "warm": Mood(0.15, 0.4), "lively": Mood(0.4, 0.15), "relaxed": Mood(0.0, 0.15),
    "calm": Mood(-0.25, 0.1), "serious": Mood(-0.15, -0.1), "gentle": Mood(-0.4, -0.05),
}
FRIEND_DELIVERY = {name: _delivery(m, name) for name, m in FRIEND_REGISTERS.items()}
TONE_WORDS = {
    "excited": "excited and upbeat: match their high energy", "warm": "warm and happy for them",
    "lively": "lively and upbeat: you're happy to be talking with them", "relaxed": "relaxed and easygoing", "calm": "calm and low-key",
    "serious": "serious, steady and calm, on their side; never argue or cheer", "gentle": "soft, gentle and caring, a bit slower",
}


def voice(register):
    """The plan for a named friend register (its target mood need not map back to it through `match`)."""
    return Plan(FRIEND_REGISTERS[register], register, FRIEND_DELIVERY[register])


def match(caller):
    """The friend's register for a caller in `caller` mood: the same energy and feeling, within reason."""
    a, v = caller.arousal, caller.valence
    if v < -0.25:  # something is wrong: sad and low -> gentle; upset or angry -> serious (their side, calm)
        register = "gentle" if a < 0.15 else "serious"
    elif v > 0.2:  # anything good: excited with them, warm if they're very quiet about it
        register = "excited" if a > -0.15 else "warm"
    else:  # glad to be talking: lively, unless they sound really low-key
        register = "lively" if a > -0.2 else ("calm" if a < -0.45 else "warm")
    return Plan(FRIEND_REGISTERS[register], register, FRIEND_DELIVERY[register])


def respond(caller, lift_turns=0, recovering=False):
    """The reply's mood and delivery for a caller in `caller` mood, `lift_turns` turns into a good stretch.
    `recovering`: the caller was frustrated or low in the last two turns, so stay warm, not bright."""
    lift = min(0.15 + 0.05 * lift_turns, 0.35)
    a, v = caller.arousal, caller.valence
    if v < -0.25 and a > 0.15:  # frustrated: steady, not mirrored
        target, register = Mood(-0.3, 0.2), "steady"
    elif v < -0.25:  # low: gentle, small lift in energy
        target, register = Mood(min(a + 0.15, 0.0), 0.3), "calm"
    else:  # meet them and lift
        target = Mood(max(min(a, 0.1 if recovering else 0.7), -0.4), min(v + lift, 0.8))
        # bright needs the words to be positive too: tone alone can't tell excited from annoyed (the prosody head
        # reads 20 % of happy clips as angry), and a chirpy answer to an annoyed caller costs more than a warm one
        # to a happy caller. The first live call got "bright" for six turns of neutral words said loudly.
        energetic = target.arousal > 0.25
        register = ("bright" if energetic and v > 0.1 else
                    "warm" if energetic or target.valence > 0.2 else "calm")
    # One fixed delivery per register for the whole reply: every reply's first two sentences can then be
    # rendered ahead of time, and the target mood still decides the register.
    return Plan(target, register, REGISTER_DELIVERY[register])
