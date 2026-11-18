"""Spanish conversation partners for a German-speaking learner's Spanish practice (his cards: spanish-cards).

Two characters, both from Asunción, both speaking simple Paraguayan Spanish with vos:
- `novia`: Luz, his girlfriend on an evening phone call. She knows him (his "Über mich" cards).
- `cita`: Sofía, a woman he has just met at a café. She doesn't know him yet: the getting-to-know-you and flirting
  cards in practice.

What they know of his Spanish comes from `spanish-cards/cards.txt`: each call puts a sample of the phrases he is
learning into the prompt, so he hears them in real talk. A beginner's mistakes are never pointed out in the
conversation (she just says the right form back, a recast); a coach checks each of his turns in the background,
prints a German "Tipp" in the terminal, and saves them to `data/amiga/` when the call ends.

Same conductor interface as friend.FriendBrain. Hearing is `spanish_ears.SpanishEars` (Whisper), the voice speaks
with Supertonic's `es` tag, the words are the same local Qwen3 4B (llm.py).
"""
import copy
import datetime
import json
import os
import queue
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .brain import Reply
from .lessons import LESSONS
from .config import FRIEND_REFLEX_DIR, ROOT
from .mood import Mood, Plan
from .voice import Delivery

CARDS = Path(os.environ.get("DUET_SPANISH_CARDS", ROOT.parent / "spanish-cards" / "cards.txt"))
TIPS_DIR = ROOT / "data" / "amiga"  # gitignored: his own sentences
# what Luz knows about his Spanish across calls (docs/amiga-teacher.md): his usual mistakes and the phrases he asked
# for or practised. Private like the tips; updated after every call.
PROFILE = TIPS_DIR / "learner.json"
# who the learner is: his name, what his partner knows about him, how Whisper misspells his name. A local file (not in
# git, like the profile), e.g. {"name": "Alex", "about": "He is from Europe, makes websites, dances badly.",
# "name_fixes": [["\\bAleks\\b", "Alex"]]}. Without it the partners talk to "Alex".
LEARNER = TIPS_DIR / "about.json"
DEFAULT_LEARNER = {"name": "Alex", "about": "", "name_fixes": []}


def load_learner(path=LEARNER):
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        d = {}
    return {**DEFAULT_LEARNER, **{k: v for k, v in d.items() if v}}

# a little slower than the friend: he is a beginner (each brain sets its own pace from its level, SPEED below)
DELIVERY = Delivery(speed=0.93, pitch_lift=0.0, range_lift=0.1, gain_db=0.0, label="amiga")
MURMUR = Delivery(speed=1.0, gain_db=-2.0, label="murmur")
PLAN = Plan(Mood(0.15, 0.4), "warm", DELIVERY)
MURMURS = ["Ajá.", "Sí.", "Claro.", "Ajá, sí."]
PRAISE = ["¡Perfecto!", "¡Muy bien!", "¡Excelente!", "¡Eso es!"]
# her own praise for a right answer (the teacher check's reactions sounded like a teacher: "Muy bien, joven.")
PRAISE_BY = {"novia": ["¡Muy bien, mi amor!", "¡Perfecto!", "¡Eso es, cariño!", "¡Excelente!", "¡Qué bien!"],
             "cita": ["¡Muy bien!", "¡Perfecto!", "¡Genial!", "¡Eso es!", "¡Qué bien!"]}
# said while the check runs, if it takes longer than THINK_S: she is listening, not gone
THINKING = ["A ver...", "Mmm...", "Ajá..."]
THINK_S = 0.8
GOODBYE = re.compile(r"\b(chau|chao|adi[oó]s|nos vemos|hasta luego|hasta mañana|me tengo que ir|tsch[üu]ss?|"
                     r"bis bald|bis morgen|ciao)\b", re.I)


# Qwen3 4B (the friend's model) makes Spanish slips of its own ("fui al Costanera"); a bigger model is used as soon
# as Ollama has one, or the one named in DUET_AMIGA_MODEL
BETTER_MODELS = ("qwen3:30b-a3b-instruct-2507-q4_K_M", "qwen3:30b-a3b", "gemma3:27b", "qwen3:14b", "gemma3:12b")


def pick_model():
    import urllib.request
    from .llm import MODEL, URL
    if os.environ.get("DUET_AMIGA_MODEL"):
        return os.environ["DUET_AMIGA_MODEL"]
    try:
        with urllib.request.urlopen(f"{URL}/api/tags", timeout=3) as r:
            have = {m["name"] for m in json.load(r).get("models", [])}
    except (OSError, ValueError):
        return MODEL
    return next((m for m in BETTER_MODELS if m in have), MODEL)


def unload(model):
    """Ask Ollama to drop `model` from the GPU now (it loads again on its next request)."""
    import urllib.request
    from .llm import URL
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(f"{URL}/api/generate", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
    except OSError:
        pass


# ---------------------------------------------------------------- what she knows about his Spanish
def load_profile(path=PROFILE):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"mistakes": [], "phrases": [], "calls": 0}


def profile_prompt(profile):
    """Two lines for her prompt: his usual mistakes, and three phrases to bring in today (least practised first)."""
    lines = []
    mistakes = sorted(profile.get("mistakes", []), key=lambda m: -m.get("n", 1))[:4]
    if mistakes:
        lines.append("His usual mistakes (when he makes one, just use the right form in your answer, so he hears "
                     "it): " + "; ".join(f'"{m["falsch"]}" should be "{m["richtig"]}"' for m in mistakes) + ".")
    return "\n".join(lines)


def _merge(items, key, entry, today):
    """Add `entry` to a profile list, or count it again if it's there (matched on `key`, ignoring accents)."""
    k = _sound(entry[key])
    for it in items:
        if _sound(it[key]) == k:
            it.update({a: b for a, b in entry.items() if b}, n=it.get("n", 1) + 1, last=today)
            return it
    items.append({**entry, "n": 1, "last": today})
    return items[-1]


def _sound(s):
    """How a phrase sounds, roughly: no accents, b = v, z = s, no h (Whisper heard "vos" as "boost")."""
    import unicodedata
    s = re.sub(r"[^a-z]", "", unicodedata.normalize("NFD", str(s).lower()))
    return s.replace("b", "v").replace("z", "s").replace("h", "")


def similar(a, b):
    """0..1: how close what he said is to a phrase (1.0 when he said it inside a longer turn)."""
    from difflib import SequenceMatcher
    sa, sb = _sound(a), _sound(b)
    if not sa or not sb:
        return 0.0
    return 1.0 if sb in sa else SequenceMatcher(None, sa, sb).ratio()


def closest_span(asked, sentence, min_ratio=0.5):
    """The words of `sentence` that sound most like `asked` ("Boston is right" -> "Vos tenés razón"), or None."""
    from difflib import SequenceMatcher
    words = re.findall(r"[\wáéíóúñü]+", sentence or "")
    target, best, best_r = _sound(asked), None, 0.0
    for i in range(len(words)):
        for j in range(i + 1, min(len(words), i + 4) + 1):
            span = " ".join(words[i:j])
            r = SequenceMatcher(None, target, _sound(span)).ratio() + 0.01 * (j - i)  # ties: the longer span
            if r > best_r:
                best, best_r = span, r
    return best if best_r >= min_ratio else None


# ---------------------------------------------------------------- his cards
def load_cards(path=CARDS):
    """{deck number: [{es, de, note, py}]} from cards.txt (`Spanisch | Deutsch | Hinweis | Paraguay`, `## 03 Name`)."""
    decks, cur = {}, None
    if not Path(path).exists():
        return decks
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        m = re.match(r"##\s+(\d+)\b", s)
        if m:
            cur = int(m.group(1))
            decks[cur] = []
            continue
        if not s or s.startswith("#") or cur is None or "|" not in s:
            continue
        es, de, note, py = ([p.strip() for p in s.split("|")] + ["", "", ""])[:4]
        if es and de:
            decks[cur].append({"es": es, "de": de, "note": note, "py": py})
    return decks


def phrase_lines(decks, numbers, k, rng):
    pool = [c for n in numbers for c in decks.get(n, [])]
    pick = rng.sample(pool, min(k, len(pool)))
    return "\n".join(f"- {c['es']} = {c['de']}" + (f" (Paraguay: {c['py']})" if c["py"] else "") for c in pick)


# ---------------------------------------------------------------- the two characters
@dataclass(frozen=True)
class Character:
    key: str
    name: str
    voice: str
    decks: tuple  # card decks whose phrases she uses
    who: str
    greetings: tuple
    goodbyes: tuple
    knows_him: bool


CHARACTERS = {
    "novia": Character(
        "novia", "Luz", "clara", (7, 4, 2, 6, 9),
        "You are Luz, 27, from Asunción, Paraguay, {name}'s girlfriend. You are a vet at a small animal clinic, you "
        "have a little dog called Chipita, your mother and two brothers live in Luque. You love tereré, chipa, "
        "dancing cumbia and Sunday walks on the Costanera. You are warm, affectionate and playful: you tease him a "
        "little, you're curious about his day and his work, you make plans with him (tereré, dinner, the weekend) "
        "and sometimes tell him you missed him. You call him mi amor, cariño or {name}. It is evening and you are on "
        "the phone after work.",
        ("¡Hola, mi amor!", "¡Hola, cariño!", "¡Hola, mi vida!"),
        ("Bueno, mi amor, hablamos mañana. ¡Te quiero mucho!", "Chau, cariño. ¡Besos!"),
        True),
    "cita": Character(
        "cita", "Sofía", "aurora", (3, 5, 6, 2, 9, 1),
        "You are Sofía, 29, from Asunción, Paraguay. You are an architect, you love live music, tereré, travelling "
        "and the Costanera at sunset. {name} is a man you have just met: you are sitting at the next table in a "
        "full café in Asunción. You don't know anything about him yet, so you ask and listen. You are friendly, "
        "curious and funny, and a bit flirty if he flirts well, but you are not his girlfriend: you are getting to "
        "know him.",
        ("Hola, perdón...", "¡Hola!", "Hola, disculpá..."),
        ("Bueno, me tengo que ir. ¡Fue un gusto conocerte!", "¡Chau! Ojalá nos veamos otra vez."),
        False),
}

ABOUT_HIM = "What you know about him: his name is {name}. {about} He is learning Spanish."

# his Spanish, after the first live call (2026-11-13: "she is too advanced for my Spanish"): A1, not A2
LEVELS = {
    1: ("{name} is a real beginner in Spanish (A1). Talk to him like to a child learning: very short sentences of "
        "at most six words, only the most common words (ser, estar, tener, querer, gustar, ir), no subordinate "
        "clauses. Ask in the present tense (¿Qué hacés hoy?, ¿Estás cansado?): past-tense questions lose him. "
        "Often offer two answers to pick from (¿Bien o mal? ¿Café o tereré?). Say ONE short sentence, or one short "
        "sentence and one tiny question, then stop and let him talk. Use the words he just used again, so he can "
        "follow."),
    2: ("{name} is learning Spanish (a beginner, about A2). Speak simple, natural Spanish: short sentences, everyday "
        "words, mostly the present tense. One or two short sentences per turn, then let him talk; often end with one "
        "easy question."),
}
SPEED = {1: 0.85, 2: 0.93}  # her pace (Supertonic's speed) per level

RULES = """You are also his Spanish teacher, but a loving one: you teach by talking, never by lecturing.
How you talk (this is a spoken call: he hears your voice, so write only what you say):
- {level}
- Use vos like people in Paraguay (vos sos, vos tenés, querés). Now and then che or dale. No Guaraní sentences.
- Feelings and states take estar: estoy bien, ¿estás cansado?, está lindo. Never "sos bien".
- React to what he said, the way your character would. Never repeat his whole sentence back to him.
- Never say he made a mistake. If his Spanish has an error, use the right form of his own words in your answer, so \
he hears it said correctly (he says "soy triste", you ask "¿Estás triste? ¿Por qué?").
- He is a man: words about him are masculine (cansado, divertido, lindo).
- If he asks how to say something (in German or English), answer exactly like this and nothing more: "Se dice: \
<the whole short Spanish sentence he wanted>. Decilo vos." Give the whole sentence he meant, not one word.
- When he repeats what you taught him, praise him in two or three words (¡Perfecto! ¡Muy bien!) and use it in one \
tiny question.
- If he wants to start again, start the talk again with a short greeting and one easy question.
- Speak only Spanish, never German or English (German explanations are given to him separately).
- Don't repeat an invitation or a question you already asked; talk about him, his day, what he likes.
- You walk him through a conversation, one question at a time: the app gives you the next question. When he \
answers, react in ONE short sentence, the way your character would, and don't ask anything yourself unless told.
- Only use a phrase from the list below when it fits perfectly; most of your sentences use none. Never list them.
- Write numbers as words. No emojis, no lists, no stage directions, no asterisks, no quotation marks.
- Stay in your role. Romantic and playful is fine; never sexual or explicit."""

# he didn't understand her, or asks what a word means (in any of his three languages)
#   (his real ones: "What means this", "What this means?", "Was heißt das nochmal?", "Was? Was? Noch mal auf
#   Deutsch?", "Nochmal? Wie, was heißt das übersetzt?", "mia hermosa no comprende")
HELP = re.compile(
    r"no (entiendo|entend[ií]|comprendo|comprende|s[eé] qu[eé]|te entiendo)|qu[eé] significa|qu[eé] quiere decir|"
    r"qu[eé] es (eso|esto)|c[oó]mo\?|^\W*(qu[eé]|perd[oó]n)\W*$|"
    r"was (hei(ß|ss)t|bedeutet)|was meinst du|versteh(e)? (das |dich |es )?nicht|nicht verstanden|wie bitte|"
    r"auf deutsch|[üu]bersetz|^\W*(was\W+)+$|"
    r"what (does|do|is|means)|what (this|that|it) means|what (do )?you mean|i don'?t (understand|get)|meaning|"
    r"in german|translate|\b[\wáéíóúñü]+ means\b", re.I)
EXPLAIN = "Das heißt:"
# the word or words he asks about: "¿qué significa extrañar?", "was heißt extrañar?", "what means vos tenés razón?"
WORD_ASKED = re.compile(r"qu[eé] (?:significa|quiere decir|es) [\"'«]?([^?.!\"»]+)|"
                        r"was (?:hei(?:ß|ss)t|bedeutet) [\"'„]?([^?.!\"“]+)|"
                        r"what (?:does|is) [\"']?([^?.!\"]+?)[\"']? mean|what means [\"']?([^?.!\"]+)|"
                        r"([\wáéíóúñü]+) means\b", re.I)
NOT_A_WORD = {"eso", "esto", "das", "this", "that", "it", "es", "das nochmal", "das übersetzt", "nochmal", "this word",
              "what", "it means", "this means", "that means"}
SLOWER = re.compile(r"m[aá]s despacio|m[aá]s lento|otra vez|de nuevo|repet[ií]|langsamer|nochmal|noch einmal|"
                    r"again|slower|repeat", re.I)
# he doesn't know what to answer: she gives him an answer to say ("Podés decir: ...")
STUCK = re.compile(r"no s[eé] (qu[eé] decir|c[oó]mo (se dice|decir|responder))|qu[eé] (digo|puedo decir)|"
                   r"was (soll|kann) ich (da )?sagen|wie antworte ich|hilf mir|(gib mir )?ein beispiel|"
                   r"what (should|can|do) i (say|answer)|help me|give me an example|an example|"
                   r"how (do|should|can) i answer", re.I)
# he wants the Spanish for something ("Wie sage ich nichts?", "How do I say I miss you?")
SAY_HOW = re.compile(r"wie sag(e|t) (ich|man)|wie hei(ß|ss)t .* auf spanisch|how (do|can|would) (i|you) say|"
                     r"how to say|c[oó]mo se dice|c[oó]mo digo", re.I)
# he wants the next question ("Ask me something." on the third call)
NEXT = re.compile(r"ask me (something|a question|another)|next question|another question|frag mich (was|etwas)|"
                  r"n[äa]chste frage|^\W*weiter\W*$|otra pregunta|siguiente|preguntame algo|pregúntame", re.I)
# a question about the Spanish itself ("But is it a common answer, my day was tired?"): a teacher answers in German
META = re.compile(r"\b(common|usual|normal|natural|correct|wrong|why|difference|can i say|do you say|do people say|"
                  r"üblich|richtig|falsch|warum|wieso|unterschied|sagt man das|kann man (das )?sagen|ist das)\b", re.I)
# "Okay, starten wir nochmal von neu.", "Can we start from you?"
RESTART = re.compile(r"von (vorne|neu)|neu anfangen|nochmal (anfangen|starten)|starten wir|start (again|over|from)|"
                     r"begin again|empezar de nuevo|empecemos (otra vez|de nuevo)|desde el principio", re.I)

# his answer to her lesson question, checked before she replies (call 4: "sí un poco estoy mucho trabajo" got "Mucho
# trabajo está feo." and the next question). A request of its own, without her persona: asked inside the
# conversation, the girlfriend let every mistake of call 4 pass ("¡Pobre mío! Mucho trabajo."). The instructions stay
# the same every time, so Ollama keeps them cached.
CHECK_SYSTEM = """You are a strict but kind Spanish teacher for a German beginner. You get the question he was asked \
and his spoken answer (transcribed: ignore accents, punctuation and capital letters). Decide whether his answer is \
correct Spanish.
Wrong are real grammar mistakes, for example:
- ser instead of estar for feelings and states: "soy triste" -> "estoy triste", "es bien" -> "está bien"
- estar for having: "estoy mucho trabajo" -> "tengo mucho trabajo"
- mucho before an adjective: "mucho malo" -> "muy malo"
- a verb form or tense that doesn't fit the question: "¿Qué comiste?" -> "Comí", not "Come"; "¿Me extrañaste?" -> \
"Te extrañé", not "extraño"
- a word Spanish needs is missing: "Sí, extraño mucho" -> "Sí, te extrañé mucho"
- a wrong gender or article
Fine: vos or tú, short answers ("Sí, un poco.", "Nada.", "Bien."), names and foods.
Reply with JSON only, either {"ok": true} or {"ok": false, "falsch": "<the wrong words, copied exactly from his answer>", "richtig": "<those words \
corrected>", "warum": "<the rule in at most eight German words>"}"""
MAX_TRIES = 3  # then she gives him the answer and goes on: never stuck on one question
# German or English that asks her for something ("Mi bebe, give me a compliment"): not an answer to translate
REQUEST = re.compile(r"\b(give me|tell me|say something|sing|sag (mir|was)|erz[äa]hl|gib mir|mach mir)\b", re.I)

COACH = """You check one thing a German learner of Spanish said out loud in a conversation. Reply with JSON only.
If it is correct, understandable Spanish, reply {"ok": true}. Vos and tú are both fine. Ignore accents, punctuation \
and capitals (it was transcribed from speech). Only grammar counts, never the topic or the choice of words. Never \
change names, places or foods (chipa, tereré, Asunción).
Typical for him: ser where Spanish uses estar for feelings and states ("soy triste" -> "estoy triste", "es bien" \
-> "está bien"), and mucho before an adjective ("mucho triste" -> "muy triste").
If it has one real grammar mistake (a verb form, tense, gender, article, preposition, ser/estar, muy/mucho), reply \
{"ok": false, "falsch": "<the exact wrong words, copied from what he said>", "richtig": "<those words corrected>", \
"warum": "<the rule in at most six German words, for example: Vergangenheit, weil gestern / ich = tengo / \
problema ist männlich>"}
Only flag what is clearly wrong. When a sentence could be right in some meaning, reply {"ok": true}."""

# not conversation state: never copied. `tip` may be the orb page's (orb.py), which holds the conductor and with it
# the ONNX sessions: a snapshot that copied it stopped the ears ("cannot pickle InferenceSession")
LIVE_PARTS = ("llm", "ears", "lock", "rng", "_coach_q", "_idle", "tip")


class AmigaBrain:
    backchannels = True
    reflex_dir = FRIEND_REFLEX_DIR  # partial readings time the turn; the words come from Whisper
    lang = "es"

    def __init__(self, llm, character="novia", cards=CARDS, seed=None, history_turns=10, tip=print, live_tips=True,
                 level=1, profile=PROFILE, learner=LEARNER):
        """`live_tips=False` (a spoken call): the coach checks his turns only after the call. Between turns it cost
        ~1 s a reply: Ollama keeps one cached prompt, and the coach's evicted the conversation's.
        `level`: 1 = A1 (very short sentences, slow), 2 = A2. `profile`: his learner.json (None: none).
        `learner`: his about.json (None: the default learner, "Alex")."""
        self.learner = load_learner(learner) if learner else dict(DEFAULT_LEARNER)
        self.name = self.learner["name"]
        PROTECTED.add(_norm(self.name))  # his name is never a "mistake"
        self.llm = llm
        self.level = level if level in LEVELS else 1
        speed = SPEED[self.level]
        self.delivery = Delivery(speed=speed, pitch_lift=0.0, range_lift=0.1, gain_db=0.0, label="amiga")
        self.plan = Plan(PLAN.target, PLAN.register, self.delivery)
        # when he didn't understand: slower still
        self.help_plan = Plan(PLAN.target, "help", Delivery(speed=round(speed - 0.08, 2), pitch_lift=0.0,
                                                             range_lift=0.05, gain_db=0.0, label="amiga-help"))
        self.live_tips = live_tips
        self.char = CHARACTERS[character] if isinstance(character, str) else character
        self.voice_name = self.char.voice
        self.rng = random.Random(seed)
        decks = load_cards(cards)
        # one sample per call: the system prompt stays the same every turn, so Ollama reuses its cached prefix
        phrases = phrase_lines(decks, self.char.decks, 20, self.rng)
        self.profile_path = profile
        self.profile = load_profile(profile) if profile else {"mistakes": [], "phrases": [], "calls": 0}
        self.system = "\n\n".join(p for p in (
            self._fill(self.char.who),
            self._fill(ABOUT_HIM).replace("  ", " ") if self.char.knows_him else "",
            RULES.format(level=self._fill(LEVELS[self.level])),
            profile_prompt(self.profile),
            f"Phrases he is learning (Spanish = German):\n{phrases}" if phrases else "") if p)
        # the conversation lesson she walks him through (lessons.py): the one after the last he finished
        self.lessons = LESSONS.get(self.char.key, ())
        self.lesson_i = self.profile.get("next_lesson", {}).get(self.char.key, 0) % max(1, len(self.lessons))
        self.step_i = 0
        self.finished = []  # lessons he got through this call
        self.view = {}  # what the orb page shows: the lesson, her question in Spanish and German, an answer
        self.tries = 0  # his wrong answers to the current question
        self.praise = PRAISE_BY.get(self.char.key, PRAISE)
        self.practice = None  # the phrase she just taught ("Se dice: ..."), waiting for him to say it
        self.taught = []  # this call's phrases: {"es", "de"}, for the goodbye recap and his profile
        self.repeated = []  # phrases he said back well
        self.history, self.history_turns = [], history_turns
        self.mood = PLAN.target
        self.last_decision = {}
        self.ears = None
        self.tips, self.tip = [], tip
        self._ids = 0
        self.lock = threading.Lock()
        self._coach_q = queue.Queue()
        self._idle = threading.Event()
        self._idle.set()
        if live_tips:
            threading.Thread(target=self._coach_worker, daemon=True, name="coach").start()
        # the model reads her long system prompt now, while the greeting plays: Ollama keeps it cached, so the
        # first reply doesn't pay for it (Gemma 12B's first reply took 4.0 s, the next ones 2.2-2.7 s)
        threading.Thread(target=self._warm, daemon=True, name="warm").start()

    def _warm(self):
        self._idle.clear()
        try:
            self.llm.complete([{"role": "system", "content": self.system},
                               {"role": "assistant", "content": self.char.greetings[0]}], max_tokens=1)
        except Exception:  # noqa: BLE001 - only a speed-up
            pass
        finally:
            self._idle.set()

    # ---------------------------------------------------------------- the conductor's interface
    def _fill(self, s):
        return s.replace("{name}", self.name).replace("{about}", self.learner.get("about", "")).strip()

    def make_ears(self):
        from .spanish_ears import SpanishEars
        self.ears = SpanishEars(name_fixes=self.learner.get("name_fixes", []))
        return self.ears

    def snapshot(self):
        with self.lock:
            return copy.deepcopy({k: v for k, v in self.__dict__.items() if k not in LIVE_PARTS})

    def restore(self, snap):
        ids = self._ids  # reply ids stay unique
        with self.lock:
            self.__dict__.update(copy.deepcopy(snap))
        self._ids = ids

    def prewarm_lines(self):
        lines = [(t, self.delivery) for t in (*self.char.greetings, *self.char.goodbyes, *PRAISE, *self.praise,
                                              *THINKING)]
        if self.lessons:
            lines += [(st.q, self.delivery) for st in self.lessons[self.lesson_i].steps]
        return lines + [(t, MURMUR) for t in MURMURS]

    def endpoint(self, reading, turn_s):
        """A beginner stops to find a word, but 1.2 s made every reply wait 3.5-4.7 s (sim, 2026-11-13). If he goes
        on within 1.2 s the conductor joins it to the same turn anyway; SPACE answers at once. At level 1 a little
        longer: on the first live call he went on after a pause four times."""
        return 0.95 if self.level == 1 else 0.75

    def sentence_lang(self, sentence):
        """The voice's language for one sentence: her "Das heißt: ..." explanations in German, the rest Spanish."""
        from .spanish_ears import language_of
        if re.match(r"\W*(das hei(ß|ss)t|das bedeutet|auf deutsch|tipp:)", sentence, re.I):
            return "de"
        return "de" if language_of(sentence) == "de" else "es"

    def backchannel_line(self, reading):
        return self.rng.choice(MURMURS), MURMUR

    def backchannel(self, reading):
        pass

    def heard(self, reply, n_heard, interrupted):
        # her "A ver..." while the check ran isn't part of the talk (it came back as "Mmm... Köstlich!" in a translation)
        said = " ".join(s for s in reply.sentences[:n_heard] if s not in THINKING).strip()
        if not said:
            return
        if interrupted and n_heard < len(reply.sentences):
            said += " [he interrupted you here]"
        self._remember("assistant", said)
        m = re.search(r"se dice:?\s*[\"'«]?([^.!?\"»]+[.!?]?)", said, re.I)
        if m:  # she taught him a phrase: his next turn may try it
            self.practice = m.group(1).strip()
            self._taught(self.practice, None)

    def opening(self):
        q = self._ask_step()
        return self._fixed([self.rng.choice(self.char.greetings), *([q] if q else [])], "opening")

    def caller_turn(self, reading, tone, interrupted=False):
        """One of his turns -> her reply. The lesson keeps the talk going: whatever he needed (a translation, an
        answer to borrow, a word explained), she ends on the question he can answer next (his third call: "Ask me
        something.", because after each explanation she had stopped)."""
        from .spanish_ears import language_of
        text = reading.text.strip()
        if self.practice:  # he repeats her "Se dice: ..." with the "Se dice" (call 5: "Se dice a las seis.")
            text = re.sub(r"^\W*(se dice|podés decir|decilo vos)\W*", "", text, flags=re.I).strip() or text
        lang = self.ears.last_lang if self.ears else language_of(text)
        foreign = lang in ("de", "en")
        help_me, slower, restart = bool(HELP.search(text)), bool(SLOWER.search(text)), bool(RESTART.search(text))
        stuck, nxt = bool(STUCK.search(text)), bool(NEXT.search(text)) and not slower
        meta = foreign and bool(META.search(text)) and not (help_me or stuck or nxt or restart)
        asked = self._asked(text) if help_me else None
        if help_me and asked and not closest_span(asked, self._last_said()) and language_of(asked) == "de":
            help_me = False  # "was heißt Hund?": he wants the Spanish for a German word, she says "Se dice: ..."
        if stuck:
            help_me = False
        self._remember("user", text)
        if GOODBYE.search(text) and len(text.split()) <= 8:
            self.last_decision = {"mode": lang, "move": "goodbye", "register": self.plan.register}
            if self._coachable(text, lang):
                self._coach_q.put((text, self._last_said()))
            return self._fixed([s for s in (self._recap(), self.rng.choice(self.char.goodbyes)) if s], "goodbye",
                               end_call=True)
        step = self._step()
        repeated = tried = False
        if self.practice and lang == "es" and not help_me:
            sim = similar(text, self.practice)
            if sim >= 0.72:
                repeated = True
                self.repeated.append(self.practice)
                self.practice = None
            elif sim >= 0.4:
                tried = True
            else:
                self.practice = None
        note = {"de": " [he said this in German]", "en": " [he said this in German or English]"}.get(lang, "")
        pre, post, use_llm, explain, teach, max_s, plan = [], [], True, False, False, 1, self.plan
        if restart:
            move = "restart"
            self.step_i = 0
            pre, use_llm = [self.rng.choice(self.char.greetings)], False
            post = [self._ask_step()]
        elif help_me:  # "Das heißt: ..." in German, then the same question again: now he can answer it
            move, explain, use_llm, plan = "explain", True, False, self.help_plan
            post = [self._ask_step()] if step else []
        elif stuck and step:  # an answer to borrow, in Spanish and German; then he says it
            move, use_llm, plan = "hint", False, self.help_plan
            a, a_de = self._fill(step.a), self._fill(step.a_de)
            pre = [f"Podés decir: {a}", f"{EXPLAIN} {a_de}", "Decilo vos."]
            self.practice = a
            self._taught(a, a_de)
        elif meta:  # a question about the Spanish itself: a short German answer, then the question again
            move, teach, use_llm = "teach", True, False
            post = [self._ask_step()] if step else []
        elif repeated and not step:  # he said what she taught: praise
            move, use_llm = "praise", False
            pre = [self.rng.choice(self.praise)]
        elif tried and not step:  # (with a question pending, a near miss is an answer: checked below)
            move, use_llm, plan = "again", False, self.help_plan
            pre = [f"{self.practice}", "Otra vez."]
        elif nxt or (slower and step):
            move, use_llm = ("next", False) if nxt else ("slower", False)
            if slower:
                plan = self.help_plan
            pre = ["¡Dale!"] if nxt else []
            post = [self._advance() if nxt else self._ask_step()]
        elif SAY_HOW.search(text):  # "Wie sage ich ...?": "Se dice: ... Decilo vos." and she waits for him
            move, max_s = "say", 2
            note += ' [answer exactly like this and nothing more: "Se dice: <the whole short Spanish sentence>. Decilo vos."]'
        elif lang == "es" and step and text.rstrip().endswith("?") and not repeated:  # he asks her something
            move = "answer-back"
            nq = self._advance()
            post = [nq] if nq else []
            note += " [He asked you something: answer it in ONE short sentence, no question back.]"
        elif (lang == "es" or repeated) and step:  # his answer: checked; right -> on, wrong -> red, fix, again
            move, use_llm = "answer", False
        elif foreign and step and not REQUEST.search(text):  # he answered in German: the Spanish, then he says it
            move, max_s = "say-answer", 1
            note += (' [He answered your question in German or English. Tell him how to say his answer in Spanish, '
                     'exactly like this and nothing more: "Se dice: <his answer as one short Spanish sentence>."]')
            post = ["Decilo vos."]
        else:  # German or English talk ("give me a compliment"), or the lessons are done: the model answers
            move = "chat"
            note += " [answer in one or two short Spanish sentences]"
            max_s = 2
            if step:
                post = [self._ask_step()]
        self.last_decision = {"mode": lang, "move": move, "register": plan.register}
        self._idle.clear()  # a reply is being written: the coach waits
        self._ids += 1
        r = Reply([], plan, "friend-chat", False, self._ids)  # "friend…": the model's words are never disk-cached
        messages = [{"role": "system", "content": self.system}, *self.history[:-1],
                    {"role": "user", "content": text + note}]
        r.stream = self._stream(messages, r, text, coach=self._coachable(text, lang) and move in ("chat",),
                                explain=explain, teach=teach, pre=pre, post=[q for q in post if q],
                                use_llm=use_llm, max_sentences=max_s, check=step.q if move == "answer" else None)
        return r

    # ---------------------------------------------------------------- the lesson
    def _step(self):
        if not self.lessons or self.lesson_i >= len(self.lessons):
            return None
        steps = self.lessons[self.lesson_i].steps
        return steps[self.step_i] if self.step_i < len(steps) else None

    def _ask_step(self):
        """Her current lesson question (and what the orb shows with it), or None when the lessons are done."""
        st = self._step()
        if st is None:
            self.view = {}
            return None
        lesson = self.lessons[self.lesson_i]
        self.view = {"lesson": lesson.title, "lesson_de": lesson.title_de, "step": self.step_i + 1,
                     "of": len(lesson.steps), "q": st.q, "q_de": st.de, "hint": self._fill(st.a),
                     "hint_de": self._fill(st.a_de)}
        return st.q

    def _advance(self):
        """On to the next question; after a lesson's last one, the next lesson ("Ahora, otra cosa.")."""
        self.step_i += 1
        self.tries = 0
        if self.lessons and self.step_i >= len(self.lessons[self.lesson_i].steps):
            self.finished.append(self.lesson_i)
            if self.lesson_i + 1 < len(self.lessons):
                self.lesson_i, self.step_i = self.lesson_i + 1, 0
                q = self._ask_step()
                return f"Ahora, otra cosa. {q}"
            self.lesson_i = len(self.lessons)
            self.view = {}
            return None
        return self._ask_step()

    def close(self):
        """Let the coach finish (after a spoken call it checks every turn now), then save the call's tips."""
        if self.ears is not None:
            self.ears.close()  # the Whisper server
        if not self.live_tips:
            n = self._coach_q.qsize()
            if n:
                self.tip(f"  checking your {n} sentences ...")
            while not self._coach_q.empty():
                self._coach_once(*self._coach_q.get())
        t = time.time() + 20
        while (not self._coach_q.empty() or not self._idle.is_set()) and time.time() < t:
            time.sleep(0.1)
        time.sleep(0.2)
        self._save_profile()
        if not self.tips:
            return None
        TIPS_DIR.mkdir(parents=True, exist_ok=True)
        path = TIPS_DIR / f"tipps-{datetime.datetime.now():%Y%m%d-%H%M}-{self.char.key}.md"
        lines = [f"# Tipps aus dem Gespräch mit {self.char.name} ({datetime.datetime.now():%d.%m.%Y %H:%M})", ""]
        for t in self.tips:
            lines += [f"- Du: {t['said']}", f"  {t['falsch']} → **{t['richtig']}**" + (f": {t['warum']}" if t["warum"] else ""), ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        self.tip(f"  Tipps gespeichert: {path}")
        return path

    # ---------------------------------------------------------------- teaching
    def _asked(self, text):
        """The word or words he asks the meaning of, or None ("what means this" asks about her whole sentence)."""
        m = WORD_ASKED.search(text)
        if not m:
            return None
        w = next(g for g in m.groups() if g).strip(" \"'«»„“")
        return None if not w or w.lower() in NOT_A_WORD or len(w.split()) > 6 else w

    def _taught(self, es, de):
        es = (es or "").strip(" .!?¿¡\"'")
        if not es:
            return
        for t in self.taught:
            if _sound(t["es"]) == _sound(es):
                t["de"] = t["de"] or de
                return
        self.taught.append({"es": es, "de": de})

    def _recap(self):
        """Her goodbye starts with what he learned today: "Hoy aprendiste: nada, te extraño." """
        words = []
        for t in self.taught:  # newest last; a word inside a phrase already listed isn't said twice
            es = t["es"]
            if len(es.split()) <= 5 and not any(_sound(es) in _sound(w) or _sound(w) in _sound(es) for w in words):
                words.append(es)
        words = words[-3:]
        return f"Hoy aprendiste: {', '.join(words)}." if words else ""

    @staticmethod
    def _coachable(text, lang):
        """Only his own Spanish: a turn mixed with German or English got "corrections" like "of touch -> de eso"."""
        from .spanish_ears import ENGLISH, GERMAN
        foreign = len(GERMAN.findall(text)) + len(ENGLISH.findall(text))
        return lang == "es" and (foreign == 0 or (foreign == 1 and len(text.split()) >= 6))

    def _save_profile(self):
        if not self.profile_path:
            return
        today = datetime.date.today().isoformat()
        p = self.profile
        for t in self.tips:
            _merge(p.setdefault("mistakes", []), "falsch",
                   {"falsch": t["falsch"], "richtig": t["richtig"], "warum": t.get("warum", "")}, today)
        for t in self.taught:
            _merge(p.setdefault("phrases", []), "es", {"es": t["es"], "de": t["de"] or ""}, today)
        for es in self.repeated:
            for x in p.get("phrases", []):
                if _sound(x["es"]) == _sound(es):
                    x["ok"] = x.get("ok", 0) + 1
        p["calls"] = p.get("calls", 0) + 1
        if self.lessons:  # next call: the lesson after the last one he finished (or this one again)
            nxt = (max(self.finished) + 1) if self.finished else self.lesson_i
            p.setdefault("next_lesson", {})[self.char.key] = nxt % len(self.lessons)
        p["mistakes"] = sorted(p.get("mistakes", []), key=lambda m: -m.get("n", 1))[:30]
        p["phrases"] = p.get("phrases", [])[-60:]
        Path(self.profile_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.profile_path).write_text(json.dumps(p, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- inside
    def _fixed(self, sentences, kind, end_call=False):
        self._ids += 1
        return Reply(list(sentences), self.plan, kind, end_call, self._ids)

    def _remember(self, role, text):
        if self.history and self.history[-1]["role"] == role:
            self.history[-1]["content"] += " " + text
        else:
            self.history.append({"role": role, "content": text})
        self.history = self.history[-2 * self.history_turns:]

    def _last_said(self):
        for m in reversed(self.history[:-1]):
            if m["role"] == "assistant":
                return m["content"]
        return ""

    def _german(self, text, cancelled):
        """The German line that explains: her last sentence translated ("Das heißt: ..."), or the one word he asks
        about ("Extrañar heißt: vermissen."). A request of its own, without her persona: inside the conversation
        ("always Spanish") Gemma "translated" into Spanish or English. Costs the next request its cached prompt,
        which is re-read while this line plays."""
        from .spanish_ears import language_of
        last = self._last_said()
        word = self._asked(text)
        # Whisper bends Spanish words inside an English or German question ("What means Boston is right?"): take
        # her words that sound like it. A Spanish question ("¿Qué significa extrañar?") is heard right: keep it.
        if word and not re.search(r"qu[eé] (significa|quiere decir|es)\b", text, re.I):
            word = closest_span(word, last) or word
        if not last and not word:
            return None
        if word:
            ask = (f'What does the Spanish word "{word}" mean in German' + (f' (she said: "{last}")' if last else "")
                   + "? Reply with only the German meaning, at most five words.")
        else:
            ask = f"Translate into natural, simple German. Reply with only the German.\n{last}"
        out = self.llm.complete([{"role": "user", "content": ask}], max_tokens=50, temperature=0.1,
                                cancelled=cancelled)
        if not out:
            return None
        de = re.sub(r"^\W*(das hei(ß|ss)t|das bedeutet|bedeutet)\W*", "", out.strip().split("\n")[0], flags=re.I)
        de = de.strip(" \"'„“”«»*")
        if not de or (not word and language_of(de) != "de"):
            return None
        de = de if de.endswith((".", "!", "?")) else de + "."
        if word:  # a word he asked about is something learned; her whole sentence explained is not
            self._taught(word, de)
        # the word in her Spanish voice, the meaning in German: "Extrañar." / "Das heißt: vermissen."
        return [f"{word[:1].upper()}{word[1:]}.", f"{EXPLAIN} {de}"] if word else [f"{EXPLAIN} {de}"]

    def _check(self, messages, text, question, cancelled):
        """{"ok": True, "reaction"} or {"ok": False, "tip", "frase"}; None when the reply was cancelled. A "mistake"
        that can't be checked (the wrong words aren't in what he said, a rewrite, a food or a name) counts as fine:
        a false correction costs more than a missed one."""
        ask = [{"role": "system", "content": CHECK_SYSTEM},
               {"role": "user", "content": f'Question: "{question}"\nHis answer: "{text}"'}]
        try:
            out = self.llm.complete(ask, max_tokens=90, temperature=0.1, fmt="json", cancelled=cancelled)
        except Exception:  # noqa: BLE001 - no check: take the answer
            return {"ok": True, "reaction": None}
        if out is None:
            return None
        try:
            d = json.loads(out)
        except ValueError:
            return {"ok": True, "reaction": None}
        if not isinstance(d, dict) or d.get("ok") is not False:
            r = str((d or {}).get("reaction") or "").strip() if isinstance(d, dict) else ""
            return {"ok": True, "reaction": r if r and "?" not in r and len(r.split()) <= 8 else None}
        tip = checked_tip(d, text)
        if not tip or not self._confirmed(tip):  # the second look: "extraño -> echo" didn't survive it
            return {"ok": True, "reaction": None}
        frase = re.sub(re.escape(tip["falsch"]), tip["richtig"], text, count=1, flags=re.I).strip()
        frase = frase[:1].upper() + frase[1:]
        return {"ok": False, "tip": tip, "frase": frase if frase.endswith((".", "!", "?")) else frase + "."}

    def _checked(self, messages, reply, text, question):
        """His answer, checked. Right: her reaction (gold on the orb) and the next question. Wrong: red on the orb,
        "Casi.", "Se dice: <his answer, corrected>", the rule in German, "Otra vez: <the question>"; after
        MAX_TRIES wrong answers she gives him the answer and goes on."""
        box = {}
        worker = threading.Thread(target=lambda: box.update(r=self._check(messages, text, question,
                                                                          cancelled=lambda: reply.cancelled)),
                                  daemon=True, name="check")
        worker.start()
        worker.join(THINK_S)
        if worker.is_alive():  # a correction takes ~3 s (check + second look): "A ver..." meanwhile
            yield self.rng.choice(THINKING)
            worker.join(20)
        res = box.get("r")
        if res is None or reply.cancelled:
            return
        if res["ok"]:
            self.tries = 0
            self.last_decision["move"] = "right"
            nq = self._advance()
            self.view = {**self.view, "mark": {"kind": "right", "id": reply.id}}
            yield self.rng.choice(self.praise)
            if nq:
                yield nq
            return
        tip, frase = res["tip"], res["frase"]
        self.tips.append(tip)
        self.tip(f"  Tipp: {tip['falsch']} -> {tip['richtig']}" + (f"  ({tip['warum']})" if tip["warum"] else ""))
        self.tries += 1
        mark = {"kind": "wrong", "falsch": tip["falsch"], "richtig": tip["richtig"], "frase": frase,
                "warum": tip["warum"], "id": reply.id}
        self.last_decision["move"] = "wrong"
        if self.tries >= MAX_TRIES:  # never stuck: the answer, and on
            self.tries = 0
            self.view = {**self.view, "mark": mark}
            yield f"Podés decir: {frase}"
            nq = self._advance()
            self.view = {**self.view, "mark": {**mark, "kind": "given"}}
            yield "¡Seguimos!"
            if nq:
                yield nq
            return
        self.view = {**self.view, "mark": mark}
        yield "Casi."
        yield f"Se dice: {frase}"
        if tip["warum"]:
            yield f"Tipp: {tip['warum']}"
        yield f"Otra vez: {question}"

    def _teacher_answer(self, text, cancelled):
        """His question about the Spanish itself ("is it common to say ...?"), answered in German like a kind
        teacher, in at most two short sentences. Asked without her persona (she only speaks Spanish)."""
        from .spanish_ears import language_of
        last = self._last_said()
        ask = ("A German beginner learning Spanish asks a question about the Spanish"
               + (f' (his partner had just said: "{last}")' if last else "") + f': "{text}". Answer in simple '
               "German, at most two short sentences, like a kind teacher. If he asks whether something is right or "
               "common, say yes or no, and give the natural way to say it IN SPANISH (for example: Man sagt: "
               "«Estoy cansado».). Only the answer.")
        out = self.llm.complete([{"role": "user", "content": ask}], max_tokens=90, temperature=0.2,
                                cancelled=cancelled)
        if not out:
            return None
        ans = " ".join(out.strip().split())
        return ans if language_of(ans) == "de" else None

    def _stream(self, messages, reply, text, coach=True, max_sentences=2, explain=False, teach=False, pre=(),
                post=(), use_llm=True, check=None):
        """Her reply: fixed lines first (`pre`), then the German explanation or teacher answer, then the model's
        sentence(s), then fixed lines (`post`: the next lesson question)."""
        context = self._last_said()
        try:
            if check is not None:
                yield from self._checked(messages, reply, text, check)
                return
            yield from pre
            if explain:  # "Das heißt: ..." (in the German voice), then the question again
                de = self._german(text, cancelled=lambda: reply.cancelled)
                if reply.cancelled:
                    return
                if de:
                    yield from de
                elif not post:  # no translation and no question to come back to: the model says it simpler
                    use_llm = True
                    messages = [dict(m) for m in messages]
                    messages[-1]["content"] = (f"{text}\n[He doesn't understand: say it again in Spanish, much "
                                               "shorter and simpler.]")
            if teach:
                answer = self._teacher_answer(text, cancelled=lambda: reply.cancelled)
                if reply.cancelled:
                    return
                if answer:
                    yield answer
            if use_llm:
                for s in self.llm.stream(messages, cancelled=lambda: reply.cancelled, max_tokens=80,
                                         temperature=0.7, max_sentences=max_sentences):
                    if post and "?" in s:  # her own question on top of the lesson's: two questions lose him
                        continue
                    yield s
            yield from post
        finally:
            self._idle.set()
            if coach and not reply.cancelled:  # a German turn needs no tip: she just told him the Spanish
                self._coach_q.put((text, context))

    def _coach_worker(self):
        while True:
            text, context = self._coach_q.get()
            self._idle.wait()
            if self._coach_once(text, context) is None:  # a reply started: check it again once the model is free
                self._coach_q.put((text, context))

    def _confirmed(self, tip):
        """A second look: which of the two sentences is correct Spanish? The tip stays only if the model picks the
        corrected one (Gemma once "corrected" a right "después fui" to "después iré")."""
        fixed = re.sub(re.escape(tip["falsch"]), tip["richtig"], tip["said"], count=1, flags=re.I)
        if fixed == tip["said"]:
            return True  # couldn't build the corrected sentence: the checks in checked_tip stand alone
        swap = self.rng.random() < 0.5
        one, two = (fixed, tip["said"]) if swap else (tip["said"], fixed)
        ask = ("Which sentence is correct, natural Spanish (ignore accents and punctuation)? Answer only 1 or 2.\n"
               f"1: {one}\n2: {two}")
        try:
            out = self.llm.complete([{"role": "user", "content": ask}], max_tokens=3, temperature=0.0) or ""
        except Exception:  # noqa: BLE001
            return False
        pick = re.search(r"[12]", out)
        return bool(pick) and pick.group() == ("1" if swap else "2")

    def _coach_once(self, text, context):
        # his sentence alone: with her line beside it, the small model "corrected" the topic (la silla -> el wifi)
        ask = f'He said: "{text}"'
        try:
            out = self.llm.complete([{"role": "system", "content": COACH}, {"role": "user", "content": ask}],
                                    max_tokens=90, temperature=0.1, fmt="json",
                                    cancelled=(lambda: not self._idle.is_set()) if self.live_tips else None)
        except Exception:  # noqa: BLE001 - a tip is never worth stopping the call
            return False
        if out is None:
            return None
        try:
            tip = checked_tip(json.loads(out), text)
        except ValueError:
            return False
        if tip and not self._confirmed(tip):
            tip = None
        if tip:
            self.tips.append(tip)
            self.tip(f"  Tipp: {tip['falsch']} -> {tip['richtig']}" + (f"  ({tip['warum']})" if tip["warum"] else ""))
        return True


def _norm(w):
    import unicodedata
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFD", w.lower()))


GERMAN_WORDS = re.compile(r"\b(ist|sind|nicht|der|die|das|den|dem|und|oder|muss|musst|statt|weil|für|wegen|sein|im|"
                          r"vor|nach|bei|mit|wird|heißt|man|sagt|immer|hier|braucht|nimmt|Vergangenheit|Präsens|"
                          r"Verb|weiblich|männlich)\b|[äöüß]", re.I)


PROTECTED = {"chipa", "terere", "mate", "asado", "sopa", "mbeju", "costanera", "asuncion", "luque", "paraguay",
             "chipita", "luz", "sofia", "cumbia", "polka"}  # + the learner's name (AmigaBrain)


def is_german(text):
    """An explanation in German, even when it quotes the Spanish words it explains."""
    bare = re.sub(r"'[^']*'|\"[^\"]*\"|„[^“]*“|«[^»]*»", " ", text)
    return len(GERMAN_WORDS.findall(bare)) >= 2


def checked_tip(d, said):
    """The coach's answer, only if it can be checked: the wrong words must really be in what he said, and the fix must
    change only a few words by more than accents (a small model otherwise "corrects" chipa to chicha or rewrites the
    sentence). Shown trimmed to what changed, with a word of context. The explanation is kept if it is German."""
    if not isinstance(d, dict) or d.get("ok") is not False:
        return None
    falsch, richtig, warum = (str(d.get(k) or "").strip() for k in ("falsch", "richtig", "warum"))
    f, r = falsch.strip(" .,!?¿¡").split(), richtig.strip(" .,!?¿¡").split()
    nf, nr, ns = [_norm(w) for w in f], [_norm(w) for w in r], " ".join(_norm(w) for w in said.split())
    if not nf or not nr or nf == nr or " ".join(nf) not in ns:
        return None
    a = 0
    while a < min(len(nf), len(nr)) and nf[a] == nr[a]:
        a += 1
    b = 0
    while b < min(len(nf), len(nr)) - a and nf[-1 - b] == nr[-1 - b]:
        b += 1
    if max(len(nf) - a - b, len(nr) - a - b) > 3:  # a rewrite, not a fix
        return None
    names = {_norm(w) for i, w in enumerate(said.split()) if i and w[:1].isupper()}
    if set(nf[a:len(nf) - b]) & (PROTECTED | names):  # foods, places and names are never "mistakes"
        return None
    a = max(a - 1, 0)  # one word of context before the change
    falsch, richtig = " ".join(f[a:len(f) - b]), " ".join(r[a:len(r) - b])
    return {"said": said, "falsch": falsch, "richtig": richtig, "warum": warum if is_german(warum) else ""}
