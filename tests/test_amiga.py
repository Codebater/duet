"""The Spanish partners (amiga.py, spanish_ears.py) without a model or a microphone: the cards, the coach's checks,
the language guess, goodbyes, German turns, snapshots."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from duet.amiga import CARDS, AmigaBrain, checked_tip, load_cards  # noqa: E402
from duet.spanish_ears import NAMES, language_of  # noqa: E402
from duet.understand import Reading  # noqa: E402


class FakeLlm:
    def __init__(self):
        self.calls = []

    def stream(self, messages, cancelled=lambda: False, max_sentences=2, **kw):
        self.calls.append(messages)
        yield "¡Qué lindo!"
        yield "¿Y después?"

    def complete(self, messages, cancelled=None, **kw):
        ask = messages[-1]["content"]
        if "Translate into" in ask or "mean in German" in ask:
            return "Ich vermisse dich."
        if "His answer:" in ask:  # his answer, checked: his real mistakes from call 4 are wrong, the rest fine
            if "estoy mucho trabajo" in ask:
                return json.dumps({"ok": False, "falsch": "estoy mucho trabajo", "richtig": "tengo mucho trabajo",
                                   "warum": "Man sagt tener, nicht estar"})
            if "comí chicha" in ask:  # a "mistake" that can't be checked (a food): taken as right
                return json.dumps({"ok": False, "falsch": "chipa", "richtig": "chicha", "frase": "x", "warum": ""})
            return json.dumps({"ok": True, "reaction": "¡Qué bien!"})
        if "asks a question about the Spanish" in ask:  # the teacher's German answer
            return "Nein, das sagt man nicht. Man sagt: Estoy cansado."
        if "Answer only 1 or 2" in ask:  # the second look: the sentence with "fui" is the right one
            first = ask.split("1: ")[1].split("\n")[0] if "1: " in ask else ""
            good = ("fui", "tengo")
            return "1" if any(g in first for g in good) else "2"
        return json.dumps({"ok": False, "falsch": "yo voy", "richtig": "yo fui",
                           "warum": "Vergangenheit: ayer fui, nicht voy"})


def reading(text):
    return Reading(text, "out_of_scope", 0.9, "smalltalk", 0.9, 0.0, 0.0, {"neg": 0, "neu": 1, "pos": 0}, 0.9, 1.0)


def brain(**kw):
    return AmigaBrain(FakeLlm(), "novia", seed=1, tip=lambda s: None, profile=kw.pop("profile", None),
                      learner=kw.pop("learner", None), **kw)


def test_cards_are_read():
    decks = load_cards(CARDS)
    if not decks:  # spanish-cards isn't next to duet on this machine
        return
    assert 7 in decks and any(c["es"] == "Te extraño." for c in decks[7])
    assert all(c["es"] and c["de"] for d in decks.values() for c in d)


def test_prompt_has_her_knowledge():
    b = brain()
    assert "Luz" in b.system and "vos" in b.system and "Alex" in b.system  # the default learner
    assert "Sofía" in AmigaBrain(FakeLlm(), "cita", tip=lambda s: None, profile=None).system


def test_tip_must_be_in_what_he_said():
    said = "Ayer yo voy a la Costanera con mis amigos."
    whole = {"ok": False, "falsch": "Ayer yo voy a la Costanera con mis amigos",
             "richtig": "Ayer yo fui a la Costanera con mis amigos", "warum": "Präsens statt Präteritum: 'voy' ist falsch"}
    t = checked_tip(whole, said)
    assert t and t["falsch"] == "yo voy" and t["richtig"] == "yo fui" and t["warum"]
    assert checked_tip({"ok": False, "falsch": "yo comí", "richtig": "yo como", "warum": ""}, said) is None
    assert checked_tip({"ok": True}, said) is None


def test_tip_never_touches_foods_names_or_accents():
    assert checked_tip({"ok": False, "falsch": "comer chipa", "richtig": "comer chicha", "warum": ""},
                       "Vamos a comer chipa") is None
    assert checked_tip({"ok": False, "falsch": "esta", "richtig": "está", "warum": ""}, "esta bien") is None


def test_spanish_explanations_are_dropped():
    t = checked_tip({"ok": False, "falsch": "yo trabajo", "richtig": "yo trabajé", "warum": "Uso el pretérito"},
                    "hoy yo trabajo mucho")
    assert t and t["warum"] == ""


def test_language_guess():
    assert language_of("Wie sagt man auf Spanisch, ich vermisse dich?") == "de"
    assert language_of("How does Spanish say I miss you?") == "en"
    assert language_of("Sí, me gusta mucho bailar.") == "es"
    assert language_of("Eu não sei, obrigado") == "other"


def test_his_name():
    import re
    from duet.spanish_ears import _clean
    fixes = [(re.compile(r"\bAleks\b", re.I), "Alex")]  # from his about.json, e.g. Whisper's spelling of his name
    assert _clean(["Me llamo Aleks. Mi amigo Aleksandra."], fixes) == "Me llamo Alex. Mi amigo Aleksandra."
    assert not NAMES  # nobody's name in the code


def test_the_learner_comes_from_his_file():
    path = Path(__file__).resolve().parent.parent / "runs" / "test-amiga-about.json"
    path.write_text(json.dumps({"name": "Tomás", "about": "He bakes bread."}), encoding="utf-8")
    try:
        b = brain(learner=path)
        assert "Tomás's girlfriend" in b.system and "He bakes bread." in b.system and "Alex" not in b.system
        b.lesson_i = len(b.lessons) - 1  # the job lesson has no name; check the hint filling on Sofía instead
        c = AmigaBrain(FakeLlm(), "cita", tip=lambda s: None, profile=None, learner=path)
        c.step_i = 1
        c._ask_step()
        assert c.view["hint"] == "Me llamo Tomás. ¿Y vos?"
    finally:
        path.unlink(missing_ok=True)


def test_goodbye_ends_the_call():
    b = brain()
    r = b.caller_turn(reading("Bueno, chau mi amor"), None)
    assert r.end_call and r.stream is None


def test_german_turn_is_marked_and_not_coached():
    b = brain(live_tips=False)
    r = b.caller_turn(reading("Wie sagt man ich vermisse dich?"), None)
    list(r.stream)
    assert "[he said this in German]" in b.llm.calls[-1][-1]["content"]
    assert b._coach_q.empty()


def test_spoken_call_coaches_after_the_call(tmp_path=None):
    b = brain(live_tips=False)
    b.lesson_i = len(b.lessons)  # the lessons are done: free talk is coached after the call (answers: live)
    r = b.caller_turn(reading("Ayer yo voy a la Costanera."), None)
    r.sentences = list(r.stream)
    b.heard(r, len(r.sentences), False)
    assert not b.tips and b._coach_q.qsize() == 1  # nothing ran during the call
    import duet.amiga as amiga
    old, amiga.TIPS_DIR = amiga.TIPS_DIR, Path(__file__).resolve().parent.parent / "runs" / "test-amiga-tips"
    try:
        path = b.close()
        assert b.tips and b.tips[0]["richtig"] == "yo fui" and path and path.exists()
        path.unlink()
    finally:
        amiga.TIPS_DIR = old


def test_what_does_it_mean_gets_an_explanation():
    for said in ("What means this", "¿Qué significa eso?", "Was heißt das?", "mia hermosa no comprende",
                 "No entiendo", "I don't understand"):
        b = brain(live_tips=False)
        b.opening()  # "¡Hola, ...!" + the lesson's first question
        b._remember("assistant", "Te extraño mucho.")
        r = b.caller_turn(reading(said), None)
        out = list(r.stream)
        # German first, then her question again, so he can answer it now (his third call: "Ask me something.")
        assert out == ["Das heißt: Ich vermisse dich.", "¿Cómo estás?"], (said, out)
        assert not b.llm.calls, said
        assert r.plan.delivery.speed < b.plan.delivery.speed and b._coach_q.empty(), said


def test_a_word_he_asks_about():
    b = brain(live_tips=False)
    b._remember("assistant", "Te extraño mucho.")
    r = b.caller_turn(reading("¿Qué significa extrañar?"), None)
    out = list(r.stream)
    assert out[:2] == ["Extrañar.", "Das heißt: Ich vermisse dich."]
    assert b.sentence_lang(out[0]) == "es" and b.sentence_lang(out[1]) == "de"


def test_slower_please():
    b = brain(live_tips=False)
    b.opening()
    r = b.caller_turn(reading("Más despacio, por favor"), None)
    assert list(r.stream) == ["¿Cómo estás?"] and r.plan is b.help_plan


def test_german_explanations_are_spoken_in_german():
    b = brain()
    assert b.sentence_lang("Das heißt: Ich vermisse dich.") == "de"
    assert b.sentence_lang("Das heißt: te extraño.") == "de"
    assert b.sentence_lang("Te extraño mucho, mi amor.") == "es"


def test_beginner_level_by_default():
    b = brain()
    assert b.level == 1 and "A1" in b.system and b.endpoint(None, 2.0) > 0.75
    b2 = AmigaBrain(FakeLlm(), "novia", tip=lambda s: None, level=2, profile=None)
    assert "A2" in b2.system and b2.plan.delivery.speed > b.plan.delivery.speed


def test_snapshot_survives_a_streaming_reply():
    b = brain()
    r = b.caller_turn(reading("Hoy trabajé mucho."), None)
    snap = b.snapshot()
    b.restore(snap)
    assert b._ids == r.id


def test_snapshot_leaves_the_orb_alone():
    import threading

    class Uncopyable:  # like the orb page: it holds the conductor, its threads and ONNX sessions
        def __deepcopy__(self, memo):
            raise TypeError("cannot pickle InferenceSession")

        def tip(self, text):
            pass

    b = brain()
    orb = Uncopyable()
    orb.lock = threading.Lock()
    b.tip = orb.tip
    b.restore(b.snapshot())
    assert b.tip == orb.tip


# ---- the teacher (docs/amiga-teacher.md): his real moments from the first two calls
def say(b, text):
    r = b.caller_turn(reading(text), None)
    out = list(r.stream) if r.stream is not None else []
    r.sentences = list(r.sentences) + out
    b.heard(r, len(r.sentences), False)
    return r


def test_his_ways_of_asking_are_heard():
    from duet.amiga import HELP, RESTART
    for s in ("What this means?", "Was heißt das nochmal?", "Was? Was? Noch mal auf Deutsch?",
              "Nochmal? Wie, was heißt das übersetzt?", "mia hermosa no comprende", "What means vos tenés razón?"):
        assert HELP.search(s), s
    for s in ("Okay, starten wir nochmal von neu.", "Can we start from you?", "Let's start again"):
        assert RESTART.search(s), s


def test_a_misheard_phrase_is_matched_by_sound():
    from duet.amiga import closest_span
    assert closest_span("Boston is right", "Exacto. Vos tenés razón.").lower().startswith("vos tenés")
    assert closest_span("boost", "Exacto. Vos tenés razón.") == "Vos"
    assert closest_span("perro", "Exacto. Vos tenés razón.") is None


def test_taught_phrase_then_he_repeats_it_and_gets_praise():
    b = brain(live_tips=False)
    b.llm.stream = lambda messages, **kw: iter(["Se dice: Vi un poco de tele.", "Decilo vos."])
    say(b, "Wie sage ich, ich habe ein bisschen ferngeschaut?")
    assert b.practice == "Vi un poco de tele."
    b.llm.calls = []
    del b.llm.stream  # back to FakeLlm.stream
    r = say(b, "Vi un poco de tele.")
    assert r.sentences[0] in b.praise and b.practice is None  # checked as his answer, and right
    assert r.sentences[1] == "¿Qué hiciste hoy?" and not b.llm.calls  # and on to the next question
    assert b.repeated == ["Vi un poco de tele."]


def test_a_near_miss_gets_the_phrase_once_more_slowly():
    b = brain(live_tips=False)
    b.lesson_i = len(b.lessons)  # no lesson question pending (with one, a near miss is checked as an answer)
    b.practice = "Vi un poco de tele."
    from duet.amiga import similar
    assert similar("Vi poco tele", "Vi un poco de tele.") >= 0.72  # close enough counts as said
    r = b.caller_turn(reading("Vi la tele"), None)
    assert list(r.stream) == ["Vi un poco de tele.", "Otra vez."] and r.plan is b.help_plan


def test_restart_starts_the_talk_again():
    b = brain(live_tips=False)
    b.step_i = 3
    r = b.caller_turn(reading("Okay, starten wir nochmal von neu."), None)
    out = list(r.stream)
    assert out[1] == "¿Cómo estás?" and b.step_i == 0 and out[0] in b.char.greetings


def test_goodbye_recaps_what_he_learned():
    b = brain(live_tips=False)
    b._taught("nada", "nichts")
    b._taught("te extraño", "ich vermisse dich")
    r = b.caller_turn(reading("Chau, mi amor"), None)
    assert r.end_call and r.sentences[0] == "Hoy aprendiste: nada, te extraño."


def test_mixed_turns_are_not_coached():
    b = brain(live_tips=False)
    assert not b._coachable("como haistest du", "es")
    assert not b._coachable("No more of touch.", "es")
    assert b._coachable("soy mucho triste", "es")


def test_profile_goes_into_the_prompt_and_grows(tmp=None):
    import duet.amiga as amiga
    path = Path(__file__).resolve().parent.parent / "runs" / "test-amiga-profile.json"
    path.write_text(json.dumps({"calls": 2, "mistakes": [{"falsch": "es bien", "richtig": "está bien", "n": 2}],
                                "phrases": [{"es": "Vos tenés razón", "de": "Du hast recht", "n": 1}]}),
                    encoding="utf-8")
    try:
        b = AmigaBrain(FakeLlm(), "novia", seed=1, tip=lambda s: None, live_tips=False, profile=path)
        assert '"es bien" should be "está bien"' in b.system  # practice phrases: the lessons do that now
        b.finished.append(0)
        b._taught("Vos tenés razón", None)
        b.repeated.append("Vos tenés razón")
        b.tips.append({"said": "soy mucho triste", "falsch": "soy mucho", "richtig": "estoy muy", "warum": ""})
        old, amiga.TIPS_DIR = amiga.TIPS_DIR, path.parent / "test-amiga-tips"
        try:
            tips_file = b.close()
        finally:
            amiga.TIPS_DIR = old
        p = json.loads(path.read_text(encoding="utf-8"))
        assert p["calls"] == 3 and any(m["falsch"] == "soy mucho" for m in p["mistakes"])
        vos = next(x for x in p["phrases"] if x["es"] == "Vos tenés razón")
        assert vos["n"] == 2 and vos["ok"] == 1 and vos["de"] == "Du hast recht"
        assert p["next_lesson"]["novia"] == 1  # he finished "Tu día": next call starts the next lesson
        if tips_file:
            tips_file.unlink()
    finally:
        path.unlink(missing_ok=True)


# ---- conversation lessons (lessons.py): she walks him through questions and keeps the talk going
def test_opening_asks_the_first_question():
    b = brain(live_tips=False)
    r = b.opening()
    assert r.sentences[0] in b.char.greetings and r.sentences[1] == "¿Cómo estás?"
    assert b.view["q_de"] == "Wie geht es dir?" and b.view["step"] == 1


def test_his_answer_gets_a_reaction_then_the_next_question():
    b = brain(live_tips=False)
    b.opening()
    r = say(b, "Estoy bien, un poco cansado.")
    assert r.sentences[0] in b.praise and r.sentences[1:] == ["¿Qué hiciste hoy?"]  # right: praise, next
    assert b.view["step"] == 2 and b.view["mark"]["kind"] == "right" and not b.llm.calls


def test_stuck_gets_an_answer_to_borrow_then_praise_and_on():
    b = brain(live_tips=False)
    b.opening()
    r = say(b, "No sé qué decir")
    assert r.sentences == ["Podés decir: Estoy bien, gracias.", "Das heißt: Mir geht es gut, danke.", "Decilo vos."]
    r = say(b, "Estoy bien, gracias.")
    assert r.sentences[1] == "¿Qué hiciste hoy?" and "Estoy bien, gracias." in b.repeated


def test_ask_me_something_goes_on():
    b = brain(live_tips=False)
    b.opening()
    r = say(b, "Ask me something.")
    assert r.sentences == ["¡Dale!", "¿Qué hiciste hoy?"]


def test_a_question_about_the_spanish_gets_a_german_teacher_answer():
    b = brain(live_tips=False)
    b.opening()
    b._remember("assistant", "Mi día está cansado.")
    r = say(b, "But is it a common answer, my day was tired?")
    assert r.sentences == ["Nein, das sagt man nicht. Man sagt: Estoy cansado.", "¿Cómo estás?"]
    assert b.sentence_lang(r.sentences[0]) == "de"


def test_how_do_i_say_waits_for_him():
    b = brain(live_tips=False)
    b.opening()
    b.llm.stream = lambda messages, **kw: iter(["Se dice: Te extraño.", "Decilo vos."])
    r = say(b, "How do I say I miss you?")
    assert r.sentences == ["Se dice: Te extraño.", "Decilo vos."] and b.practice == "Te extraño."


def test_the_lesson_runs_on_into_the_next():
    b = brain(live_tips=False)
    b.opening()
    for _ in range(len(b.lessons[0].steps) - 1):
        say(b, "Sí, bien.")
    r = say(b, "Sí, te extrañé mucho.")
    assert r.sentences[-1] == "Ahora, otra cosa. ¿Te gusta bailar?"
    assert b.finished == [0] and b.view["lesson"] == "Lo que te gusta"


# ---- live correction (after call 4: "if I say something wrong she should turn red and correct me, then ask
#      again the question until I answer right")
def test_a_wrong_answer_is_corrected_and_the_question_asked_again():
    b = brain(live_tips=False)
    b.opening()
    r = say(b, "sí un poco estoy mucho trabajo")
    assert r.sentences == ["Casi.", "Se dice: Sí un poco tengo mucho trabajo.", "Tipp: Man sagt tener, nicht estar",
                           "Otra vez: ¿Cómo estás?"]
    assert b.view["mark"]["kind"] == "wrong" and b.view["mark"]["richtig"] == "tengo" and b.view["step"] == 1
    assert b.sentence_lang(r.sentences[2]) == "de" and b.tries == 1 and b.tips[0]["falsch"] == "estoy"
    r = say(b, "Sí, un poco, tengo mucho trabajo.")  # right now: gold, and on
    assert r.sentences[0] in b.praise and r.sentences[1:] == ["¿Qué hiciste hoy?"]
    assert b.view["mark"]["kind"] == "right" and b.tries == 0


def test_repeating_her_se_dice_is_not_a_mistake():
    b = brain(live_tips=False)
    b.opening()
    b.practice = "A las seis."
    r = say(b, "Se dice a las seis.")  # call 5: he said her "Se dice" too
    assert r.sentences[0] in b.praise and "A las seis." in b.repeated


def test_three_wrong_answers_and_she_gives_it_and_goes_on():
    b = brain(live_tips=False)
    b.opening()
    for _ in range(2):
        say(b, "sí un poco estoy mucho trabajo")
    r = say(b, "sí un poco estoy mucho trabajo")
    assert r.sentences == ["Podés decir: Sí un poco tengo mucho trabajo.", "¡Seguimos!", "¿Qué hiciste hoy?"]
    assert b.tries == 0 and b.view["step"] == 2


def test_a_mistake_that_cannot_be_checked_counts_as_right():
    b = brain(live_tips=False)
    b.opening()
    r = say(b, "comí chicha")
    assert r.sentences[1] == "¿Qué hiciste hoy?" and not b.tips


def test_an_answer_in_german_gets_the_spanish_to_say():
    b = brain(live_tips=False)
    b.opening()
    b.llm.stream = lambda messages, **kw: iter(["Se dice: Bailo muy mal."])
    r = say(b, "Ich bin sehr schlecht im Tanzen.")
    assert r.sentences == ["Se dice: Bailo muy mal.", "Decilo vos."] and b.practice == "Bailo muy mal."


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
