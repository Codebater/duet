"""The friend's decisions with hand-made readings and a stand-in language model (no Ollama needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from duet.friend import (BACKCHANNELS, GLADLY, GREETINGS, IM_GOOD, IM_OKAY, NUDGES, SAFETY,  # noqa: E402
                         FriendBrain, feeling, mode_of)
from duet.prosody import ToneReading  # noqa: E402
from duet.understand import Reading  # noqa: E402


class FakeLlm:
    """Records what it was asked, answers one fixed sentence."""

    def __init__(self):
        self.asked = []

    def stream(self, messages, cancelled=lambda: False, max_sentences=2, **kw):
        self.asked.append(messages)
        yield "That makes sense."

    def complete(self, messages, cancelled=None, **kw):
        return '{"facts": []}'


def reading(text, act="smalltalk", neg=0.0, pos=0.0, complete=0.9):
    return Reading(text, "out_of_scope", 0.5, act, 0.95, 1.0 if act == "greeting" else 0.0,
                   1.0 if act == "request" else 0.0, {"neg": neg, "neu": 1 - neg - pos, "pos": pos}, complete, 1.0)


calm = ToneReading(0.0, 0.0, "")


def turn(b, text, act="smalltalk", **kw):
    r = b.caller_turn(reading(text, act, **kw), calm)
    r.sentences.extend(r.stream or [])
    b.heard(r, len(r.sentences), interrupted=False)
    return r


def test_modes():
    cases = [("I got the job! They called this morning", "request", ("listen", "good_news")),
             ("I'm kind of nervous about leaving my friends", "smalltalk", ("listen", "bad_news")),
             ("My sister Anna is going to help me move", "request", ("listen", "story")),
             ("What do you think I should do to meet people?", "request", ("help", "ideas")),
             ("What's your favorite city?", "request", ("chat", "about_you")),
             ("What day is it today?", "smalltalk", ("help", "answer")),
             ("Hey, guess what!", "greeting", ("listen", "teaser")),
             ("Thanks, I gotta go. Bye!", "goodbye", ("ritual", "bye")),
             ("Hey, how are you?", "greeting", ("ritual", "how_are_you"))]
    for text, act, want in cases:
        r = reading(text, act)
        r.sentiment = feeling(r, text)
        assert mode_of(r, text) == want, (text, mode_of(r, text), want)


def test_help_me_move_is_not_a_request_for_advice():
    r = reading("My sister is going to help me move next week", "request")
    assert mode_of(r, r.text)[0] != "help"


def test_good_news_is_met_with_energy_and_an_opener():
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    r = turn(b, "I got the job! The design studio called me this morning.", "request")
    assert b.last_decision["move"] == "listen_good" and r.plan.register in ("excited", "warm")
    assert r.sentences[0] in ("Oh wow!", "No way!", "That's amazing!", "Oh, nice!", "Aw, that's great.")
    # the opener is handed to the model as the start of its own reply, so it continues instead of repeating it
    assert b.llm.asked[-1][-1] == {"role": "assistant", "content": r.sentences[0]}
    turn(b, "I start in two weeks.", "request")
    assert b.last_decision["move"] == "listen_good_more"  # stays on their news


def test_bad_news_gets_the_gentle_voice_and_no_advice():
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    turn(b, "I had a really rough day, my best friend and I had a big fight.", neg=0.8)
    assert b.last_decision["move"] == "listen_bad" and b.last_decision["register"] in ("gentle", "serious")
    note = b.llm.asked[-1][-2]["content"]
    assert "No advice" in note and "at least" in note


def test_mood_holds_through_one_neutral_line():
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    turn(b, "I'm so sad, my dog died yesterday.", neg=0.9)
    turn(b, "Yeah. I just keep thinking about it.")
    assert b.last_decision["register"] in ("gentle", "serious")


def test_danger_leaves_friend_mode():
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    r = b.caller_turn(reading("Sometimes I feel like I want to die", neg=0.9), calm)
    assert r.kind == "safety" and r.sentences == SAFETY and r.stream is None


def test_questions_get_quick_answers_and_stories_room():
    b = FriendBrain(FakeLlm(), memory_path=None)
    assert b.endpoint(reading("What would you do?"), 1.0) <= 0.3
    assert b.endpoint(reading("Thanks!", "thanks"), 1.0) <= 0.3
    assert b.endpoint(reading("So my sister called me and", complete=0.1), 3.0) >= 1.0
    assert 0.5 <= b.endpoint(reading("My sister called me this morning.", "request"), 5.0) <= 0.8


def test_no_third_question_in_a_row():
    llm = FakeLlm()
    b = FriendBrain(llm, memory_path=None, seed=1)
    llm.stream = lambda messages, **kw: (llm.asked.append(messages), (yield "What happened next?"))[1]
    turn(b, "We went to the beach today.")
    turn(b, "Then it started raining.")
    turn(b, "We got soaked.")
    assert "don't ask a question this time" in llm.asked[-1][-2]["content"]


# ---- from the friend's first live call (runs/live-20261103-212305)

def test_enjoying_a_movie_is_nice_not_amazing():
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    loud = ToneReading(0.55, -0.3, "angry")  # how the prosody head read every turn of that call
    for _ in range(3):
        r = b.caller_turn(reading("Yes, that's nice. I'm enjoying. Collision course, it's called the movie."), loud)
        r.sentences.extend(r.stream or [])
        b.heard(r, len(r.sentences), interrupted=False)
    assert b.last_decision["move"] in ("listen_nice", "listen_good_more")
    assert b.last_decision["register"] not in ("excited", "lively")
    assert "amazing" not in b.last_decision["opener"].lower()


def test_their_usual_energy_is_not_excitement():
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    loud = ToneReading(0.6, 0.0, "angry")
    first = b.caller_turn(reading("I'm watching tv in my room"), loud)
    assert b.mood.arousal > 0.5  # the first turn's energy is met as it is
    for text in ("I'm waiting for my food", "They're robbing a bank in the movie", "They all wear masks"):
        b.caller_turn(reading(text), loud)
    assert b.mood.arousal < 0.35  # by the fourth turn their usual level reads as their normal


def test_thank_you_lets_speak_does_not_end_the_call():
    r = reading("Thank you. Let's speak.", "goodbye")
    assert mode_of(r, r.text)[1] != "bye"
    r = reading("Goodbye, hear you soon", "goodbye")
    assert mode_of(r, r.text)[1] == "bye"


def test_every_reply_starts_with_a_sound():
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    for text, act in (("Yes, bro.", "affirm"), ("Hey, how are you?", "greeting"), ("Thanks so much", "thanks"),
                      ("I think it depends", "smalltalk")):
        r = b.caller_turn(reading(text, act), calm)
        assert r.sentences and r.sentences[0], text


def test_a_full_stop_shortens_the_wait():
    b = FriendBrain(FakeLlm(), memory_path=None)
    assert b.endpoint(reading("I don't know, they're robbing some bank, I think.", complete=0.01), 2.0) <= 0.8


def test_the_mode_head_cannot_give_unasked_advice():
    r = reading("Yeah, I just keep thinking about it", neg=0.6)
    r.mode, r.mode_p, r.feeling, r.feeling_p = "ask_advice", 0.9, "negative", 0.9
    r.sentiment = feeling(r, r.text)
    assert mode_of(r, r.text) == ("listen", "bad_news")
    r = reading("What would you do?")
    r.mode, r.mode_p, r.feeling, r.feeling_p = "ask_advice", 0.9, "neutral", 0.9
    assert mode_of(r, r.text) == ("help", "ideas")


# ---- from the friend's second live call (runs/live-20261103-223017): it looped on one sentence

class LoopingLlm(FakeLlm):
    """Says the same thing every time, unless told it already said it."""

    def stream(self, messages, cancelled=lambda: False, max_sentences=2, **kw):
        self.asked.append(messages)
        told = any("you already said" in m["content"] for m in messages if m["role"] == "user")
        yield "Okay, what's new with you lately?" if told else "Aw, I see you're frustrated and I messed up."


def test_a_repeated_line_is_written_again():
    llm = LoopingLlm()
    b = FriendBrain(llm, memory_path=None, seed=1)
    first = turn(b, "No, no, no, I'm not sure which movie you're talking about.", neg=0.6)
    second = turn(b, "Okay, okay, I'm sorry.")
    assert "frustrated" in " ".join(first.sentences)
    assert "frustrated" not in " ".join(second.sentences) and "new" in " ".join(second.sentences)


def test_say_something_else_is_not_say_that_again():
    r = reading("Can you say something else?", "repeat")
    assert mode_of(r, r.text) == ("chat", "something_new")
    r = reading("Sorry, can you say that again?", "repeat")
    assert mode_of(r, r.text) == ("repeat", "repeat")


def test_how_are_you_gets_an_answer_even_after_a_hard_moment():
    b = FriendBrain(LoopingLlm(), memory_path=None, seed=1)
    turn(b, "No, no, no, I'm not sure which movie you're talking about.", neg=0.6)
    r = turn(b, "Oh, okay, I'm sorry. Tell me, how are you?", "greeting")
    assert r.sentences[0] in IM_GOOD + IM_OKAY
    assert b.last_decision["move"] == "how_are_you"


# ---- after the friend's third live call (runs/live-20261103-224122): natural variation, help, jokes, hums

def test_can_you_help_me_is_yes_gladly():
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    r = turn(b, "Can you help me?")
    assert r.sentences[0] in GLADLY and r.sentences[-1].endswith(("?", "!")) and r.stream is None
    r = reading("Can you help me with my CV?", "request")
    assert mode_of(r, r.text)[1] != "help_offer"  # with a topic, the model answers it


def test_a_joke_when_asked():
    r = reading("Do you have humor? Can you tell me a joke?")
    assert mode_of(r, r.text) == ("chat", "joke")


def test_a_hum_is_a_nudge_or_an_answer():
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    turn(b, "I went to the beach today.")
    r = turn(b, "Mmm.", "backchannel")
    assert r.sentences[0] in NUDGES
    llm = LoopingLlm()
    b = FriendBrain(llm, memory_path=None, seed=1)
    b.caller_turn(reading("I went to the beach today."), calm)  # LoopingLlm's reply doesn't end in "?"
    b.heard(type("R", (), {"sentences": ["Want to talk about it?"], "kind": "friend", "cancelled": False})(), 1, False)
    assert b.caller_turn(reading("Yeah.", "affirm"), calm).stream is not None  # their answer: the model goes on


def test_greetings_and_murmurs_vary():
    openings = {tuple(FriendBrain(FakeLlm(), memory_path=None, seed=k).opening().sentences) for k in range(12)}
    assert len(openings) >= 4 and all(o[0] in GREETINGS and o[-1].endswith("?") for o in openings)
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    said = [b.backchannel_line(reading("so I was walking home and"))[0] for _ in range(6)]
    assert len(set(said)) >= 4 and set(said) <= set(BACKCHANNELS["neutral"])


def test_glad_to_talk_by_default():
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    turn(b, "I'm just making some pasta right now.")
    assert b.last_decision["register"] == "lively"


# ---- after the friend's fourth live call (runs/live-20261103-230111)

def test_jokes_stories_and_questions_about_it_are_read_right():
    cases = [("Tell me a better joke.", ("chat", "joke")),
             ("Tell me good, good, good joke, the best one you know.", ("chat", "joke")),
             ("I didn't understand the joke.", ("chat", "explain_joke")),
             ("Tell me a story.", ("chat", "tell_story")),
             ("No, do you like music?", ("chat", "about_you")),
             ("How long did you wait?", ("chat", "about_you")),
             ("Mm-hmm.", ("chat", "hum"))]
    for text, want in cases:
        r = reading(text)
        r.sentiment = feeling(r, text)
        assert mode_of(r, text) == want, (text, mode_of(r, text))


def test_a_joke_request_gets_the_joke_move_even_read_as_repeat():
    llm = FakeLlm()
    b = FriendBrain(llm, memory_path=None, seed=1)
    for text, act, move in (("Tell me a joke, my friend.", "repeat", "joke"), ("Tell me a story.", "repeat", "story"),
                            ("I didn't understand the joke.", "smalltalk", "explain_joke")):
        b.caller_turn(reading(text, act), calm)
        assert b.last_decision["move"] == move, (text, b.last_decision["move"])


def test_the_style_examples_hold_nothing_to_mistake_for_their_words():
    from duet.friend import PERSONA
    assert "pasta" not in PERSONA and "They: '" not in PERSONA and "not about the person" in PERSONA


def test_how_long_did_you_wait_has_the_facts():
    import time
    llm = FakeLlm()
    b = FriendBrain(llm, memory_path=None, seed=1)
    b.call_start = b.last_heard_t = time.time() - 700  # greeted, then 11+ minutes of quiet
    turn(b, "How long did you wait?")
    note = llm.asked[-1][-2]["content"] if llm.asked[-1][-1]["role"] == "assistant" else llm.asked[-1][-1]["content"]
    assert "quiet for about 12 minutes" in note or "quiet for about 11 minutes" in note


# ---- after the demo call (runs/live-20261103-235822)

def test_an_offer_to_tell_gets_a_curious_yes():
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    r = turn(b, "I am good. Do you want to know about Assad?", "request")
    assert b.last_decision["move"] == "offer" and r.stream is None and "tell" in " ".join(r.sentences).lower()


def test_a_name_is_someone_they_know():
    from duet.friend import PERSONA
    assert "never a famous person" in PERSONA


def test_a_plain_no_is_not_frustration():
    r = reading("No bro, no bro. Assad is a friend.", "deny")
    r.feeling, r.feeling_p = "negative", 0.8
    assert feeling(r, r.text)["neg"] < 0.5
    r = reading("No, I'm really sad today.", "deny")
    r.feeling, r.feeling_p = "negative", 0.8
    assert feeling(r, r.text)["neg"] > 0.5


def test_the_models_capitals_are_kept():
    from duet.llm import clean
    assert clean("that's wild, doing websites together!") == "that's wild, doing websites together!"


def test_snapshot_survives_a_streaming_reply():
    b = FriendBrain(FakeLlm(), memory_path=None, seed=1)
    r = b.caller_turn(reading("We went to the beach today."), calm)
    snap = b.snapshot()  # the reply's stream is live: it must not be copied
    b.restore(snap)
    assert b._ids == r.id


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
