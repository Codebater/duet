"""The brain's policy with hand-made readings: greeting vs intent, interruptions, mood registers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from duet import replies as R  # noqa: E402
from duet.brain import Brain  # noqa: E402
from duet.prosody import ToneReading  # noqa: E402
from duet.understand import Reading  # noqa: E402


def reading(text, act="request", intent="out_of_scope", intent_p=0.9, greeting=0.0, request=None, neg=0.0, pos=0.0):
    request = (1.0 if act == "request" else 0.0) if request is None else request
    return Reading(text, intent, intent_p, act, 0.95, greeting, request,
                   {"neg": neg, "neu": 1 - neg - pos, "pos": pos}, 0.9, 1.0)


calm = ToneReading(0.0, 0.0, "")
excited = ToneReading(0.6, 0.2, "")


def test_greeting_alone_gets_greeting_and_invitation():
    b = Brain()
    r = b.caller_turn(reading("Hi there", act="greeting", greeting=0.98), calm)
    assert r.kind == "greet"
    assert r.sentences[0] in R.GREET_BACK.values() and r.sentences[-1] in R.INVITE.values()


def test_greeting_with_request_answers_after_short_greeting():
    b = Brain()
    r = b.caller_turn(reading("Hi, how much is an online store", intent="store_price", greeting=0.97), calm)
    assert r.kind == "answer:store_price"
    assert r.sentences[0] in R.GREET_BACK.values()
    assert "Shopify" in " ".join(r.sentences)


def test_request_alone_skips_greeting():
    b = Brain()
    r = b.caller_turn(reading("How much is an online store", intent="store_price"), calm)
    assert r.sentences[0] in R.ACK.values()


def test_second_greeting_is_not_answered_twice():
    b = Brain()
    b.caller_turn(reading("Hello", act="greeting", greeting=0.99), calm)
    r = b.caller_turn(reading("Hi, I need SEO", intent="seo", greeting=0.9), calm)
    assert r.sentences[0] not in R.GREET_BACK.values()


def test_how_are_you_is_answered():
    b = Brain()
    r = b.caller_turn(reading("Good morning, how are you", act="greeting", greeting=0.99), calm)
    assert any(s in R.HOW_ARE_YOU.values() for s in r.sentences)


def test_interrupted_reply_resumes_where_it_was_cut():
    b = Brain()
    r = b.caller_turn(reading("What does the care plan include", intent="care_plan"), calm)
    b.heard(r, 2, interrupted=True)  # opener + first sentence heard, cut in the second
    stop = b.caller_turn(reading("Hold on", act="stop", request=0.0), calm)
    assert stop.kind == "yield"
    b.heard(stop, len(stop.sentences), interrupted=False)  # "Sure, go ahead" played in full
    go = b.caller_turn(reading("Sorry, go on", act="continue", request=0.0), calm)
    assert go.kind == "resume" and go.sentences[1:] == r.sentences[2:]


def test_yeah_answers_our_question():
    b = Brain()
    r = b.caller_turn(reading("I want to book a consultation", intent="book_consultation"), calm)
    assert r.sentences[-1].endswith("?")
    b.heard(r, len(r.sentences), interrupted=False)
    yes = b.caller_turn(reading("yeah", act="backchannel", request=0.0), calm)
    assert yes.kind == "affirmed"


def test_low_confidence_intent_asks_to_clarify_then_answers_on_yes():
    b = Brain()
    r = b.caller_turn(reading("the thing with the site", intent="site_problem", intent_p=0.3), calm)
    assert r.kind == "clarify" and "problem with your site" in r.sentences[-1]
    yes = b.caller_turn(reading("yes", act="affirm", request=0.0), calm)
    assert yes.kind == "answer:site_problem"


def test_frustrated_caller_gets_steady_not_mirrored():
    b = Brain()
    r = b.caller_turn(reading("Hello, my site is down and nobody answers", intent="site_problem", greeting=0.9,
                              neg=0.9), excited)
    assert r.plan.register == "steady" and r.plan.target.arousal < 0
    assert r.sentences[0] in R.GREET_BACK.values() and r.sentences[1] in R.EMPATHY.values()
    thanks = b.caller_turn(reading("Okay thank you", act="thanks", request=0.0, pos=0.9), excited)
    assert thanks.plan.register != "bright"  # one thanks doesn't undo a complaint


def test_happy_caller_is_met_and_lifted():
    b = Brain()
    r = b.caller_turn(reading("I'm so excited, I want a new site", intent="website_price", pos=0.9), excited)
    assert r.plan.register == "bright"
    assert r.plan.target.valence > b.mood.valence and abs(r.plan.target.arousal - b.mood.arousal) < 0.15


# ---- from the first live call (2026-11-03, runs/live-1)

def said(b, text, act, **kw):
    r = b.caller_turn(reading(text, act=act, request=kw.pop("request", 1.0 if act == "request" else 0.0), **kw), calm)
    b.heard(r, len(r.sentences), interrupted=False)
    return r


def test_hello_is_said_once_per_call():
    b = Brain()
    first = said(b, "Hey", "greeting", greeting=0.99)
    again = said(b, "Hey, how are you?", "greeting", greeting=0.99)
    later = said(b, "Hi there", "greeting", greeting=0.99)
    assert first.sentences[0] in R.GREET_BACK.values()
    assert again.sentences[0] in R.HOW_ARE_YOU.values() and not set(again.sentences) & set(R.GREET_BACK.values())
    assert later.sentences[0] in R.GREET_AGAIN.values()


def test_date_and_time_come_from_the_clock_whatever_the_act():
    b = Brain()
    day = b.now().strftime("%A")
    assert day in " ".join(said(b, "What is today?", "smalltalk", intent="location", intent_p=0.27).sentences)
    assert day in " ".join(said(b, "What day is today?", "request", intent="location", intent_p=0.65).sentences)
    r = said(b, "What time is it", "smalltalk")
    assert r.kind == "fact:time" and "Florida" in r.sentences[-1]


def test_spoken_clock_words():
    import datetime
    assert R.spoken_date(datetime.datetime(2026, 11, 3)) == "It's Tuesday, November third."
    assert R.spoken_date(datetime.datetime(2026, 12, 31)) == "It's Thursday, December thirty-first."
    assert R.spoken_time(datetime.datetime(2026, 11, 3, 15, 5)) == "It's three oh five in the afternoon here in Florida."
    assert R.spoken_time(datetime.datetime(2026, 11, 3, 9, 42)) == "It's nine forty-two in the morning here in Florida."


def test_name_is_asked_then_remembered():
    b = Brain()
    ask = said(b, "Do you know my name?", "smalltalk")
    assert ask.kind == "ask:name"
    nice = said(b, "Alex", "smalltalk")
    assert nice.kind == "name" and "Alex" in nice.sentences[0]
    assert said(b, "Do you know my name?", "smalltalk").sentences[0] == "You're Alex."


def test_names_only_where_a_name_is_given():
    from duet.brain import name_in
    assert name_in("Hi, this is Mark from Bloom Bakery", False) == "Mark"
    assert name_in("my name is carlos", False) == "Carlos"
    assert name_in("This is ridiculous", False) is None
    assert name_in("I'm very excited", False) is None
    assert name_in("It's Priya", True) == "Priya" and name_in("It's Priya", False) is None


def test_unanswerable_question_is_admitted_and_never_repeated_word_for_word():
    b = Brain()
    one = said(b, "What's the capital of France?", "smalltalk")
    two = said(b, "What's the capital of France?", "smalltalk")
    assert one.kind == "smalltalk:cant" and one.sentences[0] in R.CANT_ANSWER
    assert one.sentences != two.sentences
    assert not any("Ha" in s for s in one.sentences + two.sentences)


def test_a_feeling_is_met_not_thanked():
    b = Brain()
    assert said(b, "I'm very excited", "smalltalk", pos=0.9).sentences[0] in R.GLAD.values()
    assert said(b, "I'm just very busy", "smalltalk", neg=0.7).sentences[0] in R.EMPATHY.values()
    assert said(b, "My pc is just talking", "smalltalk").sentences[0] in R.NOTED.values()
    # the reflex read it as a thank-you on the live call: without a thank-word it is small talk
    assert said(b, "I'm very excited.", "thanks", pos=0.9).sentences[0] in R.GLAD.values()
    assert said(b, "Cheers, appreciate it", "thanks", pos=0.9).kind == "welcome"


def test_bare_okay_gets_room_not_great_question():
    b = Brain()
    said(b, "Hey, how are you?", "greeting", greeting=0.99)
    r = said(b, "Bro", "backchannel")
    assert r.kind == "ack" and r.sentences[0] in R.LISTENING
    r = said(b, "How much is a store", "request", intent="store_price")
    assert said(b, "okay", "backchannel").sentences == [R.GREAT, R.ANYTHING_ELSE]


def test_loud_neutral_words_get_warm_not_bright():
    b = Brain()
    loud = ToneReading(0.75, -0.3, "angry")  # how the head read "How are you?" said loudly
    r = b.caller_turn(reading("How are you?", act="smalltalk"), loud)
    assert r.plan.register == "warm"


def test_talk_to_me_says_what_sona_can_do():
    b = Brain()
    r = said(b, "Speak with me more.", "repeat")
    assert r.kind == "can_do" and r.sentences[-1] in R.CAN_DO


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
