"""DUET from the command line.

    python -m duet live [--persona friend|frontdesk|novia|cita] [--remember] [--voice orion] [--record DIR]
    python -m duet text [--persona friend|novia|cita] [--remember]   type instead: see every decision and its reply

`novia` and `cita` are the Spanish practice partners (amiga.py): Luz, a girlfriend, and Sofía, a woman you just met.
    python -m duet sim [script] [--out DIR]          a scripted caller on a simulated laptop
    python -m duet devices

`live` records every call to runs/live-<date-time>/ (audio, log, and the moments you marked with keys).
"""
import argparse
import datetime
import sys
import time
from pathlib import Path

from .voice import Voice

ROOT = Path(__file__).resolve().parent.parent

# keys during a live call: each mark is saved with what DUET had decided at that moment (tools/review_call.py)
KEYS = {
    "g": "good", "b": "bad", "t": "wrong tone or energy", "l": "should have listened", "c": "should have talked back",
    "x": "missed the context", "m": "misheard me", "i": "interrupted me", "s": "too slow", "n": "note",
}
LEGEND = ("keys: SPACE stop DUET (while it talks) / answer now (while you talk) | g good  b bad  t tone  "
          "l should listen  c should talk back  x missed context  m misheard  i interrupted  s slow  n note | q end")


PERSONAS = ["friend", "frontdesk", "novia", "cita"]


def make_brain(persona, memory, typed=False, level=1):
    if persona == "frontdesk":
        from .brain import Brain
        return Brain()
    if persona in ("novia", "cita"):
        from .amiga import AmigaBrain, pick_model
        from .llm import Llm
        from .amiga import unload
        from .llm import MODEL
        model = pick_model()
        print(f"{persona}: words from {model} (loading...)")
        try:
            if model != MODEL:
                unload(MODEL)  # with the friend's 4B still loaded, Gemma 12B took minutes to load, or failed
            llm = Llm(model=model, load_timeout=300)
        except Exception as e:  # noqa: BLE001 - the bigger model didn't load: the small one still talks
            if model == MODEL:
                raise
            print(f"  {model} didn't load ({type(e).__name__}: {e}); using {MODEL}")
            llm = Llm(model=MODEL)
        # typed: a tip after every sentence; spoken: all tips after the call (keeps the replies fast)
        return AmigaBrain(llm, persona, live_tips=typed, level=level)
    from .friend import MEMORY_PATH, FriendBrain
    from .llm import Llm
    # every call starts fresh (a universal friend) unless a memory is asked for: --remember, or --memory FILE
    path = MEMORY_PATH if memory == "remember" else Path(memory) if memory and memory != "none" else None
    return FriendBrain(Llm(), memory_path=path)


def start_orb(brain, port, open_page=True):
    """The orb page for a Spanish partner (orb.py): her words, his, the lesson; opened in the browser."""
    from .orb import Orb
    try:
        orb = Orb(brain=brain, port=port)
    except OSError as e:
        print(f"orb page not started (port {port}: {e})")
        return None
    brain.tip = orb.tip
    print(f"orb page: {orb.url}")
    if open_page:
        import webbrowser
        webbrowser.open(orb.url)
    return orb


def keys():
    """Keys pressed since the last call (Windows console), lower-cased."""
    try:
        import msvcrt
    except ImportError:
        return []
    out = []
    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # arrows and function keys: two codes, ignored
            msvcrt.getwch()
            continue
        out.append(ch.lower())
    return out


def text_chat(brain):
    """Type to the friend. Shows the decisions the reflexes made and the reply as the model writes it."""
    from .prosody import ToneReading
    from .understand import TextReflex
    reflex = TextReflex(brain.reflex_dir)
    calm = ToneReading(0.0, 0.0, "")
    print("type to DUET (/mem shows its memory, /quit ends)\n" if hasattr(brain, "memory") else
          "type in Spanish (or German, to ask how to say something); /quit ends\n")
    r = brain.opening()
    r.sentences.extend(r.stream or [])
    print("duet>", " ".join(r.sentences))
    brain.heard(r, len(r.sentences), False)
    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("/quit", "/q"):
            break
        if line == "/mem" and hasattr(brain, "memory"):
            print(f"  name {brain.memory.name}; facts: {[f['text'] for f in brain.memory.facts]}")
            continue
        if not line:
            continue
        t0 = time.perf_counter()
        reading = reflex.read(line)
        reply = brain.caller_turn(reading, calm)
        d = getattr(brain, "last_decision", {})
        print(f"  [{d.get('mode')}/{d.get('move')} | {d.get('register')} | act {reading.act} | mood {brain.mood}]")
        print("duet>", " ".join(reply.sentences), end=" ", flush=True)
        first = None
        for s in (reply.stream or []):
            first = first or time.perf_counter() - t0
            reply.sentences.append(s)
            print(s, end=" ", flush=True)
        print(f"\n  ({1000 * (first or 0):.0f} ms to the first written sentence)")
        brain.heard(reply, len(reply.sentences), False)
        if reply.end_call:
            break
    if hasattr(brain, "close"):
        brain.close()


def main():
    # the call log prints arrows and quotes; on Windows a redirected stdout is cp1252, and one unprintable character
    # in the ears thread would stop DUET listening
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(prog="duet")
    sub = ap.add_subparsers(dest="cmd", required=True)
    live = sub.add_parser("live", help="talk to DUET through this laptop's microphone and speakers")
    live.add_argument("--persona", choices=PERSONAS, default="friend")
    live.add_argument("--voice", help="default: orion, or the Spanish partner's own voice")
    live.add_argument("--input", type=int)
    live.add_argument("--output", type=int)
    live.add_argument("--record", type=Path, help="default: runs/live-<date-time>")
    live.add_argument("--no-record", action="store_true")
    live.add_argument("--remember", action="store_const", const="remember", dest="memory",
                      help="the friend remembers you across calls (data/friend/memory.json); default: a fresh call")
    live.add_argument("--memory", help="a memory file of its own")
    live.add_argument("--minutes", type=float, default=15.0)
    live.add_argument("--level", type=int, choices=[1, 2], default=1,
                      help="Spanish partners: 1 = beginner (A1, very short and slow), 2 = A2")
    live.add_argument("--no-orb", action="store_true", help="Spanish partners: no orb page (orb.py)")
    live.add_argument("--orb-port", type=int, default=4330)
    text = sub.add_parser("text", help="type to the friend")
    text.add_argument("--persona", choices=["friend", "novia", "cita"], default="friend")
    text.add_argument("--level", type=int, choices=[1, 2], default=1)
    text.add_argument("--remember", action="store_const", const="remember", dest="memory")
    text.add_argument("--memory")
    sim = sub.add_parser("sim", help="a scripted caller on a simulated laptop (echo and all)")
    sim.add_argument("script", nargs="?", default="front-desk")
    sim.add_argument("--voice")
    sim.add_argument("--orb", action="store_true", help="show the call on the orb page (not opened for you)")
    sim.add_argument("--orb-port", type=int, default=4330)
    sim.add_argument("--out", type=Path)
    sub.add_parser("devices", help="list sound devices")
    a = ap.parse_args()

    if a.cmd == "devices":
        import sounddevice as sd
        print(sd.query_devices())
        from .audio import wasapi_defaults
        print("WASAPI defaults (in, out):", wasapi_defaults())
        return
    if a.cmd == "text":
        text_chat(make_brain(a.persona, a.memory, typed=True, level=a.level))
        return
    t = time.perf_counter()
    if a.cmd == "sim":
        from .sim import run_script
        spanish = a.script.split("-")[0] if a.script.startswith(("novia-", "cita-")) else None
        brain = (make_brain(spanish, "none") if spanish else
                 make_brain("friend", "none") if a.script.startswith("friend") else None)
        voice = Voice(a.voice or getattr(brain, "voice_name", "orion"), lang="es" if spanish else "en",
                      lang_for=getattr(brain, "sentence_lang", None))
        print(f"loading... voice {voice.voice}")
        orb = start_orb(brain, a.orb_port, open_page=False) if a.orb and brain is not None else None
        c, s = run_script(a.script, voice, a.out, caller_voice=Voice(lang="es" if spanish else "en"), brain=brain,
                          log=orb.log if orb else print, setup=(lambda c: setattr(orb, "c", c)) if orb else None)
        lat = [round(1000 * x) for x in c.latencies]
        if hasattr(brain, "tips"):  # the Spanish partners: let the coach finish and save its tips
            brain.close()
            if brain.ears is not None:
                t0 = c.t0
                for start, secs, audio_s, kind in brain.ears.timeline:
                    print(f"  whisper {kind:8s} at {start - t0:6.2f} s: {secs:.2f} s for {audio_s:.1f} s of audio")
        print(f"\nreply latencies (caller stops -> DUET's first sound): {lat} ms")
        if a.out:
            print(f"wrote {a.out}")
        return
    from .audio import LiveEngine
    from .duplex import Conductor
    record = None if a.no_record else a.record or ROOT / "runs" / f"live-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    brain = make_brain(a.persona, a.memory, level=a.level)
    voice = Voice(a.voice or getattr(brain, "voice_name", "orion"), lang=getattr(brain, "lang", "en"),
                  lang_for=getattr(brain, "sentence_lang", None))
    orb = start_orb(brain, a.orb_port) if hasattr(brain, "view") and not a.no_orb else None
    c = Conductor(voice, brain, log=orb.log if orb else print, record=record is not None)
    if orb:
        orb.c = c
    engine = LiveEngine(c, a.input, a.output)
    print(f"loaded in {time.perf_counter() - t:.1f} s. {engine.open()}")
    print(f"Speak after the chime. {LEGEND}")
    print("Click this terminal and press n once: a 'mark note' line shows the keys reach DUET.\n")
    c.start()
    try:
        end = time.time() + 60 * a.minutes
        while time.time() < end and not (c.ended and not c.player.busy):
            for k in keys():
                if k == " ":
                    if c.speaking_reply is not None or c.player.busy:
                        c.manual_stop()
                    elif c.state == "caller":
                        c.answer_now()
                    else:
                        c.log("key", "SPACE (nothing to stop, you aren't talking)")
                elif k == "q":
                    end = 0
                elif k in KEYS:
                    c.mark(KEYS[k])
                else:
                    c.log("key", f"{k!r} isn't a DUET key")
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    time.sleep(0.5)
    c.stop()
    engine.close()
    if hasattr(c.brain, "close"):
        print("saving what DUET remembers...")
        c.brain.close()
    if record:
        from .sim import SimEngine
        SimEngine.save(type("S", (), {"c": c, "log": []})(), record)
        print(f"wrote {record} ({len(c.feedback)} marked moments)")


if __name__ == "__main__":
    main()
