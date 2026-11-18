"""A virtual laptop for testing DUET without a person: what DUET plays goes through a simulated room (55 ms delay,
0.3 s reverb, a little speaker distortion) back into its own microphone, mixed with a scripted caller and a noise
floor. It runs on a real 20 ms clock, so the conductor's threads behave exactly as they do live.

A caller line either waits for DUET to finish ("after", a pause after DUET goes quiet) or talks over DUET
("over", a delay after DUET starts its next reply), which is how barge-in and backchannels are tested.
"""
import json
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from .config import BLOCK, DEVICE_RATE
from .voice import Delivery, Voice


@dataclass
class Line:
    text: str
    voice: str = "m1"
    speed: float = 1.0
    when: str = "after"  # after | over
    pause: float = 0.7  # after: seconds of DUET silence first
    delay: float = 1.5  # over: seconds into DUET's reply
    level: float = 0.5
    more: object = ""  # said after `gap` seconds of silence, as part of the same line (a pause to think); a
    gap: float = 0.0  # tuple of strings is several pieces, each after the gap
    lang: str = None  # Supertonic's language for this line (default: the caller voice's own)


SCRIPTS = {
    # the Spanish partners (python -m duet sim novia-... / cita-...): a learner's slips, a German question, a pause
    # to find a word, an "ajá" over her reply, a goodbye
    "novia-charla": [
        Line("Hola, mi amor. Estoy bien, pero un poco cansado.", "m1", 0.95),
        Line("Hoy yo trabajo mucho en la computadora.", "m1", 0.95, more="Y después fui al gimnasio.", gap=0.6),
        Line("Wie sagt man auf Spanisch, ich vermisse dich?", "m1", 1.0, lang="de"),
        Line("Te extraño mucho, mi amor.", "m1", 0.95),
        Line("What means this?", "m1", 1.0, lang="en"),  # his words on the first live call
        Line("¿Qué significa extrañar?", "m1", 0.95),
        Line("ajá", "m1", 1.0, when="over", delay=0.8),
        Line("Bueno, chau mi amor, hasta mañana.", "m1", 0.95),
    ],
    "cita-cafe": [
        Line("Sí, claro, está libre. Me llamo Alex. ¿Y vos?", "m2", 0.95),
        Line("Yo soy de Europa, pero vivo aquí desde un año.", "m2", 0.95),
        Line("Ayer yo voy a la Costanera, es muy linda.", "m2", 0.95),
        Line("¿Querés tomar un tereré conmigo el sábado?", "m2", 0.95),
        Line("Chau, nos vemos el sábado.", "m2", 0.95),
    ],
    "front-desk": [
        Line("Hi there!", "m1", 1.05),
        Line("I'd love a quote for an online store for my bakery, how much would that be?", "m1", 1.1),
        Line("mm-hm", "m1", 1.0, when="over", delay=2.2),
        Line("Wait, sorry, how long does that take?", "m1", 1.1, when="over", delay=2.5),
        Line("Perfect, thanks so much!", "m1", 1.1),
        Line("Bye!", "m1", 1.05),
    ],
    "frustrated": [
        Line("Hello, my website has been down since this morning and nobody answers my emails. This is really "
             "frustrating.", "f2", 1.15, level=0.6),
        Line("Can I talk to a real person?", "f2", 1.15, level=0.6),
        Line("Okay. Thank you.", "f2", 0.95),
    ],
    "greet-vs-intent": [
        Line("Good morning, how are you?", "m3", 1.0),
        Line("Do you guys do Google rankings?", "m3", 1.05),
        Line("Hold on, hold on.", "m3", 1.1, when="over", delay=1.6),
        Line("Sorry, go on.", "m3", 1.0),
        Line("Thanks, bye.", "m3", 1.0),
    ],
    # the first live call's failure: a turn longer than the recogniser's 10.24 s, and a short pause before more
    "long-talker": [
        Line("Hello.", "m2", 1.0),
        Line("So I'm just very busy at the moment, this is my problem. We are only two people and we work really "
             "efficiently, but there is never enough time for the website, so we keep putting it off, and now our "
             "old site is falling apart.", "m2", 1.0,
             more="And customers keep telling us they can't find our opening hours. How long would a new one take?",
             gap=0.4),
        Line("Okay, thank you, bye.", "m2", 1.0),
    ],
    # the friend persona (python -m duet sim friend-...): good news met with its energy, a mm-hm over the reply kept
    "friend-good-news": [
        Line("Hey, guess what!", "f1", 1.05),
        Line("I got the job at the design studio in Miami! They called me this morning.", "f1", 1.1),
        Line("I start in two weeks.", "f1", 1.05, more=("And my sister Anna is going to help me move.",
                                                         "I still can't believe it's actually happening."), gap=0.35),
        Line("mm-hm", "f1", 1.0, when="over", delay=0.9),
        Line("Thanks! Okay, I have to run. Bye!", "f1", 1.1),
    ],
    # a vent with pauses: DUET should murmur in the pauses, reflect, not advise until asked
    "friend-vent": [
        Line("Hi.", "m3", 0.95),
        Line("I had a really rough day.", "m3", 0.95, more=(
            "My best friend and I had a big fight this morning,", "and she said some things that really hurt.",
            "I don't even know if we're still friends."), gap=0.4),
        Line("Yeah. I just keep thinking about it.", "m3", 0.95),
        Line("What would you do?", "m3", 0.95),
        Line("Thanks for listening. I'll talk to you later, bye.", "m3", 0.95),
    ],
}


def room(rate=DEVICE_RATE, delay_ms=55, rt60=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n = int(0.35 * rate)
    t = np.arange(n) / rate
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60) * 0.08 * np.sqrt(16_000 / rate)
    h[0] = 0.6
    return np.concatenate([np.zeros(int(delay_ms * rate / 1000)), h]).astype(np.float32)


class SimEngine:
    def __init__(self, conductor, lines, caller_voice=None, noise_db=-55.0, echo=True):
        self.c, self.lines = conductor, list(lines)
        self.ir = room() if echo else None
        self.noise = 10 ** (noise_db / 20)
        cv = caller_voice or Voice()
        self.audio = []
        for line in self.lines:  # render the caller first, so it costs nothing during the call
            cv.set_voice(line.voice)
            a = cv.synth(line.text, Delivery(speed=line.speed), lang=line.lang)
            for piece in ([line.more] if isinstance(line.more, str) else list(line.more)):
                if piece:
                    a = np.concatenate([a, np.zeros(int(line.gap * DEVICE_RATE), np.float32),
                                        cv.synth(piece, Delivery(speed=line.speed), lang=line.lang)])
            self.audio.append(line.level / 0.5 * 0.5 * a)
        self.hist = np.zeros(len(self.ir) + BLOCK if echo else BLOCK, np.float32)
        self.log = []
        self.rng = np.random.default_rng(2)

    def run(self, max_s=90.0, idle_end_s=3.0):
        c = self.c
        step, playing, pos = 0, None, 0
        duet_on, duet_since, duet_off_since, spoke_since_line = False, 0.0, 0.0, False
        t0 = time.perf_counter()
        tick = 0
        while tick * BLOCK / DEVICE_RATE < max_s:
            t = tick * BLOCK / DEVICE_RATE
            out = c.player.pull(BLOCK)
            speaking = c.speaking_reply is not None
            if speaking and not duet_on:
                duet_on, duet_since, spoke_since_line = True, t, True
            elif not speaking and duet_on:
                duet_on, duet_off_since = False, t
            mic = np.zeros(BLOCK, np.float32)
            if self.ir is not None:
                self.hist = np.concatenate([self.hist[BLOCK:], out])
                echo = fftconvolve(self.hist, self.ir, mode="valid")[-BLOCK:]
                mic += 0.8 * np.tanh(echo / 0.8)
            if playing is None and step < len(self.lines):
                line = self.lines[step]
                ready = (line.when == "after" and not duet_on and spoke_since_line and c.state == "listen"
                         and t - duet_off_since >= line.pause) or \
                        (line.when == "over" and duet_on and t - duet_since >= line.delay)
                if ready:
                    playing, pos = self.audio[step], 0
                    pieces = [line.more] if isinstance(line.more, str) else list(line.more)
                    said = f" ({line.gap:.1f} s pause) ".join([line.text, *[p for p in pieces if p]])
                    self.log.append({"t": round(t, 2), "caller": said, "when": line.when,
                                     "seconds": round(len(self.audio[step]) / DEVICE_RATE, 2)})
                    print(f"[{t:6.2f}] (sim)     caller says \"{said}\" ({len(self.audio[step]) / DEVICE_RATE:.1f} s)"
                          + (" over DUET" if line.when == "over" else ""))
                    step += 1
                    spoke_since_line = False
            if playing is not None:
                take = min(BLOCK, len(playing) - pos)
                mic[:take] += playing[pos:pos + take]
                pos += take
                if pos >= len(playing):
                    playing = None
            mic += self.noise * self.rng.standard_normal(BLOCK).astype(np.float32)
            c.feed(mic, out)
            tick += 1
            if c.ended and not c.player.busy:
                break
            if step >= len(self.lines) and playing is None and not duet_on and not c.player.busy \
                    and c.state == "listen" and t - duet_off_since > idle_end_s:
                break
            wait = t0 + tick * BLOCK / DEVICE_RATE - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        time.sleep(0.5)

    def save(self, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        c = self.c
        if c.rec_mic:
            stereo = np.stack([np.concatenate(c.rec_mic), np.concatenate(c.rec_out)], axis=1)
            sf.write(out_dir / "call_mic_L_duet_R.wav", stereo, DEVICE_RATE)
            sf.write(out_dir / "heard_after_echo_cancel.wav", np.concatenate(c.rec_clean), 16_000)
        json.dump({"caller": self.log, "events": [asdict(e) for e in c.events],
                   "latencies_ms": [round(1000 * x) for x in c.latencies],
                   "feedback": getattr(c, "feedback", [])},
                  open(out_dir / "call.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def run_script(name, voice, out_dir=None, caller_voice=None, brain=None, log=print, setup=None):
    """Scripts named friend-* are for the friend persona (pass its brain); the others for the front desk.
    `setup(conductor)` runs before the call starts (the orb page hooks in there)."""
    from .duplex import Conductor
    c = Conductor(voice, brain, log=log, record=out_dir is not None)
    if setup:
        setup(c)
    sim = SimEngine(c, SCRIPTS[name], caller_voice)
    c.start()
    runner = threading.Thread(target=sim.run, daemon=True)
    runner.start()
    runner.join()
    c.stop()
    if out_dir is not None:
        sim.save(out_dir)
    return c, sim
