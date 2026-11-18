"""The conductor: DUET listens while it speaks.

Threads:
- **audio** (sound card callback, or the simulator): every 20 ms hands over what the microphone heard and what we
  played in the same 20 ms, so the echo canceller sees the two aligned.
- **ears + turn-taking** (`Conductor._run`): 48 -> 16 kHz, echo cancel (16 ms blocks), VAD (32 ms chunks), and the
  state machine below. While someone talks it transcribes the utterance so far every 320 ms and reads it with the
  text reflex, so it knows early whether the sentence is finished, a greeting, a "mm-hm" or a "wait".
- **mouth** (`Conductor._speak`): renders a reply sentence by sentence into the player; openers come from the cache.

States: listen -> caller (someone is talking) -> think (reply being made) -> speak -> listen. While speaking, the
caller's voice (VAD and the echo canceller's near-end score together, for most of a 224 ms window) moves it to
**overlap**: DUET ducks 12 dB at once, keeps reading the overlap, and either carries on (a backchannel, or a cough)
or stops within 80 ms and treats the overlap as the start of the caller's turn, remembering which sentence was cut.

Endpointing is semantic: after the voice stops DUET answers after 0.25 s for a question or a short closing act, 0.5 s
for a finished statement (people go on after a sentence), 0.55 s if unsure, 0.9 s if it sounds cut off ("I'd like
to ..."). It never ends a turn while the caller is still talking (up to 25 s; past the recogniser's 10.24 s the turn
is transcribed in pieces). If they go on just as DUET answers, the answer is dropped whole (queued, rendering or
playing) and their words join their last turn, read afresh by the brain as it was before that answer. DUET's sound
never plays while it listens to the caller.
"""
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, lfilter

from .aec import N as AEC_N
from .aec import EchoCanceller, chime
from .asr import Asr
from .brain import Brain
from .config import BLOCK, DEVICE_RATE, RATE, TEXT_REFLEX_DIR
from .prosody import Baseline, MoodReader, fix_octaves, track, utterance_features
from .replies import SPOKEN
from .understand import TextReflex
from .vad import Vad

CHUNK = 512  # VAD step, 32 ms
DUCK_DB = -12.0
GAP_S = 0.16  # silence between sentences
MAX_TURN_S = 25.0  # a caller turn ends here even mid-word (the audio ring holds 30 s)
JOIN_S = 1.2  # talking over the first 1.2 s of our reply (not to stop it) continues their last turn...
JOIN_GAP_S = 2.5  # ...if that turn ended at most this long before they went on
BC_PAUSE = (0.38, 0.55)  # a backchannel goes into a pause this long (Truong 2010: >= 400 ms)...
BC_MIN_TURN_S = 1.0  # ...after at least this much of their turn...
BC_GAP_S = 4.0  # ...at least this long after our last one (one every 4-15 s: Reece 2023; Poppe 2011), when the
BC_ANYWAY_S = 10.0  # pitch cue says so; after this long, without it
BC_FINISHED_S = 8.0  # after a finished sentence only in turns longer than this (a long story)
BC_GAIN = 0.55  # ~-5 dB: under the caller, not over them
DEBUG = bool(os.environ.get("DUET_DEBUG"))


class Decimator:
    """48 -> 16 kHz with a 97-tap low-pass, streaming (state carried across blocks)."""

    def __init__(self):
        self.b = firwin(97, 7000, fs=DEVICE_RATE).astype(np.float64)
        self.zi = np.zeros(len(self.b) - 1)

    def __call__(self, x):
        y, self.zi = lfilter(self.b, 1.0, np.asarray(x, np.float64), zi=self.zi)
        return y[::3]


class Player:
    """Mono 48 kHz playback for the audio callback: a queue of pieces, a smooth gain (duck), and a fade-to-stop.
    Records which pieces started and finished, so the conductor knows how far a reply got."""

    def __init__(self):
        self.lock = threading.Lock()
        self.items = deque()  # [audio, meta, position]
        self.gain, self.target = 1.0, 1.0
        self.fading = 0  # samples left in a stop fade
        self.fade_len = 1
        self.events = deque()  # ("start" | "end" | "stopped", meta, time)
        self.samples_played = 0
        self.level = 0.0  # RMS of the last block played
        self.current = None  # meta of the sentence last started

    def enqueue(self, audio, meta=None):
        with self.lock:
            self.items.append([np.asarray(audio, np.float32), meta, 0])

    def duck(self, db=DUCK_DB):
        self.target = 10 ** (db / 20)

    def unduck(self):
        self.target = 1.0

    def stop(self, fade_ms=80):
        with self.lock:
            if self.items and not self.fading:
                self.fade_len = self.fading = int(fade_ms * DEVICE_RATE / 1000)

    def drop_murmur(self):
        """Fade out a backchannel that is playing (20 ms) and drop queued ones, so a reply can start at once."""
        with self.lock:
            keep = deque()
            for item in self.items:
                audio, meta, pos = item
                if meta != "bc":
                    keep.append(item)
                elif pos > 0:  # playing: end it with a short fade
                    k = min(len(audio) - pos, int(0.02 * DEVICE_RATE))
                    item[0] = audio[:pos + k].copy()
                    item[0][pos:pos + k] *= np.linspace(1.0, 0.0, k, dtype=np.float32)
                    keep.append(item)
            self.items = keep

    def flush(self):
        """Drop every queued piece that has not started (nothing of it was heard, so no event)."""
        with self.lock:
            self.items = deque(item for item in self.items if item[2] > 0)

    @property
    def busy(self):
        return bool(self.items)

    def pull(self, n):
        out = np.zeros(n, np.float32)
        with self.lock:
            i = 0
            while i < n and self.items:
                item = self.items[0]
                audio, meta, pos = item
                if pos == 0 and meta is not None:
                    self.events.append(("start", meta, time.perf_counter()))
                    self.current = meta
                take = min(n - i, len(audio) - pos)
                out[i:i + take] = audio[pos:pos + take]
                item[2] = pos + take
                i += take
                if item[2] >= len(audio):
                    self.items.popleft()
                    if meta is not None:
                        self.events.append(("end", meta, time.perf_counter()))
            # gain ramp (~10 ms) toward the duck target
            ramp = self.gain + (self.target - self.gain) * np.minimum(1.0, np.arange(1, n + 1) / 480)
            out *= ramp
            self.gain = float(ramp[-1])
            if self.fading:
                k = min(self.fading, n)
                env = np.zeros(n, np.float32)
                env[:k] = np.linspace(self.fading / self.fade_len, (self.fading - k) / self.fade_len, k)
                out *= env
                self.fading -= k
                if self.fading == 0:  # the sentence last started (the fade may end in a gap between sentences)
                    self.events.append(("stopped", self.current, time.perf_counter()))
                    self.current = None
                    self.items.clear()
                    self.gain = self.target = 1.0
        self.samples_played += n
        self.level = float(np.sqrt(np.mean(out * out))) if n else 0.0  # what the orb page pulses with (orb.py)
        return out


@dataclass
class Event:
    t: float
    kind: str
    text: str


class Conductor:
    def __init__(self, voice, brain=None, log=print, record=False):
        self.voice, self.log_fn = voice, log
        self.brain = brain if brain is not None else Brain()  # the front desk, or friend.FriendBrain
        # each persona reads text with its own reflex (the friend's has mode and feeling heads)
        # the Spanish partners (amiga.py) bring their own ears: Whisper for the words, TurboCTC for the timing
        self.asr = self.brain.make_ears() if hasattr(self.brain, "make_ears") else Asr()
        self.vad = Vad()
        self.reflex = TextReflex(getattr(self.brain, "reflex_dir", TEXT_REFLEX_DIR))
        self.aec = EchoCanceller()
        self.tone = MoodReader()
        self.baseline = Baseline()
        self.player = Player()
        self.inbox = queue.Queue(maxsize=500)
        self.jobs = queue.Queue()
        self.dec_mic, self.dec_ref = Decimator(), Decimator()
        self.clean = np.zeros(RATE * 30, np.float32)  # ring of echo-cancelled audio, 30 s
        self.n_clean = 0  # samples written in total
        self.pending_mic, self.pending_ref, self.pending_clean = (np.zeros(0),) * 3
        self.state = "listen"
        self.events, self.t0 = [], time.perf_counter()
        self.record = record
        self.rec_mic, self.rec_out, self.rec_clean = [], [], []
        self.talk = deque(maxlen=7)
        self.reset_utterance()
        self.cancelled = set()
        self.started = set()  # replies whose first sentence reached the speaker
        self.speaking_reply = None  # Reply being played
        self.heard_upto = {}  # reply id -> sentences fully played
        self.last_play_end = 0.0
        self.turn_end_t = None  # when the caller's last turn ended (for latency)
        self.prev_turn = None  # (start, end) of the caller's last turn, so a continuation can join it
        self.brain_before = None  # the brain as it was before that turn: a joined turn is read afresh from here
        self.reply_started_t = -1.0  # when our latest reply's first sound left the player
        self.superseded = set()  # replies to a turn the caller went on with: nothing about them is remembered
        self.latencies = []
        self.running = False
        self.ended = False
        self.mouth_busy = False
        self.mouth_reply = None  # the reply the mouth is rendering (taken off the queue, maybe not yet heard)
        self._near, self._echo = 0.0, False
        self._traced = self.state
        self.last_bc, self.bc_pause = -10.0, False  # our last backchannel; decided for this pause already
        self.bc_until = -1.0  # our backchannel and its echo last until then
        self.bc_windows = []  # (start, end) of our backchannels, blanked out of the caller's audio
        self.turn_talk = []  # (time, clear speech?) per chunk of the caller's turn
        self.force_end = False  # the caller pressed "answer now"
        self.feedback = []  # moments the caller marked with a key, with what DUET had decided

    # ------------------------------------------------------------------ audio side
    def feed(self, mic48, out48):
        """Called by the audio engine every block (never blocks)."""
        try:
            self.inbox.put_nowait((np.asarray(mic48, np.float32).copy(), np.asarray(out48, np.float32).copy()))
        except queue.Full:
            pass

    # ------------------------------------------------------------------ helpers
    def now(self):
        return self.n_clean / RATE

    def log(self, kind, text):
        e = Event(round(self.now(), 2), kind, text)
        self.events.append(e)
        if self.log_fn:
            self.log_fn(f"[{e.t:6.2f}] {kind:9s} {text}")

    def audio(self, start_s, end_s=None):
        end = self.n_clean if end_s is None else min(int(end_s * RATE), self.n_clean)
        start = max(int(start_s * RATE), end - len(self.clean) + 1, 0)
        idx = np.arange(start, end) % len(self.clean)
        return self.clean[idx]

    def reset_utterance(self):
        self.turn_talk = []
        self.utt_start = None
        self.last_voice = 0.0
        self.last_partial = 0.0
        self.reading = None
        self.start_count = 0
        self.barged_in = False  # this caller turn began by talking over us
        self.prefetched = None  # last_voice of the pause whose audio the ears are already transcribing

    # ------------------------------------------------------------------ lifecycle
    def start(self, opening=True):
        self.prewarm()
        self.running = True
        threading.Thread(target=self._run, daemon=True, name="ears").start()
        threading.Thread(target=self._speak, daemon=True, name="mouth").start()
        self.player.enqueue(chime(), None)
        self.player.enqueue(np.zeros(int(0.3 * DEVICE_RATE), np.float32), None)
        if opening:
            self.say(self.brain.opening())

    def stop(self):
        self.running = False
        self.jobs.put(None)

    def prewarm(self):
        """Render every line a reply can start with, in the register it is said in (disk-cached: minutes the first
        time, then about a second)."""
        t = time.perf_counter()
        lines = self.brain.prewarm_lines()
        for text, delivery in lines:
            self.voice.cached(self.spoken(text), delivery)
        if self.log_fn and time.perf_counter() - t > 1:
            self.log_fn(f"rendered {len(lines)} openers and first lines in {time.perf_counter() - t:.1f} s "
                        f"(cached for next time)")

    @staticmethod
    def spoken(text):
        for k, v in SPOKEN.items():
            text = text.replace(k, v)
        return text

    def say(self, reply):
        self.state = "think" if self.state != "speak" else self.state
        self.jobs.put(reply)

    # ------------------------------------------------------------------ mouth
    def _speak(self):
        while self.running:
            reply = self.jobs.get()
            if reply is None:
                return
            if reply.id in self.cancelled:
                continue
            self.mouth_reply = reply
            self.mouth_busy = True
            self.heard_upto[reply.id] = 0
            # the time, a name, the model's words: never worth a file in the cache
            one_off = reply.kind.startswith(("fact", "name", "friend", "opening", "repeat", "resume"))
            whole = True
            try:
                for i, s in enumerate(self._sentences(reply)):
                    if reply.id in self.cancelled or not self.running:
                        whole = False
                        break
                    text, d = self.spoken(s), reply.plan.delivery
                    audio = (self.voice.cached(text, d) if (i <= 1 and not one_off) or self.voice.has(text, d) else
                             self.voice.synth(text, d, cancelled=lambda: reply.id in self.cancelled))
                    if audio is None or reply.id in self.cancelled:
                        whole = False
                        break
                    self.player.enqueue(audio, (reply, i))
                    self.player.enqueue(np.zeros(int(GAP_S * DEVICE_RATE), np.float32), None)
            except Exception as e:  # the language model failed mid-reply: end with what was said
                self.log("error", f"reply {reply.id}: {type(e).__name__}: {e}")
            if reply.stream is not None and not whole:
                reply.stream.close()  # stops the model writing
            if whole and self.running:
                self.player.enqueue(np.zeros(1, np.float32), (reply, -1))  # the reply's end, once it has played
            self.mouth_busy = False
            self.mouth_reply = None

    @staticmethod
    def _sentences(reply):
        """A reply's fixed sentences, then the model's as it writes them (appended to the reply as they come)."""
        k = 0
        while k < len(reply.sentences):
            yield reply.sentences[k]
            k += 1
        if reply.stream is not None:
            for s in reply.stream:
                reply.sentences.append(s)
                yield s

    # ------------------------------------------------------------------ ears + turn-taking
    def _run(self):
        while self.running:
            try:
                mic48, out48 = self.inbox.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.record:
                self.rec_mic.append(mic48)
                self.rec_out.append(out48)
            self.pending_mic = np.concatenate([self.pending_mic, self.dec_mic(mic48)])
            self.pending_ref = np.concatenate([self.pending_ref, self.dec_ref(out48)])
            while len(self.pending_mic) >= AEC_N and len(self.pending_ref) >= AEC_N:
                f = self.aec.process(self.pending_mic[:AEC_N], self.pending_ref[:AEC_N], adapt=self.state != "caller")
                self.pending_mic, self.pending_ref = self.pending_mic[AEC_N:], self.pending_ref[AEC_N:]
                self.pending_clean = np.concatenate([self.pending_clean, f.out])
                self._near = max(self._near, f.near_end)  # strongest evidence among this VAD chunk's AEC blocks
                self._echo = self._echo or f.echo_possible
                if len(self.pending_clean) >= CHUNK:
                    chunk, self.pending_clean = self.pending_clean[:CHUNK], self.pending_clean[CHUNK:]
                    idx = np.arange(self.n_clean, self.n_clean + CHUNK) % len(self.clean)
                    self.clean[idx] = chunk
                    self.n_clean += CHUNK
                    if self.record:
                        self.rec_clean.append(chunk)
                    self._chunk(self.vad.step(chunk), self._near, self._echo)
                    self._near, self._echo = 0.0, False
            self._player_events()

    def _player_events(self):
        while self.player.events:
            kind, meta, _ = self.player.events.popleft()
            if meta is None or meta == "bc":
                continue
            reply, i = meta
            if kind == "start" and i == -1 and reply.id in self.started:
                continue
            if reply.id not in self.started and (kind != "start" or reply.id in self.cancelled):
                if kind == "start":  # dropped before its first sound but already queued: silence it at once
                    self.player.flush()
                    self.player.stop(fade_ms=10)
                continue
            if kind == "start":
                self.started.add(reply.id)
                self.speaking_reply = reply
                if i == 0:
                    self.reply_started_t = self.now()
                more = " ..." if reply.stream is not None else ""
                if i == 0 and self.turn_end_t is not None:
                    self.latencies.append(self.now() - self.turn_end_t)
                    self.log("duet", f"→ \"{' '.join(reply.sentences)}{more}\"  [{reply.plan.register}, "
                                     f"{self.latencies[-1] * 1000:.0f} ms after the caller stopped]")
                    self.turn_end_t = None
                elif i == 0:
                    self.log("duet", f"→ \"{' '.join(reply.sentences)}{more}\"  [{reply.plan.register}]")
            elif kind == "end" and i >= 0:
                self.heard_upto[reply.id] = i + 1
            elif kind == "end":  # the end marker: the whole reply has played
                if reply.id not in self.superseded:
                    self.brain.heard(reply, len(reply.sentences), interrupted=False)
                if reply.stream is not None:
                    self.log("said", f"\"{' '.join(reply.sentences)}\"  [{reply.kind}]")
                self.speaking_reply = None
                self.last_play_end = self.now()
                if reply.end_call:
                    self.ended = True
            elif kind == "stopped":
                n = self.heard_upto.get(reply.id, 0)
                if reply.id not in self.superseded:
                    self.brain.heard(reply, n, interrupted=True)
                at = reply.sentences[n] if n < len(reply.sentences) else ""
                self.log("cut", f"stopped during sentence {n + 1}/{len(reply.sentences)}: \"{at}\"")
                self.speaking_reply = None
                self.last_play_end = self.now()

    def _chunk(self, p, near_end, echo_possible):
        now = self.now()
        threshold = 6.0 if echo_possible else 3.0
        talk = p > 0.5 and near_end > threshold
        self.vad_p, self.caller_talking = p, talk  # for the orb page (orb.py)
        # weaker evidence that keeps a caller's turn open; while our own sound may be in the mic, the echo canceller
        # must also see more than echo (the first live call's VAD heard DUET's residual voice as the caller)
        voice = talk or (p > 0.35 and (not echo_possible or near_end > 2.5) and now > self.bc_until)
        if self.state == "caller":
            self.turn_talk.append((now, talk))
        self.talk.append(talk)
        speaking = self.speaking_reply is not None  # a reply in progress, even between two sentences

        if self.state in ("listen", "think") and not speaking:
            if (self.state == "think" and self.jobs.empty() and not self.mouth_busy and not self.player.busy
                    and self.turn_end_t is None):
                self.state = "listen"
            self.start_count = self.start_count + 1 if talk else 0
            if self.start_count >= 3:
                joined = False
                if self.state == "think":  # they kept talking: drop the reply being made, keep listening
                    self._cancel_pending()
                    self.player.flush()  # including sentences already handed to the player
                    joined = self._join_previous(now - 0.1, "they went on before we answered")
                self.state = "caller"
                if not joined:
                    self.utt_start = self.utt_start if self.utt_start is not None else max(0.0, now - 0.35)
                self.last_voice = now
        elif speaking and self.state in ("speak", "think", "listen"):
            self.state = "speak"
            if sum(self.talk) >= 5:
                self.state = "overlap"
                self.utt_start = max(0.0, now - 0.3)
                self.overlap_start = now
                self.last_voice = now
                self.player.duck()
                self.log("overlap", f"caller talking over us (near-end {near_end:.0f}x echo): duck {DUCK_DB:.0f} dB")
        elif self.state == "speak" and not speaking:
            self.state = "listen"
            self.start_count = 0

        if self.state == "caller" and speaking and self.player.busy and not self.player.fading:  # never talk over them
            self._cancel_pending()
            self.player.flush()
            self.player.stop(fade_ms=30)
            self.log("hush", f"our audio was playing while they talked: stopped \"{self.speaking_reply.sentences[0]}\"")
        if self.state == "caller":
            self._listen_caller(now, voice)
        elif self.state == "overlap":
            self._listen_overlap(now, p, talk, speaking)
        if DEBUG and self.state != self._traced:
            self.log("state", f"{self._traced} -> {self.state} (vad {p:.2f}, near-end {near_end:.1f}, "
                              f"speaking {speaking}, player {self.player.busy}, mouth {self.mouth_busy})")
            self._traced = self.state

    def _partial(self, now):
        fresh = self.last_partial < self.last_voice and now - self.last_voice >= 0.12  # they just went quiet
        if fresh or now - self.last_partial >= 0.32:
            self.last_partial = now
            text = self.asr(self.audio(max(self.utt_start, now - 10.0)))  # the reading cares about the latest words
            self.reading = self.reflex.read(text) if text else None

    def _listen_caller(self, now, voice):
        if voice:
            self.last_voice = now
            self.bc_pause = False
        self._partial(now)
        silence = now - self.last_voice
        # slow ears (Whisper, ~1 s) start on the turn at the first pause; if it turns out to be the end, the words
        # are (nearly) ready when the endpoint comes
        if hasattr(self.asr, "prefetch") and silence >= 0.3 and self.prefetched != self.last_voice:
            self.prefetched = self.last_voice
            self.asr.prefetch(self.audio(self.utt_start, self.last_voice + 0.1))
        if self.force_end and silence > 0.05:  # they pressed "answer now"
            self.force_end = False
            self._end_turn(now)
            return
        if self.brain.backchannels and not self.bc_pause and BC_PAUSE[0] <= silence <= BC_PAUSE[1]:
            self.bc_pause = True  # one decision per pause
            self._maybe_backchannel(now)
        need = self.brain.endpoint(self.reading, now - self.utt_start)
        # never end a turn while they are talking: long turns are transcribed in pieces (Asr.long); the 25 s limit
        # only protects the 30 s audio ring
        if silence >= need or now - self.utt_start > MAX_TURN_S:
            self._end_turn(now)

    def _listen_overlap(self, now, p, talk, speaking):
        if talk:
            self.last_voice = now
        self._partial(now)
        r = self.reading
        silence = now - self.last_voice
        words = len(r.text.split()) if r else 0
        if r and r.act in ("backchannel", "affirm") and r.act_p > 0.6 and silence >= 0.3 and words <= 3:
            self.player.unduck()
            self.brain.backchannel(r)
            self.log("backchnl", f"\"{r.text}\": they're with us, keep talking")
            self.state = "speak" if speaking else "listen"
            self.reset_utterance()
        elif r and ((r.act in ("stop", "repeat", "deny") and r.act_p > 0.6) or
                    (words >= 3 and r.act not in ("backchannel", "affirm") and r.act_p > 0.5)):
            self._barge_in(now, f"\"{r.text}\" ({r.act} {r.act_p:.2f})", speaking,
                           went_on=r.act not in ("stop", "repeat", "deny"))
        elif silence < 0.3 and now - self.overlap_start > 0.9:
            self._barge_in(now, f"\"{r.text if r else '...'}\" (still talking after 0.9 s)", speaking,
                           went_on=not (r and r.act in ("stop", "repeat", "deny")))
        elif silence > 0.6:
            self.player.unduck()
            self.log("overlap", f"\"{r.text}\": carry on" if r else "no words: carry on")
            self.state = "speak" if speaking else "listen"
            self.reset_utterance()
        elif not speaking and (r is None or words <= 1) and now - self.overlap_start < 0.4:
            # our reply ended a moment into an "overlap" with no words in it: the room's echo of our last word
            self.player.unduck()
            self.log("overlap", "no words as our reply ended: our own echo, ignored")
            self.state = "listen"
            self.reset_utterance()
        elif not speaking:  # our reply ended under them: just keep listening
            self.player.unduck()
            self.state = "caller"

    def _maybe_backchannel(self, now):
        """A listener's "mm-hm": in a pause inside a turn that isn't a question or a short finished one, when the pitch
        just moved or dipped low (docs/research-good-listener.md, rules 1-4), not too often; quietly, so it never takes
        the floor."""
        r = self.reading
        if r is None or now - self.utt_start < BC_MIN_TURN_S or now - self.last_bc < BC_GAP_S:
            return
        if r.text.rstrip().endswith("?") or (r.complete_p > 0.75 and now - self.utt_start < BC_FINISHED_S):
            return  # a question, or a finished sentence in a turn under 8 s: an answer may come next, not a murmur
            # (a murmur there made the reply wait for it to end: ~1 s in the friend-good-news simulation)
        if not (self._pitch_cue() or now - self.last_bc > BC_ANYWAY_S):
            return
        text, delivery = self.brain.backchannel_line(r)
        audio = BC_GAIN * self.voice.cached(self.spoken(text), delivery)
        self.player.enqueue(audio, "bc")
        self.last_bc = now
        self.bc_until = now + len(audio) / DEVICE_RATE + 0.35  # playback delay and the room's tail
        self.bc_windows = [w for w in self.bc_windows if w[1] > now - 30] + [(now, self.bc_until)]
        self.log("listening", f"\"{text}\" in their pause")

    def _pitch_cue(self):
        """The pitch before the pause invites a backchannel: its last 100 ms rose or fell by 30 Hz or more (Truong 2010),
        or at least 110 ms of the last half second sat below this turn's 26th percentile (Ward & Tsukahara 2000)."""
        pcm = self.audio(max(self.utt_start, self.last_voice - 8.0), self.last_voice + 0.05)
        with np.errstate(divide="ignore", invalid="ignore"):
            level, f0, aper, _ = track(pcm)
        voiced = (level > max(level.max() - 35.0, -60.0)) & (aper < 0.25) & (f0 > 70) & (f0 < 420)
        if voiced.sum() < 30:
            return False
        f0 = fix_octaves(f0, voiced)
        last = np.nonzero(voiced)[0]
        last = last[last >= last[-1] - 10]  # the last 100 ms of voicing
        if len(last) >= 4 and abs(float(f0[last[-1]] - f0[last[0]])) >= 30:
            return True
        low = voiced & (f0 <= np.percentile(f0[voiced], 26))
        run = best = 0
        for k in range(max(0, len(low) - 50), len(low)):
            run = run + 1 if low[k] else 0
            best = max(best, run)
        return best >= 11

    # ------------------------------------------------------------------ the caller's keys (live)
    def manual_stop(self):
        """Stop DUET now (a key while it talks)."""
        if self.speaking_reply is not None or self.mouth_busy or not self.jobs.empty():
            self._cancel_pending()
            self.player.flush()
            self.player.stop()
            self.log("key", "stop: you stopped DUET")

    def answer_now(self):
        """Your turn is over: answer now (a key while you talk)."""
        if self.state == "caller":
            self.force_end = True
            self.log("key", "answer now")

    def mark(self, label):
        """Note this moment, with what DUET had decided, for review."""
        b = self.brain
        self.feedback.append({
            "t": round(self.now(), 2), "label": label, "decision": dict(getattr(b, "last_decision", {}) or {}),
            "speaking": " ".join(self.speaking_reply.sentences) if self.speaking_reply else None,
            "last_reply": " ".join(b.last.sentences) if getattr(b, "last", None) else None,
            "mood": str(getattr(b, "mood", ""))})
        self.log("mark", label)

    def _barge_in(self, now, why, speaking, went_on=False):
        # talking over the first moment of our answer, and not to stop or question it: they hadn't finished
        joined = (went_on and speaking and self.overlap_start - self.reply_started_t < JOIN_S
                  and self._join_previous(self.overlap_start, "they went on as we started"))
        self.barged_in = not joined
        if speaking:
            self._cancel_pending()
            self.player.stop()
            self.log("barge-in", f"{why}: stop within 80 ms")
        else:
            self.player.unduck()
        self.state = "caller"
        self.last_voice = now

    def _join_previous(self, resumed, why):
        """The caller went on with their last turn: hear it as one turn, read by the brain as it was before it
        (so a reply they talked over or never heard leaves nothing behind)."""
        if self.prev_turn is None or self.brain_before is None or resumed - self.prev_turn[1] > JOIN_GAP_S:
            return False
        self.utt_start = self.prev_turn[0]
        self.brain.restore(self.brain_before)
        self.brain_before = None
        if self.speaking_reply is not None:
            self.superseded.add(self.speaking_reply.id)
        self.log("join", f"{why}: one turn from {self.utt_start:.2f} s")
        return True

    def _cancel_pending(self):
        while not self.jobs.empty():
            try:
                r = self.jobs.get_nowait()
                if r is not None:
                    self.cancelled.add(r.id)
                    r.cancelled = True
            except queue.Empty:
                break
        for r in (self.speaking_reply, self.mouth_reply):  # playing, and the one still being rendered
            if r is not None:
                self.cancelled.add(r.id)
                r.cancelled = True

    def _end_turn(self, now):
        pcm = self.audio(self.utt_start, self.last_voice + 0.1)
        for a, b in self.bc_windows:  # our "mm-hm" isn't their words ("...this morning. Yeah." in a simulation)
            for t, talking in self.turn_talk:
                if a <= t < b and not talking:
                    i = int((t - self.utt_start) * RATE)
                    pcm[max(i - CHUNK, 0):max(i, 0)] = 0.0
        text = self.asr.long(pcm)
        if len(text.strip()) < 2 or len(pcm) < 0.25 * RATE:
            self.log("noise", f"ignored {len(pcm) / RATE:.1f} s with no words")
            self.state = "listen"
            self.reset_utterance()
            return
        reading = self.reflex.read(text)
        feats = utterance_features(pcm, self.baseline)
        tone = self.tone.read(feats)
        self.baseline.update(feats)
        self.prev_turn = (self.utt_start, self.last_voice)
        self.brain_before = self.brain.snapshot()
        reply = self.brain.caller_turn(reading, tone, interrupted=self.barged_in)
        hz = 100 * 2 ** (feats["f0_st_abs"] / 12) if feats["f0_st_abs"] else 0
        tags = (f"act={reading.act}({reading.act_p:.2f}) greeting={reading.greeting_p:.2f} "
                f"intent={reading.intent}({reading.intent_p:.2f}) "
                f"words={max(reading.sentiment, key=reading.sentiment.get)} finished={reading.complete_p:.2f} | "
                f"tone: {hz:.0f} Hz, arousal {tone.arousal:+.2f}{' ' + tone.label if tone.label else ''} | "
                f"mood {self.brain.mood}")
        self.log("caller", f"\"{text}\"  {tags}")
        self.turn_end_t = self.last_voice
        self.reset_utterance()
        self._cancel_pending()
        self.player.flush()
        self.player.drop_murmur()
        self.state = "think"
        self.say(reply)
