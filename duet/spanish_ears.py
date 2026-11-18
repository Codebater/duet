"""Ears for the Spanish partners (amiga.py): Whisper large-v3-turbo for whole turns, TurboCTC for the timing.

TurboCTC (asr.py) is English only. It stays for the partial transcripts DUET reads while someone talks (every 0.32 s,
~35 ms each): those only time the turn and spot a barge-in, so English guesses at Spanish words are good enough.
The words a reply is written from come from Whisper (multilingual, `ggml-large-v3-turbo-q5_0.bin`, already on this
laptop for captions) through ffmpeg 8.1's built-in whisper.cpp filter, which runs it on the GPU.

`WhisperServer` keeps ONE ffmpeg running with the model loaded and feeds it raw audio on stdin, one chunk of
`QUEUE_S` seconds per turn (the turn, then silence): ~0.5-0.6 s a turn (measured 2026-11-13). Starting ffmpeg per
turn costs ~1.4-1.9 s (model load + GPU setup), and pywhispercpp's wheel is CPU-only: 6-12 s a turn on this laptop.
The per-turn ffmpeg stays as the fallback, and for the rare second pass below.

Language: `auto`. Forcing `es` turns a German question into made-up Spanish ("Wie sagt man ... ich vermisse dich?"
came back as "¿Cómo se dice en español puesa de ti?"), and a learner asks in German now and then. If auto lands on
a third language (accented Spanish read as Portuguese or Italian), the turn is transcribed again as Spanish.
"""
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

from .asr import Asr
from .config import RATE

WHISPER_MODEL = Path(os.environ.get("DUET_WHISPER_MODEL",
                                    Path.home() / ".cache" / "whisper" / "ggml-large-v3-turbo-q5_0.bin"))
# seconds of audio per turn sent to the server. A multiple of 64 ms (the raw demuxer reads 1024-sample packets and
# would hold back a partial one), at most Whisper's 30 s window; longer turns go as several chunks
QUEUE_S = 24
SETTLE_S = 0.05  # a chunk's segments are written in one burst: this long without a new line = done
NO_WORDS_S = 2.5  # no line at all by then: nothing was said (or the GPU is very busy)

# what Whisper writes over silence, music or a cough (its subtitle training data); never a turn
HALLUCINATIONS = re.compile(
    r"amara\.org|subt[ií]tul|gracias por ver|suscr[ií]b|untertitel|vielen dank f[üu]rs zuschauen|"
    r"^\W*(m[úu]sica|music|aplausos|risas)\W*$", re.I)
GERMAN = re.compile(r"\b(ich|du|nicht|das|ist|und|wie|sagt|man|auf spanisch|warte|bitte|noch|einmal|verstehe|"
                    r"heißt|habe|kannst|was|wir|mein|dein|nochmal|langsamer)\b", re.I)
SPANISH = re.compile(r"\b(yo|tú|vos|que|qué|el|la|los|las|es|sí|no|y|de|en|un|una|me|te|mi|muy|pero|con|por|"
                     r"para|como|cómo|está|estás|tengo|tenés|tienes|hola|gracias|bien)\b|[ñ¿¡áéíóú]", re.I)
OTHER = re.compile(r"\b(você|não|obrigad[oa]|eu|sono|perché|grazie|anche|questo|ciao|então)\b|ção\b|ções\b", re.I)
# Whisper sometimes hears German as English and writes it in English ("How does Spanish say I miss you?", from a
# simulated "Wie sagt man auf Spanisch, ich vermisse dich?"): the meaning survives, so it's kept, as a German turn
ENGLISH = re.compile(r"\b(the|you|how|what|does|say|is|are|i|i'm|my|and|in|of|to|it|this|that|do|can)\b", re.I)
# the learner's name as Whisper spells it in Spanish (a Slavic name heard as "André"): pairs (regex, name) from his
# about.json (amiga.LEARNER), so the partner doesn't call him by the misspelling
NAMES = []
SEGMENT = re.compile(r'^\{"start":(\d+),"end":(\d+),"text":"(.*)"\}\s*$')  # its JSON doesn't escape quotes
# the English he really speaks: questions about the Spanish. Other "English" from a short turn was German misheard
# ("Was? Was? Noch mal auf Deutsch?" came out as "Buzz? Buzz? No more of touch."), so it is heard again as German
# (in his real turns a German-only pass gave "Was? Was? Noch mal auf Deutsch." and left real English as English)
ENGLISH_META = re.compile(r"\bhow (do|can|would) i say\b|\bhow to say\b|\bwhat (does|do|is)\b.*\bmean|\bwhat means\b|"
                          r"\bwhat (this|that|it) means\b|\bask me\b|\bcan we\b|\b(start|begin) (again|over|from)\b|"
                          r"\brepeat\b|\bslower\b|\bi don'?t (understand|get it)\b|\btranslate\b", re.I)


def language_of(text):
    """'de', 'en', 'es' or 'other' from the words alone."""
    de, es, en = len(GERMAN.findall(text)), len(SPANISH.findall(text)), len(ENGLISH.findall(text))
    if (OTHER.search(text) and es <= 1) or not re.search(r"[A-Za-zÀ-ÿ]", text):  # "это": Cyrillic for a mumble
        return "other"
    if en > max(es, de):
        return "en"
    return "de" if de > es else "es"


def _filter_path(path):
    # ffmpeg filter options are separated by ':', so the drive letter's colon is escaped twice (graph + option)
    return Path(path).as_posix().replace(":", "\\\\:")


def _clean(texts, names=()):
    text = " ".join(s.strip() for s in texts if s.strip() and not HALLUCINATIONS.search(s)).strip()
    for pattern, name in [*NAMES, *names]:
        text = pattern.sub(name, text)
    return text


def _pcm16(pcm):
    return (np.clip(np.asarray(pcm, np.float32), -1, 1) * 32767).astype(np.int16).tobytes()


def _key(pcm):
    return len(pcm), round(float(np.abs(pcm[::101]).sum()), 4)


class WhisperServer:
    """One ffmpeg, model loaded once on the GPU; `transcribe(pcm16k) -> text`, one turn at a time."""

    def __init__(self, model=WHISPER_MODEL, language="auto", queue_s=QUEUE_S):
        self.queue_ms, self.chunk = 1000 * queue_s, int(queue_s * RATE)
        flt = (f"whisper=model={_filter_path(model)}:language={language}:queue={queue_s}:"
               "destination=pipe\\\\:1:format=json")
        self.p = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-fflags", "nobuffer", "-probesize", "32",
             "-analyzeduration", "0", "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0", "-af", flt,
             "-f", "null", os.devnull],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.sent = 0  # chunks written so far
        self.segments, self.last_line = {}, {}  # chunk -> [(start_ms, end_ms, text)], chunk -> time of its last line
        self.cv = threading.Condition()
        self.lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True, name="whisper-out").start()
        self._write(np.zeros(self.chunk, np.float32))  # loads the model now, while the greeting plays

    @property
    def alive(self):
        return self.p.poll() is None

    def _write(self, pcm):
        n = max(1, -(-len(pcm) // self.chunk))
        buf = np.zeros(n * self.chunk, np.float32)
        buf[:len(pcm)] = pcm
        self.p.stdin.write(_pcm16(buf))
        self.p.stdin.flush()
        first, self.sent = self.sent, self.sent + n
        return first, n

    def _read(self):
        for raw in self.p.stdout:
            m = SEGMENT.match(raw.decode("utf-8", "replace").strip())
            if not m:
                continue
            start, end = int(m.group(1)), int(m.group(2))
            i = int(start // self.queue_ms)
            with self.cv:
                self.segments.setdefault(i, []).append((start - i * self.queue_ms, end - i * self.queue_ms,
                                                        m.group(3)))
                self.last_line[i] = time.perf_counter()
                self.cv.notify_all()

    def transcribe(self, pcm):
        pcm = np.asarray(pcm, np.float32)
        with self.lock:
            first, n = self._write(pcm)
            last, t0 = first + n - 1, time.perf_counter()
            with self.cv:
                while True:
                    now, seen = time.perf_counter(), self.last_line.get(last)
                    if (seen is not None and now - seen >= SETTLE_S) or now - t0 > NO_WORDS_S * n or not self.alive:
                        break
                    self.cv.wait(timeout=0.01)
                segs = [(k, s) for k in range(first, last + 1) for s in self.segments.pop(k, [])]
                for k in [k for k in self.segments if k < first]:  # late lines of a turn that timed out
                    self.segments.pop(k, None)
                self.last_line = {k: v for k, v in self.last_line.items() if k > last}
        real_ms = 1000 * len(pcm) / RATE
        # only what lies on the real audio: over the padding Whisper invents "Thank you." across the whole window
        keep = [t for k, (s, e, t) in segs
                if (k - first) * self.queue_ms + s < real_ms + 300 and e - s < real_ms + 2000]
        return _clean(keep)

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            self.p.kill()


class SpanishEars(Asr):
    """`long(pcm16k) -> text` through Whisper; `__call__` (the partials) stays TurboCTC. `last_lang` = de|en|es.

    `prefetch(pcm)` starts Whisper on a turn in the background (the conductor calls it at the caller's first pause);
    `long` on the same audio then waits for that result instead of starting again. One prefetch at a time: the
    server works turn by turn, so a stale one (they went on talking) only delays the next by ~0.5 s."""

    def __init__(self, model=WHISPER_MODEL, directml=True, server=True, name_fixes=()):
        if not Path(model).exists():
            raise RuntimeError(f"Whisper model not found: {model} (set DUET_WHISPER_MODEL)")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg (8.1 or newer, with the whisper filter) is not on PATH")
        super().__init__(directml)
        self.model = Path(model)
        self.tmp = Path(tempfile.mkdtemp(prefix="duet-whisper-"))
        self.server = WhisperServer(self.model) if server else None
        # a second model copy that only hears German (~0.6 GB more on the GPU): for turns auto heard as odd English
        self.server_de = WhisperServer(self.model, language="de") if server else None
        self.names = [(re.compile(p, re.I), n) for p, n in name_fixes]
        self.last_lang = "es"
        self._n = 0
        self._ahead = None  # (key, thread, result holder)
        self.timeline = []  # (start, seconds, audio seconds, kind): what Whisper cost in this call
        self.reused = False
        self.lock = threading.Lock()

    def prefetch(self, pcm):
        with self.lock:
            if self._ahead and self._ahead[1].is_alive():
                return
            pcm = np.array(pcm, np.float32)
            holder = {}
            t = threading.Thread(target=lambda: holder.update(r=self._transcribe(pcm, "prefetch")), daemon=True,
                                 name="whisper")
            self._ahead = (_key(pcm), t, holder)
        t.start()

    def whisper(self, pcm, language="auto", kind="turn"):
        t0 = time.perf_counter()
        if language == "auto" and self.server is not None and self.server.alive:
            text = self.server.transcribe(pcm)
        elif language == "de" and self.server_de is not None and self.server_de.alive:
            text = self.server_de.transcribe(pcm)
        else:
            text = self._once(pcm, language)
        for pattern, name in self.names:
            text = pattern.sub(name, text)
        self.timeline.append((round(t0, 3), round(time.perf_counter() - t0, 3), len(pcm) / RATE, kind))
        return text

    def _once(self, pcm, language):
        """One ffmpeg for one turn (fallback, and the Spanish-only second pass): ~1.4 s."""
        with self.lock:
            self._n += 1
            n = self._n
        wav, out = self.tmp / f"turn{n}.wav", self.tmp / f"turn{n}.txt"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(_pcm16(pcm))
        flt = (f"whisper=model={_filter_path(self.model)}:language={language}:queue=30:"
               f"destination={_filter_path(out)}:format=text")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(wav), "-af", flt,
                        "-f", "null", "-"], capture_output=True, timeout=30,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines = out.read_text(encoding="utf-8", errors="replace").splitlines() if out.exists() else []
        for f in (wav, out):
            f.unlink(missing_ok=True)
        return _clean(lines)

    def _transcribe(self, pcm, kind="turn"):
        text = self.whisper(pcm, "auto", kind)
        lang = language_of(text) if text else "es"
        if lang == "other" and re.search(r"\w{2}", text):  # not for a lone "." (noise)
            text = self.whisper(pcm, "es", kind + " (es)")
        elif lang == "en" and not ENGLISH_META.search(text):
            de = self.whisper(pcm, "de", kind + " (de)")
            if de and language_of(de) == "de":
                text = de
        return text, language_of(text) if text else "es"

    def long(self, pcm, piece_s=None, min_s=None):
        pcm = np.asarray(pcm, np.float32)
        with self.lock:
            ahead, self._ahead = self._ahead, None
        if ahead and ahead[0] == _key(pcm):
            ahead[1].join(timeout=30)
            text, lang = ahead[2].get("r", ("", "es"))
            self.reused = True
        else:
            text, lang = self._transcribe(pcm)
            self.reused = False
        self.last_lang = lang
        return text

    def close(self):
        for s in (self.server, self.server_de):
            if s is not None:
                s.close()
