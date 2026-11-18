"""Mouth: Supertonic-3 (OpenRAIL-M) on ONNX Runtime, speaking in our own Ivy-family voices.

A voice is a 0.3 MB style (`style_ttl` [1, 50, 256] + `style_dp` [1, 8, 16]); ours come from the Ivy Voice Builder
(`folio-voices`). Mood reaches the voice three ways, all per sentence:
- **pace**: Supertonic's `speed` divides the predicted duration;
- **lift**: the style moves along measured directions in style space (`tools/calibrate_mood_voice.py` renders every
  style, measures pitch level and pitch range, and fits those against the styles); the builder taught us the style
  space is smooth, so small moves stay the same speaker;
- **level**: a gentle gain.

It runs on the CPU: about 5x faster than real time (3.7 s of speech in 0.75 s, 4 threads, 8 steps). DirectML is 4x
faster per sentence but recompiles for every new latent length, and padding the latent is not exact (the estimator
spreads speech over the padded frames), so it is not used. Instead, sentences stream one at a time and the short
openers a reply starts with are rendered once and cached, so the first sound of a reply is instant.
"""
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly

from .config import DEVICE_RATE, MODELS, MOOD_VOICE, TTS_DIR, VOICES_DIR

OPENER_CACHE = MODELS / "opener-cache"


@dataclass(frozen=True)
class Delivery:
    """How to say one sentence. Built by `mood.respond` from the caller's mood."""
    speed: float = 1.05
    pitch_lift: float = 0.0  # in style-spread units along the measured pitch-level direction
    range_lift: float = 0.0  # ... along the pitch-range (liveliness) direction
    gain_db: float = 0.0
    label: str = "neutral"


class Style:
    def __init__(self, ttl, dp):
        self.ttl, self.dp = ttl.astype(np.float32), dp.astype(np.float32)


def load_style(path):
    d = json.loads(path.read_text())
    ttl = np.array(d["style_ttl"]["data"], np.float32).reshape(d["style_ttl"]["dims"])
    dp = np.array(d["style_dp"]["data"], np.float32).reshape(d["style_dp"]["dims"])
    return Style(ttl, dp)


def _preprocess(text, lang, indexer):
    """Supertone's text normalisation (helper.py), shortened to what our replies contain."""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[–‑—]", "-", text)
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"[\[\]|/#_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[.!?;:,'\")]$", text):
        text += "."
    text = f"<{lang}>{text}</{lang}>"
    ids = np.array([[indexer[ord(c)] if ord(c) < len(indexer) else 0 for c in text]], np.int64)
    return ids, np.ones((1, 1, ids.shape[1]), np.float32)


def sentences(text):
    """Split for streaming: sentences, and long sentences again at a comma or dash near the middle."""
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        while len(s) > 140:
            cut = max(s.rfind(", ", 0, 110), s.rfind(" - ", 0, 110))
            if cut < 40:
                break
            out.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            out.append(s)
    return out


class Voice:
    """`lang` is Supertonic's language tag for every sentence (en, es, de, ...); the Spanish partners use es."""

    def __init__(self, voice="orion", steps=8, threads=4, seed=7, lang="en", lang_for=None):
        self.lang = lang
        self.lang_for = lang_for  # optional sentence -> language tag (a Spanish partner's German explanations)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        onnx = TTS_DIR / "onnx"
        load = lambda n: ort.InferenceSession(str(onnx / f"{n}.onnx"), so, providers=["CPUExecutionProvider"])  # noqa: E731
        self.dp, self.enc, self.est, self.voc = (load(n) for n in
                                                 ("duration_predictor", "text_encoder", "vector_estimator", "vocoder"))
        cfg = json.loads((onnx / "tts.json").read_text())
        self.sample_rate = cfg["ae"]["sample_rate"]
        self.chunk = cfg["ae"]["base_chunk_size"] * cfg["ttl"]["chunk_compress_factor"]
        self.ldim = cfg["ttl"]["latent_dim"] * cfg["ttl"]["chunk_compress_factor"]
        self.indexer = json.loads((onnx / "unicode_indexer.json").read_text())
        self.styles = {p.stem.lower(): load_style(p) for p in sorted((TTS_DIR / "voice_styles").glob("*.json"))}
        self.styles.update({p.stem.lower(): load_style(p) for p in sorted(VOICES_DIR.glob("*.json"))})
        self.steps, self.seed = steps, seed
        self.directions = dict(np.load(MOOD_VOICE)) if MOOD_VOICE.exists() else {}
        self._cache = {}
        self.set_voice(voice)
        self.last_ms = 0.0

    def set_voice(self, name):
        self.voice = name.lower()
        self.base = self.styles[self.voice]

    def styled(self, delivery):
        """The base voice moved along the measured mood directions (if calibrated)."""
        ttl, dp = self.base.ttl, self.base.dp
        for key, amount in (("pitch", delivery.pitch_lift), ("range", delivery.range_lift)):
            if amount and f"{key}_ttl" in self.directions:
                ttl = ttl + amount * self.directions[f"{key}_ttl"]
                dp = dp + amount * self.directions[f"{key}_dp"]
        return Style(ttl, dp)

    def _key(self, text, delivery):
        # English keeps its old key, so the openers already on disk stay valid
        return (self.voice, text, delivery) if self.lang == "en" else (self.voice, text, delivery, self.lang)

    def cached(self, text, delivery=None):
        """`synth`, remembered in memory and on disk (`models/opener-cache`): for the lines a reply starts with."""
        delivery = delivery or Delivery()
        key = self._key(text, delivery)
        if key not in self._cache:
            sig = hashlib.sha1(repr((key, self.steps, self.seed, sorted(self.directions))).encode()).hexdigest()[:16]
            path = OPENER_CACHE / self.voice / f"{sig}.npy"
            if path.exists():
                self._cache[key] = np.load(path)
            else:
                self._cache[key] = self.synth(text, delivery)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self._cache[key])
        self.last_ms = 0.0
        return self._cache[key]

    def has(self, text, delivery=None):
        return self._key(text, delivery or Delivery()) in self._cache

    def synth(self, text, delivery=None, lang=None, out_rate=DEVICE_RATE, cancelled=None):
        """One sentence -> float32 mono at `out_rate`, trimmed, with 8 ms fades. Same text + delivery = same audio.
        `cancelled()` is checked between the flow-matching steps; when it says so, returns None at once."""
        delivery = delivery or Delivery()
        lang = lang or (self.lang_for(text) if self.lang_for else None) or self.lang
        t0 = time.perf_counter()
        style = self.styled(delivery)
        ids, mask = _preprocess(text, lang, self.indexer)
        dur = self.dp.run(None, {"text_ids": ids, "style_dp": style.dp, "text_mask": mask})[0] / delivery.speed
        emb = self.enc.run(None, {"text_ids": ids, "style_ttl": style.ttl, "text_mask": mask})[0]
        wav_len = int(dur[0] * self.sample_rate)
        lat_len = -(-wav_len // self.chunk)
        rng = np.random.default_rng([self.seed, len(text)])
        xt = rng.standard_normal((1, self.ldim, lat_len)).astype(np.float32)
        lat_mask = np.ones((1, 1, lat_len), np.float32)
        total = np.array([self.steps], np.float32)
        for step in range(self.steps):
            if cancelled is not None and cancelled():
                return None
            xt = self.est.run(None, {"noisy_latent": xt, "text_emb": emb, "style_ttl": style.ttl, "text_mask": mask,
                                     "latent_mask": lat_mask, "current_step": np.array([step], np.float32),
                                     "total_step": total})[0]
        wav = self.voc.run(None, {"latent": xt})[0][0, :wav_len].astype(np.float32)
        wav = _trim(wav, self.sample_rate)
        if out_rate != self.sample_rate:
            g = np.gcd(out_rate, self.sample_rate)
            wav = resample_poly(wav, out_rate // g, self.sample_rate // g).astype(np.float32)
        wav *= 10 ** (delivery.gain_db / 20)
        fade = min(int(0.008 * out_rate), len(wav) // 2)
        wav[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        wav[len(wav) - fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        self.last_ms = 1000 * (time.perf_counter() - t0)
        return np.clip(wav, -1, 1)


def _trim(wav, sr, floor_db=-45.0):
    """Cut leading/trailing near-silence, keeping 40 ms before and 80 ms after the speech."""
    hop = sr // 100
    n = len(wav) // hop
    if n < 3:
        return wav
    rms = np.sqrt(np.mean(wav[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    loud = np.nonzero(20 * np.log10(rms / (rms.max() + 1e-12)) > floor_db)[0]
    if len(loud) == 0:
        return wav
    a = max(0, (loud[0] - 4) * hop)
    b = min(len(wav), (loud[-1] + 9) * hop)
    return wav[a:b]
