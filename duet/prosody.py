"""Ear 4, tone: how something is said, measured on the user's own voice, then read as mood.

Per 10 ms frame (32 ms window, 16 kHz): RMS level, a YIN pitch estimate (70-420 Hz) with its aperiodicity, and the
share of energy above 2 kHz (spectral tilt: a tense voice is flatter). Per utterance, over voiced frames, the 12
numbers in `FEATURES`. Pitch and level are taken relative to the speaker's own baseline (`Baseline`, a slow average
over their earlier turns) because "high" only means something against how this person usually sounds.

`MoodReader` turns them into arousal (calm -1 .. +1 excited). With a trained head (`tools/train_prosody_head.py`,
`models/prosody-head/mood_head.npz`) it uses that; without one it uses fixed weights from the speech-emotion
literature (higher, wider, louder, faster, brighter = more aroused). Valence comes mostly from the words
(`understand.py`), so this module reports only a small valence hint.
"""
from dataclasses import dataclass, field

import numpy as np

from .config import PROSODY_HEAD, RATE

HOP, WIN = RATE // 100, 512  # 10 ms hop, 32 ms window
F0_MIN, F0_MAX = 70.0, 420.0
FEATURES = ("f0_st", "f0_range_st", "f0_slope_end", "f0_jitter", "level_db", "level_range_db", "voiced_share",
            "rate_syl", "hf_ratio", "pause_share", "f0_st_abs", "duration_s")


def frames(pcm):
    """[n, WIN] overlapping frames, 10 ms apart."""
    pcm = np.asarray(pcm, np.float32)
    if len(pcm) < WIN:
        pcm = np.pad(pcm, (0, WIN - len(pcm)))
    n = 1 + (len(pcm) - WIN) // HOP
    idx = np.arange(n)[:, None] * HOP + np.arange(WIN)[None, :]
    return pcm[idx]


def track(pcm):
    """Per-frame (level dB, f0 Hz or 0, aperiodicity, hf share)."""
    x = frames(pcm)
    x = x - x.mean(axis=1, keepdims=True)
    level = 10 * np.log10(np.mean(x ** 2, axis=1) + 1e-10)
    # YIN difference function via FFT autocorrelation
    lag_min, lag_max = int(RATE / F0_MAX), int(RATE / F0_MIN)
    w = WIN - lag_max
    spec_head = np.fft.rfft(x[:, :w], 2 * WIN, axis=1)
    spec = np.fft.rfft(x, 2 * WIN, axis=1)
    acf = np.fft.irfft(np.conj(spec_head) * spec, axis=1)[:, :lag_max + 1]  # sum_j x[j] x[j+tau], j < w
    energy = np.cumsum(np.concatenate([np.zeros((len(x), 1)), x ** 2], axis=1), axis=1)
    e0 = energy[:, w][:, None]  # energy of x[0:w]
    lags = np.arange(lag_max + 1)
    e_lag = energy[:, lags + w] - energy[:, lags]  # energy of x[tau:tau+w]
    d = e0 + e_lag - 2 * acf  # YIN difference function
    d[:, 0] = 0
    cum = np.cumsum(d[:, 1:], axis=1) / np.arange(1, lag_max + 1)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] / (cum + 1e-10)
    search = dn[:, lag_min:lag_max]
    below = search < 0.15
    first = np.where(below.any(axis=1), below.argmax(axis=1), search.argmin(axis=1))
    tau = first + lag_min
    # walk to the local minimum after the first dip
    for _ in range(4):
        nxt = np.minimum(tau + 1, lag_max - 1)
        tau = np.where(dn[np.arange(len(x)), nxt] < dn[np.arange(len(x)), tau], nxt, tau)
    aper = dn[np.arange(len(x)), tau]
    # parabolic interpolation
    a, b, c = (dn[np.arange(len(x)), np.clip(tau + k, 1, lag_max)] for k in (-1, 0, 1))
    shift = np.where(np.abs(a - 2 * b + c) > 1e-9, 0.5 * (a - c) / (a - 2 * b + c), 0.0)
    f0 = RATE / (tau + np.clip(shift, -1, 1))
    mag = np.abs(np.fft.rfft(x * np.hanning(WIN), axis=1)) ** 2
    hf = mag[:, int(2000 * WIN / RATE):].sum(axis=1) / (mag.sum(axis=1) + 1e-10)
    return level, f0, aper, hf


OCTAVE_MOVES = np.array([0.0, -12.0, -24.0, 12.0])  # keep, halve, quarter, double (semitones)
OCTAVE_COST = np.array([0.0, 1.0, 2.0, 2.0])  # per frame; octave-up errors are the common ones, so halving is cheapest


def fix_octaves(f0, voiced, max_gap=3):
    """Undo octave jumps: the tracker locking onto the second (or fourth) harmonic when the fundamental is weak (laptop
    and phone microphones), or onto a double period. Each voiced frame keeps its pitch or moves by whole octaves,
    whichever path through the utterance has the smallest pitch steps between voiced neighbours (frames at most
    `max_gap` apart; a longer pause lets pitch reset). A move costs OCTAVE_COST semitones per frame, so a real glide is
    never folded. Measured on the first live call: 85 octave jumps to 1; CREMA-D actors' pitch unchanged or steadier."""
    idx = np.nonzero(voiced)[0]
    if len(idx) < 3:
        return f0
    k_moves = len(OCTAVE_MOVES)
    st = 12 * np.log2(f0[idx] / 100.0)
    cand = st[:, None] + OCTAVE_MOVES[None, :]
    lo, hi = 12 * np.log2(F0_MIN / 100.0), 12 * np.log2(F0_MAX / 100.0)
    step = np.where((cand >= lo) & (cand <= hi), OCTAVE_COST[None, :], np.inf)
    step[:, 0] = 0.0
    cost, back = step[0].copy(), np.zeros((len(idx), k_moves), int)
    for k in range(1, len(idx)):
        jump = (np.abs(cand[k][None, :] - cand[k - 1][:, None]) if idx[k] - idx[k - 1] <= max_gap
                else np.zeros((k_moves, k_moves)))
        total = cost[:, None] + jump
        back[k] = total.argmin(axis=0)
        cost = total.min(axis=0) + step[k]
    choice = np.zeros(len(idx), int)
    choice[-1] = int(cost.argmin())
    for k in range(len(idx) - 1, 0, -1):
        choice[k - 1] = back[k, choice[k]]
    out = np.array(f0, dtype=np.float64)
    out[idx] = 100.0 * 2 ** (cand[np.arange(len(idx)), choice] / 12)
    return out


def utterance_features(pcm, baseline=None):
    """One utterance (16 kHz) -> dict of FEATURES. Pitch/level relative to `baseline` when given."""
    level, f0, aper, hf = track(pcm)
    loud = level > max(level.max() - 35.0, -60.0)
    voiced = loud & (aper < 0.25) & (f0 > F0_MIN) & (f0 < F0_MAX)
    f0 = fix_octaves(f0, voiced)
    dur = len(pcm) / RATE
    out = dict.fromkeys(FEATURES, 0.0)
    out["duration_s"] = dur
    out["voiced_share"] = float(voiced.mean()) if len(voiced) else 0.0
    # pauses: quiet frames inside the speech (first to last frame clearly above the noise floor), so the silence a VAD
    # or a recording leaves around the words does not count; "clearly above" is relative to this turn's own floor
    floor = float(np.percentile(level, 10)) if len(level) else -100.0
    speech = level > max(floor + 10.0, level.max() - 40.0, -70.0) if len(level) else np.zeros(0, bool)
    span = np.nonzero(speech)[0]
    inside = speech[span[0]:span[-1] + 1] if len(span) else speech
    out["pause_share"] = float(1 - inside.mean()) if len(inside) else 1.0
    if voiced.sum() < 5:
        return out
    st = 12 * np.log2(f0[voiced] / 100.0)  # semitones re 100 Hz
    med = float(np.median(st))
    # reject octave jumps against the utterance median
    st = st[np.abs(st - med) < 7]
    if len(st) < 5:
        return out
    lv = level[voiced]
    out["f0_st_abs"] = float(np.median(st))
    out["f0_st"] = out["f0_st_abs"] - (baseline.f0_st if baseline and baseline.n else _prior(out["f0_st_abs"]))
    out["f0_range_st"] = float(np.percentile(st, 90) - np.percentile(st, 10))
    out["f0_jitter"] = float(np.median(np.abs(np.diff(st)))) if len(st) > 2 else 0.0
    vi = np.nonzero(voiced)[0]
    tail = vi[vi >= vi[-1] - 50]  # last 500 ms of voicing
    if len(tail) >= 6:
        ts = 12 * np.log2(f0[tail] / 100.0)
        keep = np.abs(ts - med) < 7
        if keep.sum() >= 6:  # on the frames' real time axis (voiced frames are not contiguous)
            out["f0_slope_end"] = float(np.polyfit(tail[keep] / 100.0, ts[keep], 1)[0])  # semitones per second
    out["level_db"] = float(np.mean(lv)) - (baseline.level_db if baseline and baseline.n else -26.0)
    out["level_range_db"] = float(np.percentile(lv, 90) - np.percentile(lv, 10))
    onsets = np.sum(np.diff(voiced.astype(np.int8)) == 1) + int(voiced[0])
    out["rate_syl"] = float(onsets / max(inside.sum() / 100.0, 0.3))  # voicing onsets per second of sounding speech
    out["hf_ratio"] = float(np.mean(hf[voiced]))
    return out


def _prior(f0_st_abs):
    """Without a baseline: an adult low or high voice's typical median (110 / 200 Hz), whichever is nearer."""
    low, high = 12 * np.log2(110 / 100), 12 * np.log2(200 / 100)
    return low if abs(f0_st_abs - low) < abs(f0_st_abs - high) else high


@dataclass
class Baseline:
    """How this caller usually sounds: slow averages of their earlier turns."""
    f0_st: float = 0.0
    level_db: float = 0.0
    rate_syl: float = 0.0
    n: int = 0
    first: list = field(default_factory=list)  # (f0, level, rate) of the first turns: the start is their median

    START = 3

    def update(self, feats, weight=0.3):
        if feats["voiced_share"] < 0.1:
            return
        now = (feats["f0_st_abs"], feats["level_db"] + (self.level_db if self.n else -26.0), feats["rate_syl"])
        if self.n < self.START:  # one odd first turn (a shout, a pitch glitch) must not set the baseline
            self.first.append(now)
            self.f0_st, self.level_db, self.rate_syl = (float(v) for v in np.median(self.first, axis=0))
        else:
            self.f0_st += weight * (now[0] - self.f0_st)
            self.level_db += weight * (now[1] - self.level_db)
            self.rate_syl += weight * (now[2] - self.rate_syl)
        self.n += 1


@dataclass
class ToneReading:
    arousal: float  # -1 calm .. +1 excited
    valence_hint: float  # small; words carry valence
    label: str  # the head's class, or "" for the rule path
    feats: dict = field(default_factory=dict)


class MoodReader:
    """Prosody features -> arousal. A trained 12->16->k head if present, else literature weights."""

    RULE = {"f0_st": 0.10, "f0_range_st": 0.06, "level_db": 0.08, "level_range_db": 0.02, "rate_syl": 0.25,
            "hf_ratio": 4.0, "f0_jitter": 0.0}
    RULE_CENTER = {"f0_range_st": 5.0, "rate_syl": 4.0, "hf_ratio": 0.12, "level_range_db": 10.0}

    def __init__(self, path=PROSODY_HEAD):
        self.head = dict(np.load(path, allow_pickle=False)) if path.exists() else None
        self.labels = [str(s) for s in self.head["labels"]] if self.head is not None else []

    def read(self, feats):
        if feats["voiced_share"] < 0.05:
            return ToneReading(0.0, 0.0, "", feats)
        if self.head is not None:
            h = self.head
            x = (np.array([feats[k] for k in FEATURES], np.float32) - h["mean"]) / h["std"]
            z = np.maximum(x @ h["w1"] + h["b1"], 0) @ h["w2"] + h["b2"]
            p = np.exp(z - z.max())
            p /= p.sum()
            arousal = float(p @ h["arousal"])
            valence = float(p @ h["valence"])
            return ToneReading(float(np.clip(arousal, -1, 1)), float(np.clip(valence, -1, 1)),
                               self.labels[int(p.argmax())], feats)
        s = sum(w * (feats[k] - self.RULE_CENTER.get(k, 0.0)) for k, w in self.RULE.items())
        return ToneReading(float(np.tanh(s)), 0.0, "", feats)
