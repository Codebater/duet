"""A random telephone line, for training the ear on phone audio without copying the test calls' one fixed channel.

The 310 test calls went through one setting (`frontdesk-bench/bench/augment.py`: 300-3400 Hz, mu-law, pink noise at
12-22 dB SNR). Training draws every part at random instead: the room at the caller's end, the band edges and filter
slope, the level, mu-law / A-law / no codec, and white, pink or brown noise before and after the codec at 5-30 dB SNR.
"""
import numpy as np
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt


def rms(x):
    return np.sqrt(np.mean(x ** 2)) + 1e-9


def coloured(n, rng, slope):
    """Noise with power ~ 1/f^slope (0 white, 1 pink, 2 brown)."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1
    return np.fft.irfft(spec / f ** (slope / 2), n)


def add_noise(x, rng, snr_db):
    noise = coloured(len(x), rng, rng.choice([0, 1, 2]))
    return x + noise * rms(x) / rms(noise) * 10 ** (-snr_db / 20)


def room(x, rng, rate=16_000):
    """A small room: exponentially decaying noise impulse response, RT60 0.15-0.6 s, mixed with the dry path."""
    rt60 = rng.uniform(0.15, 0.6)
    t = np.arange(int(rt60 * rate)) / rate
    ir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0 / rng.uniform(0.3, 1.0)  # direct path, louder when the phone is close
    y = fftconvolve(x, ir)[:len(x)]
    return y * rms(x) / rms(y)


def companding(x, law, mu=255, a=87.6):
    """8-bit G.711 mu-law or A-law: compress, quantise to 256 levels, expand."""
    x = np.clip(x, -1, 1)
    s, ax = np.sign(x), np.abs(x)
    if law == "mulaw":
        y = np.log1p(mu * ax) / np.log1p(mu)
    else:
        y = np.where(ax < 1 / a, a * ax / (1 + np.log(a)), (1 + np.log(np.maximum(a * ax, 1e-12))) / (1 + np.log(a)))
    y = np.round(y * 127) / 127
    if law == "mulaw":
        ax = ((1 + mu) ** y - 1) / mu
    else:
        ax = np.where(y < 1 / (1 + np.log(a)), y * (1 + np.log(a)) / a, np.exp(y * (1 + np.log(a)) - 1) / a)
    return s * ax


def phone(pcm, rng):
    """16 kHz float audio (even length) -> the same through a random phone line, 16 kHz, same length."""
    n = len(pcm)
    x = pcm.astype(np.float64)
    if rng.random() < 0.3:
        x = room(x, rng)
    if rng.random() < 0.4:
        x = add_noise(x, rng, rng.uniform(10, 30))  # the caller's surroundings
    x = resample_poly(x, 1, 2)
    lo, hi = rng.uniform(150, 450), rng.uniform(2900, 3800)
    x = sosfilt(butter(int(rng.integers(2, 7)), [lo, hi], btype="band", fs=8000, output="sos"), x)
    x = x * 10 ** (rng.uniform(-32, -12) / 20) / rms(x)
    law = rng.choice(["mulaw", "alaw", "none"], p=[0.4, 0.3, 0.3])
    if law != "none":
        x = companding(x, law)
    if rng.random() < 0.7:
        x = add_noise(x, rng, rng.uniform(5, 30))  # the line
    x = resample_poly(x, 2, 1)[:n]
    return np.clip(np.pad(x, (0, n - len(x))), -1, 1).astype(np.float32)


def laptop(pcm, rng, echo=None):
    """16 kHz float audio (even length) -> the same as DUET hears it on a laptop, same length: often a room (the
    microphone is an arm's length away), a small microphone's band, the surroundings, and what the echo canceller
    leaves of DUET's own voice. `echo`: another clip of DUET's voice (16 kHz), placed anywhere, often across the first
    words, 15-30 dB under the speech and band-limited as a canceller's residual is."""
    n = len(pcm)
    x = pcm.astype(np.float64)
    if rng.random() < 0.7:
        x = room(x, rng)
    lo, hi = rng.uniform(80, 250), rng.uniform(5000, 7800)
    x = sosfilt(butter(int(rng.integers(1, 4)), [lo, hi], btype="band", fs=16000, output="sos"), x)
    if echo is not None and rng.random() < 0.6:
        e = sosfilt(butter(2, [rng.uniform(150, 400), rng.uniform(3000, 5000)], btype="band", fs=16000,
                           output="sos"), echo.astype(np.float64))
        at = int(rng.integers(-len(e) // 2, max(1, n - len(e) // 4)))
        placed = np.zeros(n)
        a, b = max(0, at), min(n, at + len(e))
        if b > a:
            placed[a:b] = e[a - at:b - at]
            x = x + placed * rms(x) / rms(placed[a:b]) * 10 ** (-rng.uniform(15, 30) / 20)
    if rng.random() < 0.8:
        x = add_noise(x, rng, rng.uniform(10, 35))  # a fan, the room, the street
    x = x * 10 ** (rng.uniform(-45, -18) / 20) / rms(x)  # DUET's real calls: -46 to -19 dBFS per turn
    return np.clip(x, -1, 1).astype(np.float32)
