"""TurboCTC's front end (`duet/asr.py features`, a port of frontdesk-bench's features.rs), and the Ear's causal variant.

16 kHz -> 80 HTK log-mels (n_fft 512, win 400, hop 160), floored 8 below a maximum, `/4 + 1`, deltas, two 10 ms frames
stacked per 20 ms row: [rows, 320]. TurboCTC floors against the whole clip's maximum, which a stream doesn't know yet;
`causal=True` floors each frame against the maximum so far, so the Ear trains on exactly what it hears live.
"""
import numpy as np

RATE = 16_000
N_FFT, WIN, HOP, N_MELS, FLOOR = 512, 400, 160, 80, 8.0


def _htk_mel_filterbank():
    nyq = RATE / 2
    hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)  # noqa: E731
    mel_to_hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)  # noqa: E731
    f_pts = mel_to_hz(hz_to_mel(nyq) * np.arange(N_MELS + 2) / (N_MELS + 1))
    f = nyq * np.arange(N_FFT // 2 + 1) / (N_FFT // 2)
    down = (f[None, :] - f_pts[:-2, None]) / (f_pts[1:-1] - f_pts[:-2])[:, None]
    up = (f_pts[2:, None] - f[None, :]) / (f_pts[2:] - f_pts[1:-1])[:, None]
    return np.maximum(0.0, np.minimum(down, up)).astype(np.float32)  # [80, 257]


_MEL = _htk_mel_filterbank()
_WINDOW = np.zeros(N_FFT, np.float32)
_WINDOW[(N_FFT - WIN) // 2:(N_FFT - WIN) // 2 + WIN] = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)


def features(pcm, causal=False):
    """16 kHz float mono -> [rows, 320] (two 10 ms frames per row)."""
    mel_frames = len(pcm) // HOP
    frames = 2 * -(-mel_frames // 2)
    if frames == 0 or len(pcm) <= N_FFT // 2:
        return np.zeros((0, 320), np.float32)
    x = np.asarray(pcm, np.float32)
    needed = (frames - 1) * HOP + 1
    if len(x) < needed:
        x = np.pad(x, (0, needed - len(x)))
    x = np.pad(x, N_FFT // 2, mode="reflect")
    idx = np.arange(frames)[:, None] * HOP + np.arange(N_FFT)[None, :]
    spec = np.fft.rfft(x[idx] * _WINDOW, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    logmel = np.log10(np.maximum(power @ _MEL.T, 1e-10))
    top = np.maximum.accumulate(logmel.max(axis=1))[:, None] if causal else logmel.max()
    logmel = np.maximum(logmel, top - FLOOR) / 4.0 + 1.0
    prev = logmel[np.maximum(np.arange(frames) - 1, 0)]
    nxt = logmel[np.minimum(np.arange(frames) + 1, frames - 1)]
    rows = np.concatenate([logmel, (nxt - prev) / 2.0], axis=1)  # [frames, 160]
    return rows.reshape(frames // 2, 320).astype(np.float32)
