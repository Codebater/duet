"""Ear 3, words: Granite Speech 5.0 TurboCTC (470M, Apache-2.0) on ONNX Runtime + DirectML.

The front end is `frontdesk-bench/engine-rs/src/features.rs` in numpy: torchaudio's MelSpectrogram (16 kHz, n_fft 512,
win 400, hop 160, 80 HTK mels, power 2), log10 floored 8 below the clip maximum, `/4 + 1`, deltas, frame pairs stacked
to 320 values per 20 ms row. The graph is the voice reflex's static one: 512 rows (10.24 s), so longer audio keeps its
last 10.24 s. Greedy CTC, token 0 = blank. English only.
"""
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .config import ASR_DIR, RATE

N_FFT, WIN, HOP, N_MELS, FLOOR, ROWS = 512, 400, 160, 80, 8.0, 512


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


def features(pcm):
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
    logmel = np.maximum(logmel, logmel.max() - FLOOR) / 4.0 + 1.0
    prev = logmel[np.maximum(np.arange(frames) - 1, 0)]
    nxt = logmel[np.minimum(np.arange(frames) + 1, frames - 1)]
    rows = np.concatenate([logmel, (nxt - prev) / 2.0], axis=1)  # [frames, 160]
    return rows.reshape(frames // 2, 320).astype(np.float32)


class Asr:
    """`Asr()(pcm16k) -> text`. One session; calls take ~35 ms on the laptop's Radeon (DirectML, fp16)."""

    def __init__(self, directml=True):
        so = ort.SessionOptions()
        so.add_session_config_entry("ep.dml.disable_graph_fusion", "1")  # DML fusion miscomputes this graph
        so.log_severity_level = 3
        providers = (["DmlExecutionProvider"] if directml else []) + ["CPUExecutionProvider"]
        name = "turboctc_fp16.onnx" if directml else "turboctc_fp32.onnx"
        self.session = ort.InferenceSession(str(ASR_DIR / name), so, providers=providers)
        self.tok = Tokenizer.from_file(str(ASR_DIR / "tokenizer.json"))
        self.last_ms = 0.0
        self(np.zeros(RATE // 2, np.float32))  # warm-up: DirectML compiles on the first run

    def __call__(self, pcm):
        t0 = time.perf_counter()
        feats = features(pcm)
        if len(feats) == 0:
            return ""
        n = min(len(feats), ROWS)
        x = np.zeros((1, ROWS, 320), np.float32)
        x[0, :n] = feats[len(feats) - n:]
        mask = np.zeros((1, ROWS), np.int64)
        mask[0, :n] = 1
        logits = self.session.run(["logits"], {"input_features": x, "attention_mask": mask})[0][0]
        valid = -(-n // 4)  # the encoder subsamples rows 4x
        ids = logits[:valid].argmax(-1)
        keep = [int(t) for j, t in enumerate(ids) if t != 0 and (j == 0 or t != ids[j - 1])]
        text = self.tok.decode(keep, skip_special_tokens=True).strip()
        self.last_ms = 1000 * (time.perf_counter() - t0)
        return text

    def long(self, pcm, piece_s=10.0, min_s=6.0):
        """Any length: one pass per piece of at most `piece_s`, each cut at the quietest 100 ms between `min_s` and
        `piece_s` into it (a pause, nearly always), so a long turn is transcribed whole instead of its last 10.24 s."""
        pcm = np.asarray(pcm, np.float32)
        limit, step = int(piece_s * RATE), RATE // 10
        parts, s = [], 0
        while len(pcm) - s > limit:
            lo = s + int(min_s * RATE)
            seg = pcm[lo:s + limit]
            n = len(seg) // step
            quiet = int(np.argmin((seg[:n * step].reshape(n, step) ** 2).mean(axis=1)))
            cut = lo + quiet * step + step // 2
            parts.append(self(pcm[s:cut]))
            s = cut
        parts.append(self(pcm[s:]))
        return " ".join(p for p in parts if p)
