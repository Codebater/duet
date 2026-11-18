"""Ear 2, voice activity: Silero VAD v5 (MIT, 2 MB ONNX), 32 ms steps at 16 kHz, ~0.1 ms each.

Stateful: each 512-sample chunk is fed with the previous chunk's last 64 samples in front, as Silero's own wrapper
does. `Vad.step(chunk) -> probability of speech`.
"""
import numpy as np
import onnxruntime as ort

from .config import RATE, VAD_ONNX

CHUNK, CONTEXT = 512, 64


class Vad:
    def __init__(self):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(VAD_ONNX), so, providers=["CPUExecutionProvider"])
        self.sr = np.array(RATE, np.int64)
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros(CONTEXT, np.float32)

    def step(self, chunk):
        x = np.concatenate([self.context, np.asarray(chunk, np.float32)])[None, :]
        out, self.state = self.session.run(None, {"input": x, "state": self.state, "sr": self.sr})
        self.context = x[0, -CONTEXT:]
        return float(out[0, 0])
