"""Understanding, the text reflex: one small ONNX model, one pass, five answers (~5-10 ms on the CPU).

bge-small-en-v1.5 (MIT, 33M params) fine-tuned with five heads on one CLS vector (`tools/train_text_reflex.py`):
- **intent**: which of Reflexx's 26 front-desk intents (book, price, timeline, ... out of scope);
- **act**: what kind of turn it is: request, greeting, small talk, backchannel, stop, repeat, continue, thanks,
  goodbye, affirm, deny;
- **flags**: does it contain a greeting, does it contain a request (both can be true: "Hi, I need a quote");
- **sentiment**: negative / neutral / positive in the words;
- **complete**: does the sentence sound finished, so the turn-taker can answer sooner or wait.
This is Reflexx's rule applied to dialogue: decide, never generate, in a single forward pass.
"""
import json
import re
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .config import TEXT_REFLEX_DIR


@dataclass
class Reading:
    text: str
    intent: str
    intent_p: float
    act: str
    act_p: float
    greeting_p: float
    request_p: float
    sentiment: dict  # {"neg": p, "neu": p, "pos": p}
    complete_p: float
    ms: float
    mode: str = None  # the friend's heads (text-reflex-v5 on): what they want, and how strong the feeling is
    mode_p: float = 0.0
    feeling: str = None
    feeling_p: float = 0.0

    @property
    def valence(self):
        return self.sentiment["pos"] - self.sentiment["neg"]

    @property
    def has_greeting(self):
        return self.greeting_p > 0.5

    @property
    def has_request(self):
        return self.request_p > 0.5 and self.act == "request"


def normalize(text):
    """The same cleanup the training data had: whitespace, and no trailing punctuation."""
    return re.sub(r"[\s.,!?;:]+$", "", re.sub(r"\s+", " ", text).strip())


class TextReflex:
    def __init__(self, path=TEXT_REFLEX_DIR, threads=2):
        meta = json.loads((path / "meta.json").read_text())
        self.intents, self.acts, self.sents = meta["intents"], meta["acts"], meta["sentiments"]
        self.modes, self.feelings = meta.get("modes"), meta.get("feelings")
        self.max_length = meta["max_length"]
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.head_tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.head_tok.enable_truncation(self.max_length)
        # a long turn is read from its end: the question comes last ("... How long would a new one take?"), and a
        # first 64 tokens of a 20 s turn missed it in the long-talker simulation; its opening is read for the greeting
        self.tok.enable_truncation(self.max_length, direction="left")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(path / "text_reflex.onnx"), so, providers=["CPUExecutionProvider"])
        self.read("warm up")

    def _run(self, enc):
        ids = np.array([enc.ids], np.int64)
        return self.session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})

    def read(self, text):
        t0 = time.perf_counter()
        enc = self.tok.encode(normalize(text) or "")
        intent, act, flags, sent, complete, *friend = self._run(enc)
        greeting = float(flags[0, 0])
        if enc.overflowing:  # longer than 64 tokens: the opening decides the greeting flag
            greeting = max(greeting, float(self._run(self.head_tok.encode(normalize(text)))[2][0, 0]))
        i, a = int(intent[0].argmax()), int(act[0].argmax())
        extra = {}
        if friend and self.modes:
            m, f = int(friend[0][0].argmax()), int(friend[1][0].argmax())
            extra = {"mode": self.modes[m], "mode_p": float(friend[0][0, m]), "feeling": self.feelings[f],
                     "feeling_p": float(friend[1][0, f])}
        return Reading(text, self.intents[i], float(intent[0, i]), self.acts[a], float(act[0, a]),
                       greeting, float(flags[0, 1]), dict(zip(self.sents, map(float, sent[0]))),
                       float(complete[0]), 1000 * (time.perf_counter() - t0), **extra)
