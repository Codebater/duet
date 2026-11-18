"""Paths and tuning constants shared by every DUET module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

RATE = 16_000  # what the ears work in: VAD, echo canceller, ASR, prosody
DEVICE_RATE = 48_000  # what the sound card runs at, in and out
BLOCK = 960  # 20 ms at 48 kHz: one audio callback

ASR_DIR = MODELS / "turboctc"
VAD_ONNX = MODELS / "vad" / "silero_vad.onnx"
TTS_DIR = MODELS / "supertonic3"
VOICES_DIR = MODELS / "voices"
# v4 (2026-11-03): feelings and plain statements are small talk, and small talk is out of scope for intent (v1 read
# "I'm very excited" as thanks); v2 and v3 were steps on the way, compared in text-reflex-v4/REPORT.md
TEXT_REFLEX_DIR = MODELS / "text-reflex-v4"
# the friend's: v4's heads plus mode (7) and feeling (5), trained on everyday talk written by the local model; it
# costs the front desk's intent ~1-2 points (text-reflex-v6/REPORT.md), so each persona loads its own
FRIEND_REFLEX_DIR = MODELS / "text-reflex-v6"
# v3: octave-jump fix, pauses inside the speech, end slope on real time (v2: the same head before those; v1 read
# utterance length as mood)
PROSODY_HEAD = MODELS / "prosody-head" / "mood_head_v3.npz"
MOOD_VOICE = MODELS / "mood-voice" / "directions.npz"
