"""Echo canceller on a simulated laptop: our voice through a delayed, reverberant, slightly distorting path, plus a
caller who talks over us. Checks cancellation, that the caller survives, and that `near_end` tells them apart."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from duet.aec import N, EchoCanceller, chime  # noqa: E402
from duet.vad import Vad  # noqa: E402
from duet.voice import Voice  # noqa: E402

RATE = 16_000


def room(rate, delay_ms=55, rt60=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n = int(0.35 * rate)
    t = np.arange(n) / rate
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60) * 0.08
    h[0] = 0.6
    return np.concatenate([np.zeros(int(delay_ms * rate / 1000)), h])


def simulate(voice, out_dir=None):
    agent = resample_poly(voice.synth("We build websites, online stores and custom software for small businesses, "
                                      "and most projects start within a week of the first call."), 1, 3)
    voice.set_voice("f2")
    caller = resample_poly(voice.synth("Sorry, wait, how much does that cost?"), 1, 3)
    voice.set_voice("orion")
    warm = resample_poly(chime(), 1, 3)  # the startup chime, as the live system plays it
    lead = len(warm) + RATE // 2
    n = lead + len(agent) + RATE
    ref = np.zeros(n)
    ref[:len(warm)] = warm
    ref[lead:lead + len(agent)] = agent
    echo = fftconvolve(ref, room(RATE))[:n]
    echo = 0.8 * np.tanh(echo / 0.8) if "--linear" not in sys.argv else echo  # small speaker distortion
    near = np.zeros(n)
    start = lead + int(3.0 * RATE)
    near[start:start + len(caller)] = 0.5 * caller
    rng = np.random.default_rng(1)
    mic = echo + near + 10 ** (-55 / 20) * rng.standard_normal(n)
    return ref, mic, near, start, len(caller), lead


def test_cancels_echo_and_keeps_caller():
    voice = Voice("orion")
    ref, mic, near, start, n_caller, lead = simulate(voice)
    aec, vad = EchoCanceller(), Vad()
    out, ne, vp = [], [], []
    for i in range(0, len(mic) - N + 1, N):
        f = aec.process(mic[i:i + N], ref[i:i + N])
        out.append(f.out)
        ne.append(f.near_end)
    out = np.concatenate(out)
    for i in range(0, len(out) - 511, 512):
        vp.append(vad.step(out[i:i + 512]))
    ne, vp = np.array(ne), np.repeat(vp, 2)[:len(ne)]
    blocks = lambda a, b: slice(a // N, b // N)  # noqa: E731
    echo_only = blocks(lead + RATE // 2, start)
    talk = blocks(start + 1600, start + n_caller - 1600)
    erle = 10 * np.log10(np.mean(mic[lead + RATE // 2:start] ** 2) / np.mean(out[lead + RATE // 2:start] ** 2))
    kept = np.corrcoef(out[start:start + n_caller], near[start:start + n_caller])[0, 1]
    late = slice(start - RATE, start)
    erle_late = 10 * np.log10(np.mean(mic[late] ** 2) / np.mean(out[late] ** 2))
    print(f"ERLE echo-only {erle:.1f} dB (last second before the caller {erle_late:.1f} dB), caller kept corr "
          f"{kept:.3f}, near_end median echo-only {np.median(ne[echo_only]):.1f} vs talk {np.median(ne[talk]):.1f}, "
          f"blocks over 6: echo-only {np.mean(ne[echo_only] > 6):.2f} vs talk {np.mean(ne[talk] > 6):.2f}, "
          f"VAD>0.5 echo-only {np.mean(vp[echo_only] > 0.5):.2f} vs talk {np.mean(vp[talk] > 0.5):.2f}")
    if "--write" in sys.argv:
        d = Path(sys.argv[sys.argv.index("--write") + 1])
        d.mkdir(parents=True, exist_ok=True)
        sf.write(d / "aec_mic.wav", mic.astype(np.float32), RATE)
        sf.write(d / "aec_out.wav", out, RATE)
    # The VAD alone can't tell: our residual echo is speech too. The near-end score has to.
    assert erle > 10
    assert kept > 0.8
    assert np.median(ne[talk]) > 4 * np.median(ne[echo_only])
    # duplex.py's decision, chunk by chunk (32 ms = 2 AEC blocks): talk = VAD > 0.5 and near-end > 6;
    # overlap = 5 of the last 7 chunks
    n_chunks = len(ne) // 2
    chunk_ne = ne[:2 * n_chunks].reshape(n_chunks, 2).max(axis=1)
    talk_chunk = (vp[:2 * n_chunks:2] > 0.5) & (chunk_ne > 6)
    overlap = np.array([talk_chunk[max(0, k - 6):k + 1].sum() >= 5 for k in range(n_chunks)])
    t_chunk = np.arange(n_chunks) * 2 * N
    false_hits = overlap[(t_chunk > lead + RATE // 2) & (t_chunk < start)].sum()
    hits = np.nonzero(overlap & (t_chunk >= start))[0]
    first = (t_chunk[hits[0]] - start) / RATE * 1000 if len(hits) else None
    print(f"overlap decisions on echo alone: {false_hits}; caller caught {first:.0f} ms after they started")
    assert false_hits == 0
    assert first is not None and first < 400


if __name__ == "__main__":
    test_cancels_echo_and_keeps_caller()
