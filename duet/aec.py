"""Ear 1, self-hearing: cancel our own voice from the microphone so DUET can listen while it speaks.

A partitioned-block frequency-domain NLMS filter (overlap-save, 256-sample blocks = 16 ms at 16 kHz, 25 partitions =
400 ms of echo path, enough for WASAPI's round trip plus a room) learns the path from what we play (the reference)
to what the microphone hears. It uses two paths: the background filter adapts on every block; the foreground filter,
whose output is used, takes the background's weights only when the background cancels clearly better. When the
caller talks over us (double talk) the background's error jumps, it stops being copied, and the foreground keeps a
clean echo path instead of learning the caller's voice as echo.

Each block also reports `near_end`: residual power over the residual echo we would expect from this echo estimate
(estimated echo power / measured cancellation), so ~1 means "only our own echo is left" and >> 1 means the caller is
speaking. The barge-in logic reads that together with the VAD.
"""
from dataclasses import dataclass

import numpy as np

N = 256
# Measured on five simulated calls (2026-11-03): echo alone scored as a voice in 294 blocks with the old settings,
# 153 with these; real caller speech during our sound caught in 219 of 377 blocks (262 before).
REF_ACTIVE = 1e-6  # mean power (-60 dB) above which a block counts as us playing: a sentence's soft fade-out is not
TAIL_HOLD = 4  # blocks (64 ms): our last sound is still on its way out of the speaker and across the room
TAIL_DECAY = 10 ** (-3.0 / 10)  # then it fades 3 dB per 16 ms block


@dataclass
class AecFrame:
    out: np.ndarray  # cleaned microphone block (N samples)
    near_end: float  # residual / expected residual echo
    erle_db: float  # current cancellation estimate
    echo_db: float  # level of the echo estimate in this block
    echo_possible: bool  # something we played is still inside the echo span


class EchoCanceller:
    def __init__(self, taps_ms=400, rate=16_000, mu=1.0):
        self.M = int(np.ceil(taps_ms * rate / 1000 / N))
        self.mu = mu
        self.Wf = np.zeros((self.M, N + 1), np.complex128)  # foreground (used)
        self.Wb = np.zeros((self.M, N + 1), np.complex128)  # background (learning)
        self.X = np.zeros((self.M, N + 1), np.complex128)
        self.x_prev = np.zeros(N)
        self.Px = np.full(N + 1, 1e-6)
        self.pe_f = self.pe_b = self.pm = self.py = 1e-10
        self.erle = 1.0
        self.noise = 1e-7
        self.copies = 0
        self.echo_level = 0.0  # mic power of our own voice (echo alone)
        self.since_ref = 10 ** 6  # blocks since we last played anything

    def _filter(self, W, d):
        y = np.fft.irfft((W * self.X).sum(axis=0), 2 * N)[N:]
        return d - y, y

    def process(self, mic, ref, adapt=True):
        """One block: mic and ref (what was played into this block's time span), N samples each, 16 kHz.
        `adapt=False` while the caller holds the floor: our backchannel then plays over their voice, and learning from
        that double talk would teach the filter the caller."""
        mic = np.asarray(mic, np.float64)
        ref = np.asarray(ref, np.float64)
        xk = np.fft.rfft(np.concatenate([self.x_prev, ref]))
        self.x_prev = ref
        self.X = np.roll(self.X, 1, axis=0)
        self.X[0] = xk
        self.Px = 0.9 * self.Px + 0.1 * np.abs(xk) ** 2
        ref_active = np.mean(ref ** 2) > REF_ACTIVE

        e_f, y_f = self._filter(self.Wf, mic)
        e_b, _ = self._filter(self.Wb, mic)
        if ref_active and adapt:
            E = np.fft.rfft(np.concatenate([np.zeros(N), e_b]))
            G = self.mu * np.conj(self.X) * E / (self.M * self.Px + 1e-6)
            g = np.fft.irfft(G, 2 * N, axis=1)[:, :N]  # gradient constraint: keep the filter causal and N long
            self.Wb += np.fft.rfft(np.concatenate([g, np.zeros((self.M, N))], axis=1), axis=1)

        pm, pf, pb, py = (float(np.mean(v ** 2)) for v in (mic, e_f, e_b, y_f))
        if pf > pm:
            # The filter predicts echo that isn't there (typically a longer tail than the room has, in the gap
            # between two sentences): its output is our own voice, inverted. The microphone is the better guess.
            e_f, pf = mic, pm
        a = 0.3
        self.pm += a * (pm - self.pm)
        self.pe_f += a * (pf - self.pe_f)
        self.pe_b += a * (pb - self.pe_b)
        self.py += a * (py - self.py)
        if ref_active and adapt:
            if self.pe_b < 0.6 * self.pe_f and self.pe_b < self.pm:
                self.Wf[:] = self.Wb
                self.copies += 1
                self.pe_f = self.pe_b
            elif self.pe_b > 4 * self.pm:  # background ran away (double talk): restart it from the foreground
                self.Wb[:] = self.Wf
                self.pe_b = self.pe_f
        # our echo can still arrive while anything we played is inside the filter's span (the room's tail)
        echo_possible = ref_active or float(np.abs(self.X).max()) > 1e-2
        if not echo_possible:  # noise floor, minimum-statistics style: falls fast, rises slowly (~8 s)
            self.noise += (0.2 if pm < self.noise else 0.002) * (pm - self.noise)

        # A caller can only add energy to the microphone, so near-end power is at most what the microphone heard.
        residual = min(self.pe_f, self.pm)
        # After our last sound the room still rings and the filter cancels that tail poorly: expect our voice at the
        # level it had in the mic, held for the playback delay, then fading like a room (the friend's first sim heard
        # the tail of "How are you doing?" as the caller: near-end 10-14 for 150 ms).
        self.since_ref = 0 if ref_active else self.since_ref + 1
        tail = 0.0 if ref_active else self.echo_level * TAIL_DECAY ** max(0, self.since_ref - TAIL_HOLD)
        expected = max(self.py / max(self.erle, 1.0), tail) + 2 * self.noise
        near_end = residual / expected if echo_possible else residual / (2 * self.noise)
        if ref_active and near_end < 2.5:  # how loud our own voice is in the mic, from blocks of echo alone
            self.echo_level += 0.3 * (self.pm - self.echo_level)
        # cancellation estimate, learned only in blocks that look like echo alone (else double talk teaches it
        # that we cancel badly, and the caller starts to look like echo)
        if ref_active and py > 10 * self.noise and pf < 0.5 * pm and near_end < 2.5:
            self.erle += 0.05 * (min(pm / max(pf, 1e-12), 1e4) - self.erle)
        return AecFrame(e_f.astype(np.float32), float(near_end), float(10 * np.log10(self.erle)),
                        float(10 * np.log10(py + 1e-12)), bool(echo_possible))


def chime(rate=48_000, seconds=1.2, level_db=-20.0, seed=3):
    """A soft startup sound that teaches the echo canceller the room fast: pink-ish noise under two quiet tones,
    faded in and out. Broadband noise converges NLMS in about a second where speech takes several."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / rate)
    spec *= np.where(f > 80, 1 / np.sqrt(np.maximum(f, 80) / 80), 0) * (f < 7500)
    noise = np.fft.irfft(spec, n)
    noise /= np.sqrt(np.mean(noise ** 2))
    t = np.arange(n) / rate
    tones = 0.5 * np.sin(2 * np.pi * 523.25 * t) + 0.4 * np.sin(2 * np.pi * 783.99 * t) * (t > seconds / 2)
    env = np.minimum(1, np.minimum(t / 0.15, (seconds - t) / 0.3))
    x = (noise + tones) * env
    return (x / np.sqrt(np.mean(x ** 2)) * 10 ** (level_db / 20)).astype(np.float32)
