"""The sound card: one full-duplex WASAPI stream at 48 kHz, 20 ms blocks, so every microphone block arrives with
the exact block we played at the same time (what the echo canceller needs)."""
import numpy as np
import sounddevice as sd

from .config import BLOCK, DEVICE_RATE


def wasapi_defaults():
    for api in sd.query_hostapis():
        if "WASAPI" in api["name"]:
            return api["default_input_device"], api["default_output_device"]
    return sd.default.device


class LiveEngine:
    def __init__(self, conductor, input=None, output=None):
        self.c = conductor
        din, dout = wasapi_defaults()
        self.input = din if input is None else input
        self.output = dout if output is None else output
        self.in_ch = 1
        self.status_count = 0

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            self.status_count += 1
        out = self.c.player.pull(frames)
        outdata[:] = out[:, None]
        self.c.feed(indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0], out)

    def open(self):
        names = sd.query_devices(self.input)["name"], sd.query_devices(self.output)["name"]
        for in_ch, out_ch in ((1, 1), (1, 2), (sd.query_devices(self.input)["max_input_channels"], 2)):
            try:
                self.stream = sd.Stream(device=(self.input, self.output), samplerate=DEVICE_RATE, blocksize=BLOCK,
                                        dtype="float32", channels=(in_ch, out_ch), latency="low",
                                        callback=self._callback)
                self.stream.start()
                lat = self.stream.latency
                return f"{names[0]} -> {names[1]}, {in_ch} in / {out_ch} out, latency in {lat[0] * 1000:.0f} ms " \
                       f"out {lat[1] * 1000:.0f} ms"
            except sd.PortAudioError as e:
                last = e
        raise last

    def close(self):
        self.stream.stop()
        self.stream.close()
