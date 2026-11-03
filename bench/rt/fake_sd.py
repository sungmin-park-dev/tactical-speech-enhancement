"""Fake `sounddevice` module for running the repository's live path without hardware.

It emulates one full-duplex device clock that drives the callback every
`blocksize` samples. The clock can be offset by `CONFIG["ppm"]`. When the
callback thread wakes later than `CONFIG["buffer_periods"]` periods, missed
periods are dropped and the next callback carries an xrun status flag, as
PortAudio/ALSA would report input overflow / output underflow.

This is a measurement harness, not a model of any specific ALSA driver.
"""

from __future__ import annotations

import threading
import time

import numpy as np

CONFIG = {
    "ppm": 0.0,
    "buffer_periods": 3,       # how late a callback may be before an xrun is declared
    "input_provider": None,    # callable(ordinal) -> float32 (blocksize,)
    "on_tick": None,           # callable(record dict)
    "reported_latency": (0.016, 0.032),
}

STATE = {"callbacks": 0, "xruns": 0, "dropped_periods": 0, "thread_native_id": None}


class PortAudioError(Exception):
    pass


class CallbackFlags:
    def __init__(self, input_overflow: bool = False, output_underflow: bool = False) -> None:
        self.input_overflow = input_overflow
        self.output_underflow = output_underflow

    def __bool__(self) -> bool:
        return self.input_overflow or self.output_underflow

    def __repr__(self) -> str:
        return f"CallbackFlags(in_over={self.input_overflow}, out_under={self.output_underflow})"


def query_devices(*_args, **_kwargs):
    return "0 fake full-duplex device, 1 in, 1 out (16000 Hz)"


def check_input_settings(**_kwargs):
    return None


def check_output_settings(**_kwargs):
    return None


class Stream:
    def __init__(self, samplerate, blocksize, device, channels, dtype, latency, callback):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.callback = callback
        self._running = False
        self._thread = None
        self.latency = CONFIG["reported_latency"]

    @property
    def active(self) -> bool:
        return self._running

    def __enter__(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, name="fake-audio", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._running = False
        if self._thread is not None:
            self._thread.join(2.0)
        return False

    def _run(self) -> None:
        STATE["thread_native_id"] = threading.get_native_id()
        ppm = float(CONFIG["ppm"])
        period = self.blocksize / (self.samplerate * (1.0 + ppm * 1e-6))
        provider = CONFIG["input_provider"]
        on_tick = CONFIG["on_tick"]
        limit = CONFIG["buffer_periods"] * period
        t0 = time.monotonic() + 0.05
        k = 0
        ordinal = 0
        pending_flag = False
        while self._running:
            deadline = t0 + k * period
            now = time.monotonic()
            if deadline > now:
                time.sleep(deadline - now)
            woke = time.monotonic()
            late = woke - deadline
            if late > limit:
                missed = int(late // period)
                k += missed
                STATE["xruns"] += 1
                STATE["dropped_periods"] += missed
                pending_flag = True
                deadline = t0 + k * period
            indata = np.empty((self.blocksize, 1), dtype=np.float32)
            indata[:, 0] = provider(ordinal) if provider is not None else 0.0
            outdata = np.zeros((self.blocksize, 1), dtype=np.float32)
            flags = CallbackFlags(pending_flag, pending_flag)
            pending_flag = False
            started = time.monotonic()
            self.callback(indata, outdata, self.blocksize, None, flags)
            ended = time.monotonic()
            STATE["callbacks"] += 1
            if on_tick is not None:
                on_tick({
                    "ordinal": ordinal, "k": k, "deadline": deadline, "woke": woke,
                    "start": started, "end": ended, "status": bool(flags),
                    "out_rms": float(np.sqrt(np.mean(outdata[:, 0] ** 2))),
                })
            ordinal += 1
            k += 1
