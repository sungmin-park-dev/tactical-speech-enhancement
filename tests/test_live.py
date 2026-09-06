from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tactical_speech_enhancement import live
from tactical_speech_enhancement.config import Settings


def args():
    return SimpleNamespace(
        bind="127.0.0.1:41001",
        peer="127.0.0.1:41002",
        input_device=0,
        output_device=0,
        duration=0.001,
    )


@pytest.fixture
def fake_runtime(monkeypatch):
    state = {"closed": False, "network_closed": False, "submit": [], "played": []}

    class Controller:
        def __init__(self, *a, **kw):
            pass

        def process(self, block, **kw):
            return SimpleNamespace(sequence=1, audio=block.copy())

        def close(self):
            state["closed"] = True
            return {"worker_stopped": True}

        def summary(self):
            return {"worker_stopped": True}

    class Peer:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

        def submit(self, sequence, audio):
            state["submit"].append(sequence)

        def playout(self, now):
            return np.full(256, 0.99, dtype=np.float32)

        def close(self):
            state["network_closed"] = True

        def summary(self):
            return {"workers_alive": 0}

    class Stream:
        active = True
        latency = (0.01, 0.01)

        def __init__(self, **kw):
            self.callback = kw["callback"]

        def __enter__(self):
            out = np.zeros((256, 1), dtype=np.float32)
            self.callback(np.full((256, 1), 0.1, dtype=np.float32), out, 256, None, False)
            state["played"].append(out)
            return self

        def __exit__(self, *a):
            pass

    sd = SimpleNamespace(
        Stream=Stream,
        check_input_settings=lambda **kw: None,
        check_output_settings=lambda **kw: None,
    )
    monkeypatch.setattr(live, "audio_module", lambda: sd)
    monkeypatch.setattr(live, "GtcrnStreamingEnhancer", lambda *a, **kw: object())
    monkeypatch.setattr(live, "TxController", Controller)
    monkeypatch.setattr(live, "UdpPeer", Peer)
    return state, sd, Peer


def test_callback_final_limit_reports_whole_callback_and_cleans_up(fake_runtime):
    state, _, _ = fake_runtime
    report = live.run_peer(args(), Path("unused"), Settings())
    assert np.max(np.abs(state["played"][0])) <= 10 ** (-6 / 20)
    assert state["closed"] and state["network_closed"]
    assert state["submit"] == [1]
    assert report["audio"]["callback_timing_samples"] == 1
    assert report["audio"]["callback_max_ms"] > 0
    assert report["settings"]["peak_threshold"] == 0.8


def test_keyboard_interrupt_during_device_start_still_cleans_up(fake_runtime):
    state, sd, _ = fake_runtime

    def interrupted(**kw):
        raise KeyboardInterrupt

    sd.Stream = interrupted
    report = live.run_peer(args(), Path("unused"), Settings())
    assert report["portaudio_reported_latency_seconds"] is None
    assert state["closed"] and state["network_closed"]


def test_transport_shutdown_error_does_not_skip_model_cleanup(fake_runtime, monkeypatch):
    state, _, Peer = fake_runtime

    def failing_close(self):
        raise RuntimeError("injected network close failure")

    monkeypatch.setattr(Peer, "close", failing_close)
    with pytest.raises(RuntimeError, match="network close"):
        live.run_peer(args(), Path("unused"), Settings())
    assert state["closed"]


@pytest.mark.parametrize(
    "value",
    [
        "255.255.255.255:5000",
        "224.0.0.1:5000",
        "0.0.0.0:5000",
        "127.0.0.1:0",
        "127.0.0.1:65536",
        "localhost:5000",
    ],
)
def test_remote_endpoint_requires_numeric_unicast(value):
    with pytest.raises(ValueError):
        live.endpoint(value, remote=True)


def test_bind_can_use_all_interfaces():
    assert live.endpoint("0.0.0.0:5000") == ("0.0.0.0", 5000)
