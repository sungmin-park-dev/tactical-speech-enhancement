"""protection_mode="duck": attenuate anomalies, keep loud/clipped speech, no retroactive cut."""

from __future__ import annotations

import numpy as np
import pytest

from tactical_speech_enhancement.config import Settings
from tactical_speech_enhancement.engine import TxController
from tactical_speech_enhancement.guard import InputGuard
from tactical_speech_enhancement.worker import SynchronousWorker

DUCK = Settings(protection_mode="duck")
rng = np.random.default_rng(7)


def speech_like(rms, n=256):
    x = rng.standard_normal(n).astype(np.float32)
    return (x / np.sqrt(np.mean(x ** 2)) * rms).astype(np.float32)


def opened(settings=DUCK, rms=0.05, frames=40):
    guard = InputGuard(settings)
    for _ in range(frames):
        guard.inspect(speech_like(rms))
    assert guard.stage == "normal"
    return guard


def test_default_mode_is_duck_and_mute_remains_available():
    assert Settings().protection_mode == "duck"
    assert Settings(protection_mode="mute").protection_mode == "mute"


@pytest.mark.parametrize("kwargs", [{"protection_mode": "soft"}, {"duck_gain_db": 0.0},
                                    {"relative_trigger_db": -1.0}, {"gain_fade_ms": 50.0}])
def test_duck_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_blast_is_attenuated_not_zeroed_with_bounded_gain_slope():
    guard = opened()
    blast = np.clip(speech_like(0.9), -1, 1)
    gains = np.concatenate([guard.inspect(blast).gains for _ in range(3)])
    assert np.all(gains > 0)
    assert gains[-1] == pytest.approx(DUCK.duck_gain, rel=1e-5)
    assert np.max(np.abs(np.diff(gains))) <= (1 - DUCK.duck_gain) / DUCK.gain_fade_samples + 1e-6


def test_sustained_loud_clipped_speech_is_not_treated_as_a_blast():
    guard = opened(rms=0.05)
    shout = np.clip(speech_like(0.6) * 3, -1, 1)   # saturated, but sustained
    gains = [guard.inspect(shout).gains[-1] for _ in range(200)]
    assert guard.ignored_relative > 0
    assert gains[-1] == pytest.approx(1.0)          # back to unity after the level adapts


def test_engine_duck_keeps_frames_before_anomaly_and_attenuates_the_anomaly():
    class Identity:
        def __init__(self):
            self.previous = np.zeros(256, np.float32)

        def reset(self):
            self.previous = np.zeros(256, np.float32)

        def process(self, block):
            out, self.previous = self.previous, block.copy()
            return out

    model = Identity()
    ctrl = TxController(model, DUCK, SynchronousWorker(model, DUCK))
    outputs = []
    frames = [speech_like(0.05) for _ in range(40)] + [np.clip(speech_like(0.9), -1, 1)] + \
        [speech_like(0.05) for _ in range(6)]
    for i, block in enumerate(frames):
        outputs.append(ctrl.process(block, now=i * 0.016).audio)
    # outputs[k] carries source k-2; the anomaly is source 40, the two before it are 38, 39
    assert np.any(outputs[40]) and np.any(outputs[41])       # not cut retroactively
    blast_out = outputs[42]
    assert np.any(blast_out)
    assert np.sqrt(np.mean(blast_out[64:] ** 2)) < 0.05 * np.sqrt(np.mean(frames[40] ** 2))


@pytest.mark.parametrize("bad", [np.full(256, np.nan, np.float32), np.full(256, 2.0, np.float32), np.zeros(10, np.float32)])
def test_default_mode_still_zeroes_malformed_input(bad):
    class Identity:
        def reset(self):
            pass

        def process(self, block):
            return block.copy()

    settings = Settings()
    model = Identity()
    ctrl = TxController(model, settings, SynchronousWorker(model, settings))
    for i in range(40):
        ctrl.process(speech_like(0.05), now=i * 0.016)
    outputs = [ctrl.process(bad, now=40 * 0.016).audio]
    outputs += [ctrl.process(speech_like(0.05), now=(41 + k) * 0.016).audio for k in range(2)]
    assert not np.any(outputs[2])          # the malformed source frame, two ticks later
    assert all(np.max(np.abs(o)) <= 10 ** (-6 / 20) for o in outputs)
