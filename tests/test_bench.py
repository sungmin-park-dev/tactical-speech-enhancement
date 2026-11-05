"""Smoke tests for the benchmark helpers that need no external assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "bench"


@pytest.fixture(autouse=True)
def _bench_path(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCH / "eval"))
    monkeypatch.syspath_prepend(str(BENCH / "rt"))


def test_si_sdr_is_scale_invariant_and_detects_noise():
    import metrics

    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000)
    assert metrics.si_sdr(3 * ref, ref) > 100
    noisy = ref + 0.1 * rng.standard_normal(16000)
    assert 15 < metrics.si_sdr(noisy, ref) < 25


def test_speech_loss_ratio_counts_muted_speech_frames():
    import metrics

    ref = np.tile(np.r_[np.ones(256), np.zeros(256)], 20) * 0.1
    out = ref.copy()
    out[:2560] = 0  # mute the first 5 of 20 active frames
    assert metrics.speech_loss_ratio(out, ref) == pytest.approx(0.25)


def test_buffer_policy_signatures():
    import naive_sim

    fifo = naive_sim.simulate("fifo", minutes=6, rtf=1.02)
    bounded = naive_sim.simulate("bounded8", minutes=6, rtf=1.02)
    assert fifo["lat_ms_min5"] > 10 * fifo["lat_ms_min1"] / 2   # unbounded queue keeps growing
    assert bounded["lat_ms_max"] <= 8 * 16 + 1e-6 + 32           # bounded by the jitter maximum
    stall = naive_sim.simulate("bounded8", minutes=6, stall_every=60, stall_ms=300)
    adaptive = naive_sim.simulate("adaptive", minutes=6, stall_every=60, stall_ms=300)
    assert stall["lat_ms_min5"] > adaptive["lat_ms_min5"]           # ratchet vs convergence
