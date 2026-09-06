import math

import numpy as np
import pytest

from tactical_speech_enhancement.guard import InputGuard, PeakLimiter


def frame(value=0.1, dtype=np.float32):
    return np.full(256, value, dtype=dtype)


def opened_guard():
    guard = InputGuard()
    for _ in range(12):
        guard.inspect(frame())
    assert guard.stage == "normal"
    return guard


def test_startup_confirm_then_continuous_sample_ramp():
    guard = InputGuard()
    for _ in range(4):
        assert not guard.inspect(frame()).gains.any()
    gains = np.concatenate([guard.inspect(frame()).gains for _ in range(8)])
    np.testing.assert_array_equal(gains, np.arange(1, 2049) / 2048)
    assert guard.stage == "normal"
    assert np.all(guard.inspect(frame()).gains == 1)


@pytest.mark.parametrize("index", [0, 128, 255])
def test_impulse_mutes_whole_frame_and_recovers_within_256_ms(index):
    guard = opened_guard()
    impulse = frame()
    impulse[index] = 0.9
    hit = guard.inspect(impulse)
    assert hit.unsafe and hit.reason == "large_peak" and not hit.gains.any()
    for _ in range(8):
        assert not guard.inspect(frame()).gains.any()
    ramp = np.concatenate([guard.inspect(frame()).gains for _ in range(8)])
    assert len(ramp) == 2048 and ramp[-1] == 1
    assert np.max(np.diff(np.r_[0, ramp])) <= 1 / 2048
    assert np.min(np.diff(ramp)) >= 0
    assert guard.stage == "normal"
    # 16 frames after the last bad frame end; at most one extra frame
    # separates the actual last bad sample from that frame end.
    assert 16 * 16 == 256
    assert 256 + 16 <= 272


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_peak_and_recovery_thresholds_include_exact_boundaries(dtype):
    guard = opened_guard()
    below = np.nextafter(dtype(0.8), dtype(0))
    assert not guard.inspect(frame(below, dtype)).unsafe
    assert guard.inspect(frame(0.8, dtype)).unsafe
    for _ in range(4):
        guard.inspect(frame())
    for _ in range(4):
        assert not guard.inspect(frame(0.6, dtype)).unsafe
    assert guard.stage == "recovery"
    assert guard.inspect(frame(0.6, dtype)).gains[-1] == 0.125


def test_saturation_is_detected_across_the_frame_boundary():
    guard = opened_guard()
    a, b = frame(), frame()
    a[-1], b[0] = 0.985, -0.985
    first = guard.inspect(a)
    second = guard.inspect(b)
    assert first.reason == "large_peak" and not first.saturated
    assert second.reason == "saturation" and second.saturated
    assert second.generation == first.generation + 1


def test_isolated_rail_samples_do_not_count_as_consecutive_saturation():
    guard = opened_guard()
    block = frame()
    block[[4, 6]] = 0.99
    assert guard.inspect(block).reason == "large_peak"


@pytest.mark.parametrize("block", [
    np.full(256, np.nan), np.full(256, np.inf), np.full(256, -np.inf),
    np.full(256, 1.01), np.zeros(255), np.zeros((256, 1)), np.zeros(256, dtype=int),
    np.zeros(256, dtype=complex), ["x"] * 256, None,
])
def test_malformed_input_fails_closed_and_recovers_automatically(block):
    guard = opened_guard()
    result = guard.inspect(block)
    assert not result.valid and result.unsafe
    assert not result.audio.any() and not result.gains.any()
    for _ in range(16):
        guard.inspect(frame())
    assert guard.gain == 1


def test_device_overload_is_protected_even_with_small_finite_input():
    result = opened_guard().inspect(frame(), overload=True)
    assert result.reason == "device_overload" and result.unsafe


def test_midband_does_not_trigger_normal_but_resets_confirmation_and_pauses_ramp():
    guard = opened_guard()
    assert np.all(guard.inspect(frame(0.7)).gains == 1)
    guard.inspect(frame(0.9))
    for _ in range(4):
        guard.inspect(frame())
    for _ in range(3):
        guard.inspect(frame())
    guard.inspect(frame(0.7))
    for _ in range(3):
        assert not guard.inspect(frame()).gains.any()
    assert guard.stage == "confirm"
    guard.inspect(frame())
    first = guard.inspect(frame())
    assert first.gains[-1] == 0.125
    paused = guard.inspect(frame(0.7))
    assert np.all(paused.gains == 0.125)
    resumed = guard.inspect(frame())
    assert resumed.gains[0] == 0.125 + 1 / 2048
    assert resumed.gains[-1] == 0.25


def test_repeated_impulse_restarts_hold_without_an_upward_gain_jump():
    guard = opened_guard()
    for _ in range(4):
        assert not guard.inspect(frame(0.9)).gains.any()
        for _ in range(10):
            guard.inspect(frame())
        assert guard.gain == 0.25
    guard.inspect(frame(0.9))
    for _ in range(4):
        assert not guard.inspect(frame()).gains.any()
    assert guard.stage == "confirm"


def test_output_limiter_attacks_and_enforces_representable_ceiling():
    limiter = PeakLimiter()
    for value in (1, -1, 1e30, -1e30, 0.5):
        output = limiter.process(frame(value))
        assert np.isfinite(output).all()
        assert float(np.max(np.abs(output))) <= 10 ** (-6 / 20)


def test_limiter_release_time_is_an_exponential_constant_not_full_recovery():
    limiter = PeakLimiter()
    limiter.process(np.array([1.0], dtype=np.float32))
    initial = limiter.gain
    limiter.process(np.zeros(1600, dtype=np.float32))
    assert limiter.gain == pytest.approx(1 - (1 - initial) * math.exp(-1), abs=1e-12)
    assert limiter.gain < 1


def test_nonfinite_final_output_silences_entire_frame():
    limiter = PeakLimiter()
    block = frame()
    block[77] = np.nan
    assert not limiter.process(block).any()
    assert limiter.invalid_blocks == 1


@pytest.mark.parametrize("position", [-1, 0, 1])
def test_float32_threshold_neighbors_for_peak_saturation_and_recovery(position):
    def neighbor(value):
        rounded = np.float32(value)
        if position == 0:
            return rounded
        return np.nextafter(rounded, np.float32(0 if position < 0 else 1))

    peak_guard = opened_guard()
    peak_input = frame()
    peak_input[47] = neighbor(0.80)
    assert peak_guard.inspect(peak_input).unsafe == (position >= 0)

    rail_guard = opened_guard()
    rail_input = frame()
    rail_input[47:49] = neighbor(0.985)
    rail = rail_guard.inspect(rail_input)
    assert rail.unsafe  # Even just below the rail threshold exceeds peak=0.80.
    assert rail.saturated == (position >= 0)
    assert rail.reason == ("saturation" if position >= 0 else "large_peak")

    recovery_guard = opened_guard()
    recovery_guard.inspect(frame(0.9))
    for _ in range(4):
        recovery_guard.inspect(frame())
    for _ in range(4):
        result = recovery_guard.inspect(frame(neighbor(0.60)))
        assert not result.unsafe
        assert not result.gains.any()
    assert recovery_guard.stage == ("recovery" if position <= 0 else "confirm")
    following = recovery_guard.inspect(frame(neighbor(0.60)))
    assert bool(following.gains.any()) == (position <= 0)


def test_sustained_saturation_repeatedly_resets_hold_and_recovers_after_last_frame():
    guard = opened_guard()
    for index in range(12):
        saturated = frame(0.985 if index % 2 == 0 else -1.0)
        result = guard.inspect(saturated)
        assert result.unsafe and result.saturated and result.reason == "saturation"
        assert not result.gains.any()
        assert result.generation == index + 1
    # The hold starts after the final saturated frame, regardless of how long
    # saturation lasted; confirmation and ramp may not overlap it.
    for _ in range(8):
        assert not guard.inspect(frame(0.55)).gains.any()
    recovery = np.concatenate([guard.inspect(frame(0.55)).gains for _ in range(8)])
    np.testing.assert_array_equal(recovery, np.arange(1, 2049) / 2048)
    assert guard.gain == 1 and guard.stage == "normal"
    assert guard.summary()["saturated_frames"] == 12
