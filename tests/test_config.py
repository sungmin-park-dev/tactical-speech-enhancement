from dataclasses import FrozenInstanceError

import pytest

from tactical_speech_enhancement.config import Settings


def test_default_contract_is_explicit_and_frozen():
    s = Settings()
    assert (s.frame_ms, s.hold_frames, s.normal_confirm_frames) == (16, 4, 4)
    assert (s.recovery_samples, s.model_crossfade_samples) == (2048, 1024)
    assert (s.jitter_target_frames, s.jitter_max_frames, s.telemetry_capacity) == (4, 8, 4096)
    assert s.to_dict()["peak_threshold"] == 0.8
    with pytest.raises(FrozenInstanceError):
        s.peak_threshold = 1


@pytest.mark.parametrize("changes", [
    {"sample_rate": 48000}, {"frame_samples": 512}, {"sample_rate": 16000.0},
    {"peak_threshold": float("nan")}, {"ceiling_dbfs": float("inf")},
    {"recovery_threshold": 0.8}, {"peak_threshold": 1.1},
    {"hold_ms": 63}, {"normal_confirm_ms": 0}, {"recovery_ms": 0},
    {"recovery_ms": 0.01}, {"model_crossfade_ms": 0}, {"model_deadline_ms": 17},
    {"model_deadline_ms": 0}, {"model_confirm_frames": True},
    {"limiter_release_ms": 0}, {"ceiling_dbfs": 6}, {"ceiling_dbfs": -5},
    {"saturation_samples": 0}, {"jitter_max_frames": 3}, {"jitter_max_frames": 65},
    {"telemetry_capacity": 4097},
])
def test_invalid_settings_are_rejected_before_startup(changes):
    with pytest.raises(ValueError):
        Settings(**changes)


def test_guard_configuration_can_be_adjusted_without_changing_frame_contract():
    s = Settings(hold_ms=32, normal_confirm_ms=16, recovery_ms=16, peak_threshold=0.9)
    assert (s.hold_frames, s.normal_confirm_frames, s.recovery_samples) == (2, 1, 256)
