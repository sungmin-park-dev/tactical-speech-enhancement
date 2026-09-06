"""Validated software-test defaults; these are not acoustic calibration values."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Settings:
    sample_rate: int = 16_000
    frame_samples: int = 256
    peak_threshold: float = 0.80
    recovery_threshold: float = 0.60
    saturation_threshold: float = 0.985
    saturation_samples: int = 2
    hold_ms: float = 64.0
    normal_confirm_ms: float = 64.0
    recovery_ms: float = 128.0
    model_deadline_ms: float = 16.0
    model_confirm_frames: int = 2
    model_crossfade_ms: float = 64.0
    ceiling_dbfs: float = -6.0
    limiter_release_ms: float = 100.0
    jitter_target_frames: int = 4
    jitter_max_frames: int = 8
    telemetry_capacity: int = 4096

    def __post_init__(self) -> None:
        if self.sample_rate != 16_000 or self.frame_samples != 256:
            raise ValueError("the pinned streaming model requires 16000 Hz / 256 samples")
        for name in (
            "sample_rate", "frame_samples", "saturation_samples", "model_confirm_frames",
            "jitter_target_frames", "jitter_max_frames", "telemetry_capacity",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "peak_threshold", "recovery_threshold", "saturation_threshold", "hold_ms",
            "normal_confirm_ms", "recovery_ms", "model_deadline_ms", "model_crossfade_ms",
            "ceiling_dbfs", "limiter_release_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.recovery_threshold < self.peak_threshold <= self.saturation_threshold <= 1:
            raise ValueError("thresholds must satisfy 0 < recovery < peak <= saturation <= 1")
        if not -120 <= self.ceiling_dbfs <= -6:
            raise ValueError("ceiling_dbfs must be between -120 and -6")
        for name in ("hold_ms", "normal_confirm_ms"):
            frames = getattr(self, name) / self.frame_ms
            if frames < 1 or not frames.is_integer():
                raise ValueError(f"{name} must be a positive whole number of frames")
        for name in ("recovery_ms", "model_crossfade_ms"):
            samples = getattr(self, name) * self.sample_rate / 1000
            if samples < 1 or not samples.is_integer():
                raise ValueError(f"{name} must resolve to a positive whole number of samples")
        if not 0 < self.model_deadline_ms <= self.frame_ms:
            raise ValueError("model_deadline_ms must be in (0, 16] for fixed two-frame alignment")
        if self.limiter_release_ms <= 0:
            raise ValueError("limiter_release_ms must be positive")
        if self.jitter_target_frames > self.jitter_max_frames:
            raise ValueError("jitter target must not exceed its maximum")
        if self.jitter_max_frames > 64:
            raise ValueError("jitter_max_frames must not exceed 64")
        if self.telemetry_capacity > 4096:
            raise ValueError("telemetry_capacity must not exceed 4096")

    @property
    def frame_seconds(self) -> float:
        return self.frame_samples / self.sample_rate

    @property
    def frame_ms(self) -> float:
        return 1000 * self.frame_seconds

    @property
    def hold_frames(self) -> int:
        return round(self.hold_ms / self.frame_ms)

    @property
    def normal_confirm_frames(self) -> int:
        return round(self.normal_confirm_ms / self.frame_ms)

    @property
    def recovery_samples(self) -> int:
        return round(self.recovery_ms * self.sample_rate / 1000)

    @property
    def model_crossfade_samples(self) -> int:
        return round(self.model_crossfade_ms * self.sample_rate / 1000)

    def to_dict(self) -> dict:
        return asdict(self)
