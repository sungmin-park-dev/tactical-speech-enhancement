"""Capture-owned anomaly detection, source-aligned gain and final digital limiting."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Settings


@dataclass(frozen=True)
class InputDecision:
    audio: np.ndarray
    gains: np.ndarray
    generation: int
    valid: bool
    unsafe: bool
    reason: str
    stage: str
    peak: float | None
    saturated: bool


class InputGuard:
    """Advance exactly once per captured frame, independently of inference.

    A decision belongs to that input frame. Its gain envelope must travel with
    the original audio through the fixed alignment delay. A new generation also
    cancels any previously buffered output immediately.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.generation = 0
        self.stage = "confirm"
        self.gain = 0.0
        self._hold_remaining = 0
        self._normal_seen = 0
        self._ramp_samples = 0
        self._rail_run = 0
        self.frames = 0
        self.anomalies = 0
        self.reentries = 0
        self.invalid_frames = 0
        self.saturated_frames = 0

    def inspect(self, block: object, *, overload: bool = False) -> InputDecision:
        """Reject malformed PCM before it can enter the model queue."""
        s = self.settings
        self.frames += 1
        zero = np.zeros(s.frame_samples, dtype=np.float32)
        valid = True
        reason = "normal"
        peak = None
        saturated = False
        try:
            samples = np.asarray(block)
            if samples.shape != (s.frame_samples,) or samples.dtype.kind != "f":
                valid, reason = False, "invalid_format"
            elif not np.isfinite(samples).all():
                valid, reason = False, "nonfinite_input"
            else:
                peak = float(np.max(np.abs(samples)))
                if peak > 1:
                    valid, reason = False, "input_out_of_range"
        except (TypeError, ValueError, OverflowError):
            valid, reason = False, "invalid_format"
        if overload:
            valid, reason = False, "device_overload"
        if valid:
            # Compare thresholds in the input representation, including exact
            # float32 boundary values such as float32(0.60).
            peak_threshold = float(np.asarray(s.peak_threshold, dtype=samples.dtype))
            recovery_threshold = float(np.asarray(s.recovery_threshold, dtype=samples.dtype))
            rail_threshold = float(np.asarray(s.saturation_threshold, dtype=samples.dtype))
            rail = np.abs(samples) >= rail_threshold
            run = self._rail_run
            for is_rail in rail:
                run = min(run + 1, s.saturation_samples) if is_rail else 0
                saturated |= run >= s.saturation_samples
            self._rail_run = run
            if saturated:
                reason = "saturation"
            elif peak >= peak_threshold:
                reason = "large_peak"
            audio = samples.astype(np.float32, copy=True)
            normal = peak <= recovery_threshold
        else:
            self._rail_run = 0
            self.invalid_frames += 1
            audio = zero.copy()
            normal = False
        unsafe = not valid or reason in ("saturation", "large_peak")
        if unsafe:
            self.reentries += self.stage == "normal"
            self.anomalies += 1
            self.saturated_frames += saturated
            self.generation += 1
            self.stage = "hold"
            self._hold_remaining = s.hold_frames
            self._normal_seen = 0
            self._ramp_samples = 0
            self.gain = 0.0
            gains = zero
        elif self.stage == "hold":
            gains = zero
            self._hold_remaining -= 1
            reason = "hold"
            if self._hold_remaining == 0:
                self.stage = "confirm"
        elif self.stage == "confirm":
            gains = zero
            self._normal_seen = self._normal_seen + 1 if normal else 0
            reason = "normal_confirmation"
            if self._normal_seen == s.normal_confirm_frames:
                self.stage = "recovery"
        elif self.stage == "recovery":
            reason = "recovery" if normal else "recovery_paused"
            if normal:
                positions = np.arange(1, s.frame_samples + 1, dtype=np.float64)
                gains = np.minimum((positions + self._ramp_samples) / s.recovery_samples, 1)
                gains = gains.astype(np.float32)
                self._ramp_samples = min(self._ramp_samples + s.frame_samples, s.recovery_samples)
                self.gain = float(gains[-1])
                if self._ramp_samples == s.recovery_samples:
                    self.stage = "normal"
            else:
                gains = np.full(s.frame_samples, self.gain, dtype=np.float32)
        else:
            gains = np.ones(s.frame_samples, dtype=np.float32)
        return InputDecision(
            audio=audio, gains=gains, generation=self.generation, valid=valid,
            unsafe=unsafe, reason=reason, stage=self.stage, peak=peak, saturated=saturated,
        )

    def summary(self) -> dict:
        return {
            "frames": self.frames, "anomalies": self.anomalies,
            "reentries": self.reentries, "invalid_frames": self.invalid_frames,
            "saturated_frames": self.saturated_frames, "generation": self.generation,
            "stage": self.stage, "gain": self.gain,
        }


class PeakLimiter:
    """Zero-lookahead sample limiter with exponential release and a final clamp.

    Nonfinite output silences the whole frame. This limits digital samples only;
    it does not establish a sound-pressure or hearing-protection rating.
    """

    def __init__(
        self, ceiling_dbfs: float = -6.0, sample_rate: int = 16_000, release_ms: float = 100.0,
    ) -> None:
        if not all(math.isfinite(x) for x in (ceiling_dbfs, sample_rate, release_ms)):
            raise ValueError("limiter parameters must be finite")
        if not -120 <= ceiling_dbfs <= 0 or sample_rate <= 0 or release_ms <= 0:
            raise ValueError("invalid limiter ceiling, sample rate or release time")
        self.ceiling = 10 ** (ceiling_dbfs / 20)
        rounded = np.float32(self.ceiling)
        self._float32_ceiling = (
            np.nextafter(rounded, np.float32(0)) if rounded > self.ceiling else rounded
        )
        self._release = math.exp(-1 / (sample_rate * release_ms / 1000))
        self.gain = 1.0
        self.invalid_blocks = 0

    def reset(self) -> None:
        self.gain = 1.0

    def process(self, block: np.ndarray) -> np.ndarray:
        samples = np.asarray(block)
        if samples.ndim != 1:
            raise ValueError("limiter expects one-dimensional mono PCM")
        output = np.zeros(samples.shape, dtype=np.float32)
        if samples.dtype.kind != "f" or not np.isfinite(samples).all():
            self.invalid_blocks += 1
            return output
        gain = self.gain
        for i, sample in enumerate(samples):
            magnitude = abs(float(sample))
            required = 1.0 if magnitude <= self.ceiling else self.ceiling / magnitude
            if required < gain:
                gain = required
            else:
                gain = self._release * gain + (1 - self._release)
            value = float(sample) * gain
            output[i] = min(max(value, -self.ceiling), self.ceiling)
        self.gain = gain
        np.clip(output, -self._float32_ceiling, self._float32_ceiling, out=output)
        if not np.isfinite(output).all():
            # Also cover overflow during conversion/arithmetic from extended
            # precision input types. No nonfinite result reaches playback.
            self.invalid_blocks += 1
            output.fill(0)
            self.gain = 1.0
        return output
