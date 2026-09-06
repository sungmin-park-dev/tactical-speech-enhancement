"""Capture controller: detect first, select a due source frame, then enqueue."""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import Settings
from .guard import InputDecision, InputGuard, PeakLimiter
from .worker import Enhancer, InferenceJob, InferenceWorker, Worker


@dataclass(frozen=True)
class TxFrame:
    audio: np.ndarray
    sequence: int
    diagnostics: dict


class TxController:
    """Keep source n-2 due at capture n without waiting for inference.

    Only call process from one capture owner. The source decision and gain
    envelope share the two-frame dry delay. No inference result can modify
    protection state or extend the capture schedule.
    """

    def __init__(
        self, enhancer: Enhancer, settings: Settings | None = None,
        worker: Worker | None = None, clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings or Settings()
        self.guard = InputGuard(self.settings)
        self.worker = worker if worker is not None else InferenceWorker(enhancer, self.settings, clock)
        self._clock = clock
        self._limiter = PeakLimiter(
            self.settings.ceiling_dbfs, self.settings.sample_rate, self.settings.limiter_release_ms,
        )
        self._sources: deque[InputDecision] = deque(maxlen=2)
        self._sequence = 0
        self._previous_time = -math.inf
        self._model_good = 0
        self._model_mix = 0.0
        self._closed = False
        self._counts: Counter = Counter()
        self._durations: deque[float] = deque(maxlen=self.settings.telemetry_capacity)
        self._duration_total = 0.0
        self._duration_max = 0.0
        self._close_status: dict = {}

    def process(self, block: object, *, now: float | None = None, overload: bool = False) -> TxFrame:
        if self._closed:
            raise RuntimeError("capture controller is closed")
        measured_start = time.perf_counter()
        now = self._clock() if now is None else now
        if not math.isfinite(now) or now < self._previous_time:
            raise ValueError("capture timestamps must be finite and monotonic")
        self._previous_time = now
        sequence = self._sequence
        source_sequence = sequence - 2
        s = self.settings

        # This is deliberately before take() or submit(), even on a full queue.
        decision = self.guard.inspect(block, overload=overload)
        if decision.unsafe:
            self.worker.invalidate(decision.generation)

        result = self.worker.take(source_sequence, decision.generation, now)
        source = self._sources.popleft() if len(self._sources) == 2 else None
        output = np.zeros(s.frame_samples, dtype=np.float32)
        route = "startup"
        source_muted = True
        if source is not None:
            source_muted = (
                source.unsafe or not source.valid or source.generation != decision.generation
                or not np.any(source.gains)
            )
            if source_muted:
                route = "muted"
            else:
                # Validate injected workers too; tags and deadline are capture's
                # final authority, not a permission granted by result arrival.
                usable = (
                    result is not None and result.source_sequence == source_sequence
                    and result.input_sequence == source_sequence + 1
                    and result.generation == decision.generation
                    and result.completed_at < result.submitted_at + s.model_deadline_ms / 1000
                    and result.completed_at <= now
                    and result.audio.shape == (s.frame_samples,)
                    and result.audio.dtype.kind == "f" and np.isfinite(result.audio).all()
                )
                if usable:
                    self._model_good += 1
                    self._counts["model_results_used"] += 1
                    if self._model_good >= s.model_confirm_frames:
                        weights = self._model_mix + np.arange(1, s.frame_samples + 1) / s.model_crossfade_samples
                        weights = np.minimum(weights, 1).astype(np.float32)
                        self._model_mix = float(weights[-1])
                    else:
                        weights = np.zeros(s.frame_samples, dtype=np.float32)
                    output = source.audio * (1 - weights) + result.audio * weights
                    route = "model" if self._model_mix == 1 else "model_crossfade"
                else:
                    self._model_good = 0
                    self._model_mix = 0.0
                    output = source.audio.copy()
                    route = "dry"
                    self._counts["dry_fallback_frames"] += 1
                output *= source.gains
        if source_muted:
            self._model_good = 0
            self._model_mix = 0.0

        # Every branch reaches the final guard; limiting cannot unmute zero.
        output = self._limiter.process(output)

        # Submit after inspection and output selection. Neither a bad frame nor
        # a hidden copy of it may later reappear as a model or dry result.
        if decision.valid and not decision.unsafe:
            self.worker.submit(InferenceJob(sequence, decision.generation, decision.audio.copy(), now))
        self._sources.append(decision)
        self._sequence += 1
        self._counts["frames"] += 1
        self._counts["muted_source_frames"] += source_muted and source is not None
        self._counts["max_alignment_frames"] = max(self._counts["max_alignment_frames"], len(self._sources))
        duration = time.perf_counter() - measured_start
        self._durations.append(duration)
        self._duration_total += duration
        self._duration_max = max(self._duration_max, duration)
        return TxFrame(output, source_sequence, {
            "input_sequence": sequence, "source_sequence": source_sequence,
            "generation": decision.generation, "input_reason": decision.reason,
            "input_stage": decision.stage, "input_unsafe": decision.unsafe,
            "input_peak": decision.peak, "source_muted": source_muted,
            "source_gain_start": float(source.gains[0]) if source is not None else 0,
            "source_gain_end": float(source.gains[-1]) if source is not None else 0,
            "source_reason": source.reason if source is not None else "startup",
            "route": route, "model_mix": self._model_mix,
            "capture_processing_ms": duration * 1000,
        })

    def summary(self) -> dict:
        frames = self._counts["frames"]
        return {
            **dict(self._counts), "guard": self.guard.summary(), "worker": self.worker.summary(),
            "alignment_frames": len(self._sources), "alignment_delay_ms": 2 * self.settings.frame_ms,
            "capture_processing_mean_ms": self._duration_total / frames * 1000 if frames else 0,
            "capture_processing_max_ms": self._duration_max * 1000,
            "capture_processing_recent_p95_ms": float(np.percentile(self._durations, 95) * 1000) if frames else 0,
            "capture_latency_sample_count": len(self._durations),
            "capture_latency_sample_capacity": self.settings.telemetry_capacity,
            "limiter_invalid_blocks": self._limiter.invalid_blocks,
            "closed": self._closed, **self._close_status,
        }

    def close(self) -> dict:
        if not self._closed:
            self._closed = True
            self._close_status = self.worker.close()
            self._sources.clear()
        return dict(self._close_status)
