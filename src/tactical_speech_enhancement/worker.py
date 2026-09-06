"""A single inference worker with one replaceable pending job and two results.

    Model calls never hold the capture-facing condition lock. A stuck native
    inference cannot be forcibly interrupted; close reports that condition and
    the daemon worker cannot delay application shutdown.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .config import Settings


class Enhancer(Protocol):
    def process(self, block: np.ndarray) -> np.ndarray: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class InferenceJob:
    input_sequence: int
    generation: int
    audio: np.ndarray
    submitted_at: float

    @property
    def source_sequence(self) -> int:
        return self.input_sequence - 1


@dataclass(frozen=True)
class InferenceResult:
    input_sequence: int
    source_sequence: int
    generation: int
    audio: np.ndarray
    submitted_at: float
    completed_at: float


class Worker(Protocol):
    def submit(self, job: InferenceJob) -> None: ...

    def invalidate(self, generation: int) -> None: ...

    def take(self, source_sequence: int, generation: int, now: float) -> InferenceResult | None: ...

    def summary(self) -> dict: ...

    def close(self, timeout: float = 0.05) -> dict: ...


class _WorkerCore:
    def __init__(self, enhancer: Enhancer, settings: Settings | None = None) -> None:
        if not callable(getattr(enhancer, "process", None)) or not callable(getattr(enhancer, "reset", None)):
            raise TypeError("enhancer must define process(block) and reset()")
        self.enhancer = enhancer
        self.settings = settings or Settings()
        self._condition = threading.Condition()
        self._generation = 0
        self._results: dict[int, InferenceResult] = {}
        self._pending: InferenceJob | None = None
        self._running = False
        self._closed = False
        self._last_sequence: int | None = None
        self._last_generation: int | None = None
        self._reset_needed = True
        self._consumed_through = -2
        self._counts: Counter = Counter()
        self._latencies: deque[float] = deque(maxlen=self.settings.telemetry_capacity)
        self._latency_total = 0.0
        self._latency_max = 0.0

    def invalidate(self, generation: int) -> None:
        with self._condition:
            if generation <= self._generation:
                return
            self._generation = generation
            self._counts["discarded_stale"] += len(self._results) + (self._pending is not None)
            self._results.clear()
            self._pending = None

    def take(self, source_sequence: int, generation: int, now: float) -> InferenceResult | None:
        with self._condition:
            for sequence in tuple(self._results):
                if sequence < source_sequence:
                    del self._results[sequence]
                    self._counts["discarded_stale"] += 1
            self._consumed_through = max(self._consumed_through, source_sequence)
            result = self._results.pop(source_sequence, None)
            if result is None:
                return None
            if result.generation != generation or result.completed_at > now:
                self._counts["discarded_stale"] += 1
                return None
            return result

    def _evaluate(self, job: InferenceJob, completed_clock: Callable[[], float]) -> None:
        # Only the one inference execution owner touches streaming model state.
        contiguous = (
            not self._reset_needed
            and self._last_sequence == job.input_sequence - 1
            and self._last_generation == job.generation
        )
        sequence_gap = self._last_sequence is not None and self._last_sequence != job.input_sequence - 1
        error = False
        output = None
        try:
            if not contiguous:
                self.enhancer.reset()
            value = np.asarray(self.enhancer.process(job.audio))
            if value.shape != (self.settings.frame_samples,) or value.dtype.kind != "f":
                raise ValueError("invalid model output format")
            if not np.isfinite(value).all() or np.max(np.abs(value)) > np.finfo(np.float32).max:
                raise ValueError("nonfinite or unrepresentable model output")
            output = value.astype(np.float32, copy=True)
        except Exception:
            # Error strings may contain local artifact paths, so counters are
            # sufficient for public runtime diagnostics.
            error = True
        finished = completed_clock()
        duration = max(0.0, finished - job.submitted_at)
        self._last_sequence = job.input_sequence
        self._last_generation = job.generation
        self._reset_needed = error
        with self._condition:
            self._counts["processed"] += 1
            self._counts["resets"] += not contiguous
            self._counts["sequence_gaps"] += sequence_gap
            self._counts["model_errors"] += error
            self._latencies.append(duration)
            self._latency_total += duration
            self._latency_max = max(self._latency_max, duration)
            deadline = job.submitted_at + self.settings.model_deadline_ms / 1000
            late = finished >= deadline
            self._counts["deadline_misses"] += late
            stale = (
                self._closed or job.generation != self._generation
                or job.source_sequence <= self._consumed_through
            )
            self._counts["discarded_stale"] += stale
            if error or late or stale or not contiguous:
                return
            self._results[job.source_sequence] = InferenceResult(
                input_sequence=job.input_sequence, source_sequence=job.source_sequence,
                generation=job.generation, audio=output,
                submitted_at=job.submitted_at, completed_at=finished,
            )
            while len(self._results) > 2:
                del self._results[min(self._results)]
                self._counts["results_evicted"] += 1
            self._counts["max_results"] = max(self._counts["max_results"], len(self._results))

    def summary(self) -> dict:
        with self._condition:
            counts = dict(self._counts)
            processed = self._counts["processed"]
            values = list(self._latencies)
            counts.update({
                "pending": int(self._pending is not None), "running": int(self._running),
                "result_slots": len(self._results), "generation": self._generation,
                "closed": self._closed, "latency_sample_count": len(values),
                "latency_sample_capacity": self.settings.telemetry_capacity,
                "latency_mean_ms": self._latency_total / processed * 1000 if processed else 0,
                "latency_max_ms": self._latency_max * 1000,
            })
        counts["latency_recent_p95_ms"] = float(np.percentile(values, 95) * 1000) if values else 0
        return counts


class InferenceWorker(_WorkerCore):
    """Run inference outside capture, including when the model fails to return."""

    def __init__(
        self, enhancer: Enhancer, settings: Settings | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(enhancer, settings)
        self._clock = clock
        self._thread = threading.Thread(target=self._run, name="tse-inference", daemon=True)
        self._thread.start()

    def submit(self, job: InferenceJob) -> None:
        with self._condition:
            if self._closed:
                return
            self._counts["submitted"] += 1
            if job.generation != self._generation:
                self._counts["discarded_stale"] += 1
                return
            if self._pending is not None:
                self._counts["queue_replacements"] += 1
            self._pending = job
            self._counts["max_pending"] = 1
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                job, self._pending = self._pending, None
                self._running = True
                self._counts["max_running"] = 1
            if self._clock() >= job.submitted_at + self.settings.model_deadline_ms / 1000:
                self._reset_needed = True
                with self._condition:
                    self._counts["expired_before_inference"] += 1
                    self._counts["deadline_misses"] += 1
                    self._running = False
                continue
            self._evaluate(job, self._clock)
            with self._condition:
                self._running = False

    def close(self, timeout: float = 0.05) -> dict:
        with self._condition:
            self._closed = True
            self._pending = None
            self._results.clear()
            self._condition.notify()
        self._thread.join(max(0, timeout))
        return {"worker_stopped": not self._thread.is_alive(), "worker_mode": "asynchronous"}


class SynchronousWorker(_WorkerCore):
    """Offline adapter for identical source tagging, without real-time claims.

    The model is called synchronously at submit and its virtual duration is zero.
    Never use this adapter in the live audio callback; use InferenceWorker there.
    """

    def submit(self, job: InferenceJob) -> None:
        with self._condition:
            if self._closed:
                return
            self._counts["submitted"] += 1
            if job.generation != self._generation:
                self._counts["discarded_stale"] += 1
                return
            self._running = True
            self._counts["max_running"] = 1
        self._evaluate(job, lambda: job.submitted_at)
        with self._condition:
            self._running = False

    def close(self, timeout: float = 0.05) -> dict:
        with self._condition:
            self._closed = True
            self._results.clear()
        return {"worker_stopped": True, "worker_mode": "offline_synchronous"}
