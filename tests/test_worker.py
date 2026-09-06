import threading
import time

import numpy as np
import pytest

from tactical_speech_enhancement.config import Settings
from tactical_speech_enhancement.worker import InferenceJob, InferenceWorker, SynchronousWorker


class DelayedIdentity:
    def __init__(self):
        self.previous = np.zeros(256, dtype=np.float32)
        self.resets = 0

    def reset(self):
        self.previous.fill(0)
        self.resets += 1

    def process(self, block):
        result = self.previous.copy()
        self.previous = block.copy()
        return result


def job(sequence, *, generation=0, now=0):
    return InferenceJob(sequence, generation, np.full(256, sequence / 100, np.float32), now)


def wait_until(predicate, timeout=2):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.001)
    assert predicate(), "worker did not reach expected state"


def test_first_job_after_reset_cannot_forge_the_missing_previous_source():
    model = DelayedIdentity()
    worker = SynchronousWorker(model)
    worker.submit(job(9))
    assert worker.take(8, 0, 1) is None
    worker.submit(job(10))
    result = worker.take(9, 0, 1)
    np.testing.assert_array_equal(result.audio, np.full(256, 0.09, np.float32))
    assert result.input_sequence == 10 and result.source_sequence == 9
    assert model.resets == 1


def test_gap_and_generation_change_reset_state_on_the_execution_owner():
    model = DelayedIdentity()
    worker = SynchronousWorker(model)
    worker.submit(job(0))
    worker.submit(job(1))
    worker.submit(job(3))
    assert worker.take(2, 0, 1) is None
    worker.invalidate(1)
    assert model.resets == 2  # invalidation itself never enters the model
    worker.submit(job(4, generation=1))
    assert model.resets == 3
    assert worker.take(3, 1, 1) is None
    assert worker.summary()["sequence_gaps"] == 1


def test_result_memory_is_two_slots_and_metrics_are_bounded():
    worker = SynchronousWorker(DelayedIdentity(), Settings(telemetry_capacity=8))
    for i in range(40):
        worker.submit(job(i))
    stats = worker.summary()
    assert stats["processed"] == 40
    assert stats["max_results"] == stats["result_slots"] == 2
    assert stats["latency_sample_count"] == 8
    assert worker.take(1, 0, 1) is None
    assert worker.take(38, 0, 1).source_sequence == 38


@pytest.mark.parametrize("output", [
    np.full(256, np.nan), np.full(256, np.inf), np.zeros(255),
    np.zeros(256, dtype=int), np.full(256, 1e100),
])
def test_invalid_model_result_is_not_published(output):
    class BadModel(DelayedIdentity):
        def process(self, block):
            return output

    worker = SynchronousWorker(BadModel())
    worker.submit(job(0))
    worker.submit(job(1))
    assert worker.take(0, 0, 1) is None
    assert worker.summary()["model_errors"] == 2


def test_async_stall_replaces_pending_without_blocking_and_resets_after_gap():
    entered, release = threading.Event(), threading.Event()

    class BlockingModel(DelayedIdentity):
        def process(self, block):
            entered.set()
            release.wait(2)
            return super().process(block)

    # A controlled clock keeps this a synchronization test rather than a
    # scheduler-speed benchmark on whichever machine runs CI.
    model = BlockingModel()
    worker = InferenceWorker(model, clock=lambda: 0)
    try:
        worker.submit(job(0))
        assert entered.wait(1)
        worker.submit(job(1))
        worker.submit(job(2))
        stats = worker.summary()
        assert stats["pending"] == stats["running"] == 1
        assert stats["queue_replacements"] == 1
        release.set()
        wait_until(lambda: worker.summary().get("processed") == 2)
        assert worker.summary()["sequence_gaps"] == 1
        assert model.resets == 2
    finally:
        release.set()
        assert worker.close()["worker_stopped"]


def test_inflight_old_generation_cannot_republish_after_invalidation():
    entered, release = threading.Event(), threading.Event()

    class BlockingSecond(DelayedIdentity):
        def process(self, block):
            if block[0] > 0:
                entered.set()
                release.wait(2)
            return super().process(block)

    worker = InferenceWorker(BlockingSecond(), clock=lambda: 0)
    try:
        worker.submit(job(0))
        wait_until(lambda: worker.summary().get("processed") == 1)
        worker.submit(job(1))
        assert entered.wait(1)
        worker.invalidate(1)
        release.set()
        wait_until(lambda: worker.summary().get("processed") == 2)
        assert worker.take(0, 1, 1) is None
        assert worker.summary()["result_slots"] == 0
    finally:
        release.set()
        worker.close()


def test_expired_pending_job_is_skipped_before_entering_the_model():
    model = DelayedIdentity()
    worker = InferenceWorker(model, clock=lambda: 0.017)
    try:
        worker.submit(job(0, now=0))
        wait_until(lambda: worker.summary().get("expired_before_inference") == 1)
        assert model.resets == 0
        assert worker.summary().get("processed", 0) == 0
    finally:
        worker.close()


def test_close_reports_a_stuck_worker_instead_of_waiting_for_inference():
    entered, release = threading.Event(), threading.Event()

    class Stuck(DelayedIdentity):
        def process(self, block):
            entered.set()
            release.wait(2)
            return super().process(block)

    worker = InferenceWorker(Stuck(), clock=lambda: 0)
    worker.submit(job(0))
    assert entered.wait(1)
    try:
        start = time.monotonic()
        assert not worker.close(timeout=0)["worker_stopped"]
        assert time.monotonic() - start < 0.25
    finally:
        release.set()
        assert worker.close(timeout=1)["worker_stopped"]
