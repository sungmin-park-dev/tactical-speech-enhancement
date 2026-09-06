import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tactical_speech_enhancement.config import Settings
from tactical_speech_enhancement.engine import TxController
from tactical_speech_enhancement.worker import (
    InferenceResult,
    InferenceWorker,
    SynchronousWorker,
    _WorkerCore,
)


def frame(value=0.1):
    return np.full(256, value, np.float32)


class DelayedModel:
    def __init__(self, scale=0.5):
        self.previous = frame(0)
        self.scale = scale
        self.bad = False

    def reset(self):
        self.previous = frame(0)

    def process(self, block):
        if self.bad:
            raise RuntimeError("injected inference failure")
        output = self.previous * self.scale
        self.previous = block.copy()
        return output


class VirtualWorker(_WorkerCore):
    """Discrete-event execution of the same publication rules and queue limits."""

    def __init__(self, model, delay, settings=None):
        super().__init__(model, settings)
        self.delay = delay
        self.active = None
        self.ready_at = math.inf

    def submit(self, job):
        self._counts["submitted"] += 1
        if self.active is None:
            self.active = job
            self._running = True
            self.ready_at = job.submitted_at + self.delay
        else:
            self._counts["queue_replacements"] += self._pending is not None
            self._pending = job
            self._counts["max_pending"] = 1

    def advance(self, now):
        while self.active is not None and self.ready_at <= now:
            finished = self.ready_at
            self._evaluate(self.active, lambda: finished)
            self.active, self._pending = self._pending, None
            self._running = self.active is not None
            self.ready_at = finished + self.delay if self.active else math.inf

    def close(self, timeout=0.05):
        self._closed = True
        return {"worker_stopped": self.active is None or self.delay != math.inf}


def offline_controller(model=None, settings=None):
    model = model or DelayedModel()
    settings = settings or Settings()
    return TxController(model, settings, SynchronousWorker(model, settings))


def test_dry_and_model_share_exactly_two_capture_frames_of_alignment():
    controller = offline_controller(DelayedModel(scale=1))
    outputs = [controller.process(frame(n / 1000), now=n * 0.016) for n in range(50)]
    assert [o.sequence for o in outputs] == list(range(-2, 48))
    for n in range(20, 50):
        np.testing.assert_allclose(outputs[n].audio, frame((n - 2) / 1000), atol=1e-8)
        assert outputs[n].diagnostics["route"] == "model"
    assert controller.summary()["max_alignment_frames"] == 2
    assert controller.summary()["alignment_delay_ms"] == 32
    controller.close()


def test_model_only_failure_switches_to_aligned_dry_without_reentering_guard():
    model = DelayedModel()
    controller = offline_controller(model)
    for n in range(20):
        controller.process(frame(), now=n * 0.016)
    model.bad = True
    controller.process(frame(0.2), now=0.320)
    failed = controller.process(frame(0.3), now=0.336)
    assert failed.sequence == 19 and failed.diagnostics["route"] == "dry"
    np.testing.assert_array_equal(failed.audio, frame(0.1))
    assert controller.guard.gain == 1 and controller.guard.anomalies == 0
    model.bad = False
    mixes = []
    for n in range(22, 34):
        result = controller.process(frame(), now=n * 0.016)
        mixes.append(result.diagnostics["model_mix"])
    assert 0.25 in mixes and 0.5 in mixes and 0.75 in mixes and mixes[-1] == 1
    controller.close()


@pytest.mark.parametrize("delay", [0.008, 0.015])
def test_virtual_on_time_inference_has_no_steady_state_misses(delay):
    model = DelayedModel()
    worker = VirtualWorker(model, delay)
    controller = TxController(model, worker=worker)
    for n in range(80):
        now = n * 0.016
        worker.advance(now)
        output = controller.process(frame(), now=now)
        if n >= 20:
            assert output.diagnostics["route"] == "model"
            np.testing.assert_array_equal(output.audio, frame(0.05))
    assert worker.summary().get("deadline_misses", 0) == 0
    controller.close()


@pytest.mark.parametrize("delay", [0.016, 0.017, 0.200, math.inf])
def test_virtual_late_or_stuck_inference_never_extends_output_schedule(delay):
    model = DelayedModel()
    worker = VirtualWorker(model, delay)
    controller = TxController(model, worker=worker)
    seen = []
    for n in range(80):
        now = n * 0.016
        worker.advance(now)
        output = controller.process(frame(0.9 if n in (30, 33) else 0.1), now=now)
        seen.append(output.sequence)
        if 20 <= n < 30 or n >= 55:
            assert output.diagnostics["route"] == "dry"
            np.testing.assert_array_equal(output.audio, frame())
        if 30 <= n <= 43:
            assert not output.audio.any()
    assert seen == list(range(-2, 78))
    assert worker.summary()["pending"] <= 1
    assert worker.summary()["result_slots"] <= 2
    assert controller.summary()["alignment_frames"] == 2
    controller.close()


@pytest.mark.parametrize("offset", [0, 127, 255])
def test_overlap_impulse_invalidates_buffered_audio_before_result_selection(offset):
    model = DelayedModel()
    controller = offline_controller(model)
    for n in range(24):
        controller.process(frame(), now=n * 0.016)
    impulse = frame()
    impulse[offset] = 0.99
    outputs = [controller.process(impulse, now=24 * 0.016)]
    for n in range(25, 45):
        outputs.append(controller.process(frame(), now=n * 0.016))
    # Capture 24 immediately cancels older aligned output, and source 24
    # stays muted when its scheduled output arrives two captures later.
    assert outputs[0].sequence == 22 and not outputs[0].audio.any()
    assert outputs[2].sequence == 24 and not outputs[2].audio.any()
    assert outputs[2].diagnostics["source_muted"]
    assert outputs[-1].diagnostics["source_gain_end"] == 1
    assert controller.guard.generation == 1
    controller.close()


def test_late_wrong_generation_result_cannot_open_guard_or_replay_bad_source():
    class ForgingWorker(SynchronousWorker):
        def take(self, sequence, generation, now):
            return InferenceResult(sequence + 1, sequence, generation - 1, frame(0.5), now - 0.016, now - 0.010)

    model = DelayedModel()
    controller = TxController(model, worker=ForgingWorker(model))
    for n in range(20):
        controller.process(frame(), now=n * 0.016)
    shock = controller.process(frame(0.9), now=0.320)
    assert not shock.audio.any()
    for n in range(21, 31):
        output = controller.process(frame(), now=n * 0.016)
        assert not output.audio.any()
    assert controller.guard.generation == 1
    controller.close()


def test_malformed_input_never_enters_inference_or_dry_bypass():
    controller = offline_controller()
    for n in range(20):
        controller.process(frame(), now=n * 0.016)
    before = controller.worker.summary()["submitted"]
    malformed = controller.process(np.full(256, np.nan), now=0.320)
    assert controller.worker.summary()["submitted"] == before
    assert not malformed.audio.any()
    controller.process(frame(), now=0.336)
    source = controller.process(frame(), now=0.352)
    assert source.sequence == 20 and not source.audio.any()
    controller.close()


def test_actual_stalled_worker_does_not_block_capture_or_protection_on_full_queue():
    entered, release = threading.Event(), threading.Event()

    class Stuck(DelayedModel):
        def process(self, block):
            entered.set()
            release.wait(3)
            return super().process(block)

    model = Stuck()
    worker = InferenceWorker(model, clock=lambda: 0)
    controller = TxController(model, worker=worker)
    controller.process(frame(), now=0)
    assert entered.wait(1)
    try:
        start = time.monotonic()
        for n in range(1, 24):
            output = controller.process(frame(0.9 if n == 20 else 0.1), now=n * 0.016)
            assert output.sequence == n - 2
            if n >= 20:
                assert not output.audio.any()
        assert time.monotonic() - start < 1
        assert controller.guard.generation == 1
        assert worker.summary()["max_pending"] == worker.summary()["running"] == 1
        assert not controller.close()["worker_stopped"]
    finally:
        release.set()
        assert worker.close(timeout=1)["worker_stopped"]


def test_capture_telemetry_is_bounded_and_close_is_idempotent():
    controller = offline_controller(settings=Settings(telemetry_capacity=8))
    for n in range(40):
        controller.process(frame(), now=n * 0.016)
    summary = controller.summary()
    assert summary["frames"] == 40 and summary["capture_latency_sample_count"] == 8
    assert summary["capture_processing_max_ms"] >= summary["capture_processing_mean_ms"] >= 0
    assert controller.close() == controller.close()
    with pytest.raises(RuntimeError):
        controller.process(frame())


def test_invalid_capture_timestamps_are_rejected():
    controller = offline_controller()
    controller.process(frame(), now=1)
    with pytest.raises(ValueError):
        controller.process(frame(), now=0)
    with pytest.raises(ValueError):
        controller.process(frame(), now=math.nan)
    controller.close()


@pytest.mark.parametrize("placement", ["interior", "boundary", "sustained_and_repeated"])
@pytest.mark.parametrize("model_fails", [False, True])
def test_public_speech_with_impulses_has_no_dry_leakage_and_bounded_recovery(placement, model_fails):
    speech, rate = sf.read(
        Path(__file__).parent / "fixtures" / "speech" / "speaker-0061.wav", dtype="float32",
    )
    assert rate == 16000 and speech.shape == (160000,)
    speech *= np.float32(0.55 / np.max(np.abs(speech)))
    # Whole-frame silence slightly longer than one second finishes startup
    # protection while preserving the source indices used below.
    audio = np.concatenate((np.zeros(16384, np.float32), speech, np.zeros(8192, np.float32)))
    first_bad = 64 + 120
    if placement == "interior":
        audio[first_bad * 256 + 83:first_bad * 256 + 86] = [0.99, -0.99, 0.99]
        bad_sources = {first_bad}
    elif placement == "boundary":
        # Two rail samples straddle capture frames; both original intervals
        # must remain silent even though only the second completes saturation.
        audio[(first_bad + 1) * 256 - 1:(first_bad + 1) * 256 + 1] = [0.99, -0.99]
        bad_sources = {first_bad, first_bad + 1}
    else:
        audio[first_bad * 256:(first_bad + 6) * 256] = np.float32(0.985)
        # Re-impact during the first recovery ramp restarts protection.
        repeated = first_bad + 16
        audio[repeated * 256 + 40:repeated * 256 + 43] = [1.0, -1.0, 1.0]
        bad_sources = set(range(first_bad, first_bad + 6)) | {repeated}
    final_bad = max(bad_sources)
    audio = np.pad(audio, (0, (-len(audio)) % 256))
    model = DelayedModel(scale=1)
    model.bad = model_fails
    controller = offline_controller(model)
    source_outputs = {}
    generations = []
    try:
        for sequence, block in enumerate(audio.reshape(-1, 256)):
            result = controller.process(block, now=sequence * 0.016)
            if first_bad - 2 <= result.sequence <= final_bad + 18:
                source_outputs[result.sequence] = result
            if sequence in bad_sources:
                assert result.diagnostics["input_unsafe"]
                assert not result.audio.any()  # Cancels already-buffered output immediately.
                generations.append(result.diagnostics["generation"])
        for source in bad_sources:
            assert source_outputs[source].diagnostics["source_muted"]
            assert np.count_nonzero(source_outputs[source].audio) == 0
        for source in range(final_bad + 1, final_bad + 9):
            assert np.count_nonzero(source_outputs[source].audio) == 0
        assert source_outputs[final_bad + 16].diagnostics["source_gain_end"] == 1
        assert source_outputs[final_bad + 18].diagnostics["source_gain_end"] == 1
        assert (16 * 256 / rate) * 1000 == 256  # Recovery bound on original-source time.
        assert len(set(generations)) == len(bad_sources)
        summary = controller.summary()
        assert summary["guard"]["anomalies"] == len(bad_sources)
        assert summary["guard"]["gain"] == 1
        assert summary["worker"]["submitted"] == len(audio) // 256 - len(bad_sources)
    finally:
        controller.close()
