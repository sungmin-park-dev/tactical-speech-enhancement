"""Unit tests for the direct ONNX GTCRN demo/reference adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

from tactical_speech_enhancement.artifacts import default_model_path
from tactical_speech_enhancement.gtcrn import (
    EXPECTED_MODEL_METADATA,
    HOP_LENGTH,
    MODEL_INPUTS,
    MODEL_OUTPUTS,
    NUM_BINS,
    OFFICIAL_MODEL_SHA256,
    STATE_INPUT_NAMES,
    STATE_SHAPES,
    GtcrnFrameError,
    GtcrnModelError,
    GtcrnOnnxSpectralEngine,
    GtcrnStream,
    GtcrnStreamingEnhancer,
)


class IdentitySpectralEngine:
    def __init__(self) -> None:
        self.calls: List[np.ndarray] = []
        self.reset_count = 0

    def enhance(self, spectrum: np.ndarray) -> np.ndarray:
        self.calls.append(np.asarray(spectrum).copy())
        return np.asarray(spectrum).copy()

    def reset(self) -> None:
        self.reset_count += 1


class FakeNode:
    def __init__(self, name: str, shape: Tuple[int, ...]) -> None:
        self.name = name
        self.shape = list(shape)
        self.type = "tensor(float)"


class FakeModelMeta:
    def __init__(self, metadata: Dict[str, str]) -> None:
        self.custom_metadata_map = metadata


class FakeSession:
    def __init__(self, metadata: Dict[str, str] = None) -> None:
        self._metadata = dict(EXPECTED_MODEL_METADATA)
        if metadata is not None:
            self._metadata.update(metadata)
        self.state_feeds: List[Tuple[np.ndarray, ...]] = []

    def get_modelmeta(self) -> FakeModelMeta:
        return FakeModelMeta(self._metadata)

    def get_inputs(self) -> List[FakeNode]:
        return [FakeNode(name, shape) for name, shape in MODEL_INPUTS]

    def get_outputs(self) -> List[FakeNode]:
        return [FakeNode(name, shape) for name, shape in MODEL_OUTPUTS]

    def run(self, output_names: List[str], feeds: Dict[str, np.ndarray]) -> List[np.ndarray]:
        assert output_names == [name for name, _ in MODEL_OUTPUTS]
        states = tuple(feeds[name].copy() for name in STATE_INPUT_NAMES)
        self.state_feeds.append(states)
        return [feeds["mix"].copy()] + [state + 1.0 for state in states]


def test_periodic_sqrt_hann_wola_is_identity_with_one_hop_delay() -> None:
    rng = np.random.default_rng(7)
    first = rng.uniform(-0.8, 0.8, HOP_LENGTH).astype(np.float32)
    second = rng.uniform(-0.8, 0.8, HOP_LENGTH).astype(np.float32)
    third = rng.uniform(-0.8, 0.8, HOP_LENGTH).astype(np.float32)

    engine = IdentitySpectralEngine()
    stream = GtcrnStream(engine)

    np.testing.assert_array_equal(stream.process(first), np.zeros(HOP_LENGTH, dtype=np.float32))
    np.testing.assert_allclose(stream.process_hop(second), first, atol=2e-6)
    np.testing.assert_allclose(stream.process_hop(third), second, atol=2e-6)
    np.testing.assert_allclose(stream.flush(), third, atol=2e-6)

    assert stream.startup_delay_hops == 1
    assert stream.latency_samples == HOP_LENGTH
    assert len(engine.calls) == 4
    assert all(frame.shape == (NUM_BINS,) for frame in engine.calls)
    assert engine.reset_count == 1


def test_stream_reset_clears_wola_and_delegates_model_reset() -> None:
    engine = IdentitySpectralEngine()
    stream = GtcrnStream(engine)
    first = np.linspace(-0.5, 0.5, HOP_LENGTH, dtype=np.float32)
    second = first[::-1].copy()

    stream.process_hop(first)
    np.testing.assert_allclose(stream.process_hop(second), first, atol=2e-6)
    stream.reset()

    assert engine.reset_count == 1
    np.testing.assert_array_equal(
        stream.process_hop(second), np.zeros(HOP_LENGTH, dtype=np.float32)
    )
    np.testing.assert_allclose(stream.process_hop(first), second, atol=2e-6)


def test_onnx_contract_carries_exact_three_states_and_reset_zeros_them() -> None:
    session = FakeSession()
    engine = GtcrnOnnxSpectralEngine(session)
    spectrum = np.linspace(0.0, 1.0, NUM_BINS, dtype=np.float32).astype(np.complex64)

    np.testing.assert_array_equal(engine.enhance(spectrum), spectrum)
    engine.enhance(spectrum)
    engine.reset()
    engine.enhance(spectrum)

    assert engine.state_shapes == STATE_SHAPES
    assert len(session.state_feeds) == 3
    for state, shape in zip(session.state_feeds[0], STATE_SHAPES):
        assert state.shape == shape
        assert state.dtype == np.float32
        assert not np.any(state)
    for state in session.state_feeds[1]:
        np.testing.assert_array_equal(state, np.ones_like(state))
    for state in session.state_feeds[2]:
        assert not np.any(state)


def test_onnx_contract_rejects_wrong_metadata() -> None:
    with pytest.raises(GtcrnModelError, match="metadata 'hop_length' mismatch"):
        GtcrnOnnxSpectralEngine(FakeSession({"hop_length": "128"}))


@pytest.mark.parametrize(
    "bad_hop, message",
    [
        (np.zeros(HOP_LENGTH - 1, dtype=np.float32), "shape mismatch"),
        (
            np.full(HOP_LENGTH, np.nan, dtype=np.float32),
            "contains NaN or infinity",
        ),
    ],
)
def test_stream_rejects_bad_audio_hops(bad_hop: np.ndarray, message: str) -> None:
    stream = GtcrnStream(IdentitySpectralEngine())
    with pytest.raises(GtcrnFrameError, match=message):
        stream.process_hop(bad_hop)


def test_model_loader_rejects_unpinned_artifact_before_importing_ort(
    tmp_path: Path,
) -> None:
    wrong_model = tmp_path / "gtcrn_simple.onnx"
    wrong_model.write_bytes(b"not the official model")
    with pytest.raises(GtcrnModelError, match="size mismatch"):
        GtcrnOnnxSpectralEngine.from_model(wrong_model)


def test_stable_constructor_rejects_caller_selected_digest() -> None:
    with pytest.raises(GtcrnModelError, match="pinned official digest"):
        GtcrnStreamingEnhancer(
            Path("unused.onnx"), expected_sha256="0" * len(OFFICIAL_MODEL_SHA256)
        )


DEFAULT_REAL_MODEL = default_model_path()
REAL_MODEL = Path(os.environ.get("TSE_MODEL_PATH", str(DEFAULT_REAL_MODEL)))


@pytest.mark.skipif(not REAL_MODEL.is_file(), reason="pinned GTCRN artifact not present")
def test_real_model_contract_when_artifact_is_present() -> None:
    pytest.importorskip("onnxruntime")
    engine = GtcrnOnnxSpectralEngine.from_model(REAL_MODEL)
    output = engine.enhance(np.zeros(NUM_BINS, dtype=np.complex64))
    assert output.shape == (NUM_BINS,)
    assert np.all(np.isfinite(output))
    engine.reset()
