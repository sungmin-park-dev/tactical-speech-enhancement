"""Integrity-pinned GTCRN streaming adapter with a one-hop WOLA delay.

NumPy and ONNX Runtime may allocate; this is a Python software reference,
not a hard real-time implementation. Inference runs outside the audio callback.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Protocol, Sequence, Tuple

import numpy as np

SAMPLE_RATE = 16_000
N_FFT = 512
WINDOW_LENGTH = 512
HOP_LENGTH = 256
NUM_BINS = N_FFT // 2 + 1

OFFICIAL_MODEL_SIZE_BYTES = 535_638
OFFICIAL_MODEL_SHA256 = "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534"

EXPECTED_MODEL_METADATA: Mapping[str, str] = {
    "model_type": "gtcrn",
    "comment": "gtcrn_simple",
    "version": "1",
    "sample_rate": "16000",
    "model_url": (
        "https://github.com/Xiaobin-Rong/gtcrn/blob/main/stream/onnx_models/gtcrn_simple.onnx"
    ),
    "maintainer": "k2-fsa",
    "comment2": "Please see also https://github.com/Xiaobin-Rong/gtcrn",
    "conv_cache_shape": "2,1,16,16,33",
    "tra_cache_shape": "2,3,1,1,16",
    "inter_cache_shape": "2,1,33,16",
    "n_fft": "512",
    "hop_length": "256",
    "window_length": "512",
    "window_type": "hann_sqrt",
}

MODEL_INPUTS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("mix", (1, NUM_BINS, 1, 2)),
    ("conv_cache", (2, 1, 16, 16, 33)),
    ("tra_cache", (2, 3, 1, 1, 16)),
    ("inter_cache", (2, 1, 33, 16)),
)
MODEL_OUTPUTS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("enh", (1, NUM_BINS, 1, 2)),
    ("conv_cache_out", (2, 1, 16, 16, 33)),
    ("tra_cache_out", (2, 3, 1, 1, 16)),
    ("inter_cache_out", (2, 1, 33, 16)),
)
STATE_INPUT_NAMES = tuple(item[0] for item in MODEL_INPUTS[1:])
STATE_SHAPES = tuple(item[1] for item in MODEL_INPUTS[1:])
OUTPUT_NAMES = tuple(item[0] for item in MODEL_OUTPUTS)


class GtcrnError(RuntimeError):
    """Base class for GTCRN adapter failures."""


class GtcrnModelError(GtcrnError):
    """The model artifact or ONNX session violates the pinned contract."""


class GtcrnFrameError(GtcrnError, ValueError):
    """An audio hop or spectral frame is malformed."""


class SpectralEngine(Protocol):
    """Minimal injectable spectral interface used by :class:`GtcrnStream`."""

    def enhance(self, spectrum: np.ndarray) -> np.ndarray:
        """Return an enhanced one-sided complex spectrum of shape ``(257,)``."""

    def reset(self) -> None:
        """Reset all model-specific streaming state."""


class PassthroughSpectralEngine:
    """Identity engine for WOLA diagnostics and deterministic tests."""

    def enhance(self, spectrum: np.ndarray) -> np.ndarray:
        return _require_spectrum(spectrum, "passthrough input spectrum").copy()

    def reset(self) -> None:
        return None


def validate_official_model(path: Path) -> Path:
    """Validate the exact sherpa-distributed GTCRN model artifact."""

    model_path = Path(path)
    if not model_path.is_file():
        raise GtcrnModelError("GTCRN model file does not exist: %s" % model_path)

    size = model_path.stat().st_size
    if size != OFFICIAL_MODEL_SIZE_BYTES:
        raise GtcrnModelError(
            "GTCRN model size mismatch: expected %d bytes, got %d bytes (%s)"
            % (OFFICIAL_MODEL_SIZE_BYTES, size, model_path)
        )

    digest = hashlib.sha256()
    with model_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != OFFICIAL_MODEL_SHA256:
        raise GtcrnModelError(
            "GTCRN model SHA-256 mismatch: expected %s, got %s (%s)"
            % (OFFICIAL_MODEL_SHA256, actual_sha256, model_path)
        )

    return model_path


def periodic_sqrt_hann(length: int = WINDOW_LENGTH) -> np.ndarray:
    """Return PyTorch-compatible periodic square-root Hann coefficients."""

    if length <= 0:
        raise ValueError("window length must be positive")
    indices = np.arange(length, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * indices / float(length))
    return np.sqrt(hann).astype(np.float32)


def _normalise_node_shape(shape: Iterable[Any], label: str) -> Tuple[int, ...]:
    values = []
    for value in shape:
        if isinstance(value, (int, np.integer)):
            values.append(int(value))
        else:
            raise GtcrnModelError("%s must have a static integer shape; got %r" % (label, shape))
    return tuple(values)


def _validate_nodes(
    actual_nodes: Sequence[Any],
    expected_nodes: Sequence[Tuple[str, Tuple[int, ...]]],
    kind: str,
) -> None:
    if len(actual_nodes) != len(expected_nodes):
        raise GtcrnModelError(
            "GTCRN %s count mismatch: expected %d, got %d"
            % (kind, len(expected_nodes), len(actual_nodes))
        )

    for index, (actual, expected) in enumerate(zip(actual_nodes, expected_nodes)):
        expected_name, expected_shape = expected
        actual_name = getattr(actual, "name", None)
        actual_type = getattr(actual, "type", None)
        actual_shape = _normalise_node_shape(
            getattr(actual, "shape", ()), "%s[%d] %s" % (kind, index, actual_name)
        )
        if actual_name != expected_name:
            raise GtcrnModelError(
                "GTCRN %s[%d] name mismatch: expected %r, got %r"
                % (kind, index, expected_name, actual_name)
            )
        if actual_type != "tensor(float)":
            raise GtcrnModelError(
                "GTCRN %s %r type mismatch: expected 'tensor(float)', got %r"
                % (kind, expected_name, actual_type)
            )
        if actual_shape != expected_shape:
            raise GtcrnModelError(
                "GTCRN %s %r shape mismatch: expected %r, got %r"
                % (kind, expected_name, expected_shape, actual_shape)
            )


def _require_float32_tensor(value: Any, expected_shape: Tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != expected_shape:
        raise GtcrnModelError(
            "%s shape mismatch: expected %r, got %r" % (label, expected_shape, array.shape)
        )
    if array.dtype != np.float32:
        raise GtcrnModelError("%s dtype mismatch: expected float32, got %s" % (label, array.dtype))
    if not np.all(np.isfinite(array)):
        raise GtcrnModelError("%s contains NaN or infinity" % label)
    return array


def _require_spectrum(value: Any, label: str) -> np.ndarray:
    spectrum = np.asarray(value)
    if spectrum.shape != (NUM_BINS,):
        raise GtcrnFrameError(
            "%s shape mismatch: expected (%d,), got %r" % (label, NUM_BINS, spectrum.shape)
        )
    if not np.iscomplexobj(spectrum):
        raise GtcrnFrameError("%s must be a complex spectrum" % label)
    if not np.all(np.isfinite(spectrum.real)) or not np.all(np.isfinite(spectrum.imag)):
        raise GtcrnFrameError("%s contains NaN or infinity" % label)
    return spectrum


class GtcrnOnnxSpectralEngine:
    """Stateful ONNX spectral engine for the pinned four-input GTCRN graph.

    ``session`` injection keeps model-contract and recurrent-state behavior
    unit-testable without importing ONNX Runtime.  Normal application code
    should construct the engine with :meth:`from_model`, which validates the
    official file size and SHA-256 before lazily importing ONNX Runtime.
    """

    def __init__(self, session: Any):
        self._session = session
        self._validate_session_contract()
        self._states: Tuple[np.ndarray, ...] = ()
        self.reset()

    @classmethod
    def from_model(cls, model_path: Path, num_threads: int = 1) -> "GtcrnOnnxSpectralEngine":
        model_path = validate_official_model(Path(model_path))
        if not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer")

        try:
            ort = importlib.import_module("onnxruntime")
        except ModuleNotFoundError as exc:
            raise GtcrnModelError(
                "onnxruntime is required to load GTCRN; install the project dependencies"
            ) from exc

        try:
            ort.disable_telemetry_events()
            providers = ort.get_available_providers()
            if "CPUExecutionProvider" not in providers:
                raise GtcrnModelError(
                    "onnxruntime does not provide CPUExecutionProvider; available: %r" % providers
                )
            options = ort.SessionOptions()
            options.intra_op_num_threads = num_threads
            options.inter_op_num_threads = 1
            if hasattr(ort, "ExecutionMode"):
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except GtcrnModelError:
            raise
        except Exception as exc:
            raise GtcrnModelError(
                "failed to create GTCRN ONNX Runtime session for %s: %s" % (model_path, exc)
            ) from exc
        return cls(session)

    def _validate_session_contract(self) -> None:
        try:
            model_meta = self._session.get_modelmeta()
            metadata: Dict[str, str] = dict(model_meta.custom_metadata_map)
            actual_inputs = self._session.get_inputs()
            actual_outputs = self._session.get_outputs()
        except Exception as exc:
            raise GtcrnModelError(
                "unable to inspect GTCRN ONNX session metadata/I/O: %s" % exc
            ) from exc

        for key, expected in EXPECTED_MODEL_METADATA.items():
            actual = metadata.get(key)
            if actual != expected:
                raise GtcrnModelError(
                    "GTCRN metadata %r mismatch: expected %r, got %r" % (key, expected, actual)
                )

        _validate_nodes(actual_inputs, MODEL_INPUTS, "input")
        _validate_nodes(actual_outputs, MODEL_OUTPUTS, "output")

    def reset(self) -> None:
        self._states = tuple(np.zeros(shape, dtype=np.float32) for shape in STATE_SHAPES)

    @property
    def state_shapes(self) -> Tuple[Tuple[int, ...], ...]:
        return STATE_SHAPES

    def enhance(self, spectrum: np.ndarray) -> np.ndarray:
        spectrum = _require_spectrum(spectrum, "GTCRN input spectrum")

        mix = np.empty(MODEL_INPUTS[0][1], dtype=np.float32)
        mix[0, :, 0, 0] = spectrum.real
        mix[0, :, 0, 1] = spectrum.imag
        if not np.all(np.isfinite(mix)):
            raise GtcrnFrameError(
                "GTCRN input spectrum overflows float32 or contains non-finite data"
            )

        feeds: Dict[str, np.ndarray] = {"mix": mix}
        feeds.update(zip(STATE_INPUT_NAMES, self._states))
        try:
            outputs = self._session.run(list(OUTPUT_NAMES), feeds)
        except Exception as exc:
            raise GtcrnModelError("GTCRN ONNX inference failed: %s" % exc) from exc

        if len(outputs) != len(MODEL_OUTPUTS):
            raise GtcrnModelError(
                "GTCRN inference returned %d outputs; expected %d"
                % (len(outputs), len(MODEL_OUTPUTS))
            )

        enhanced = _require_float32_tensor(
            outputs[0], MODEL_OUTPUTS[0][1], "GTCRN enhanced spectrum"
        )
        next_states = tuple(
            _require_float32_tensor(value, shape, "GTCRN state %s" % name)
            for value, (name, shape) in zip(outputs[1:], MODEL_OUTPUTS[1:])
        )
        self._states = next_states

        result = enhanced[0, :, 0, 0] + 1j * enhanced[0, :, 0, 1]
        return _require_spectrum(result, "GTCRN output spectrum")


class GtcrnStream:
    """16 kHz/256-sample GTCRN stream with periodic sqrt-Hann WOLA.

    Each call returns exactly one hop.  The first call returns a zero hop;
    subsequent calls return enhanced audio with one-hop stream delay.  Use
    :meth:`flush` at end-of-stream to obtain the final pending hop.
    """

    latency_samples = HOP_LENGTH

    def __init__(self, spectral_engine: SpectralEngine):
        if not callable(getattr(spectral_engine, "enhance", None)):
            raise TypeError("spectral_engine must define enhance(spectrum)")
        if not callable(getattr(spectral_engine, "reset", None)):
            raise TypeError("spectral_engine must define reset()")

        self._engine = spectral_engine
        self._window = periodic_sqrt_hann(WINDOW_LENGTH)
        self._analysis = np.zeros(WINDOW_LENGTH, dtype=np.float32)
        self._overlap_add = np.zeros(WINDOW_LENGTH, dtype=np.float32)
        self._started = False

    @classmethod
    def from_model(cls, model_path: Path, num_threads: int = 1) -> "GtcrnStream":
        return cls(GtcrnOnnxSpectralEngine.from_model(model_path, num_threads))

    @property
    def startup_delay_hops(self) -> int:
        return 1

    def reset(self) -> None:
        self._analysis.fill(0.0)
        self._overlap_add.fill(0.0)
        self._started = False
        self._engine.reset()

    def process_hop(self, samples: np.ndarray) -> np.ndarray:
        hop = np.asarray(samples)
        if hop.shape != (HOP_LENGTH,):
            raise GtcrnFrameError(
                "audio hop shape mismatch: expected (%d,), got %r" % (HOP_LENGTH, hop.shape)
            )
        if np.iscomplexobj(hop):
            raise GtcrnFrameError("audio hop must contain real-valued samples")
        try:
            hop = hop.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise GtcrnFrameError("audio hop must be numeric float samples") from exc
        if not np.all(np.isfinite(hop)):
            raise GtcrnFrameError("audio hop contains NaN or infinity")

        self._analysis[:-HOP_LENGTH] = self._analysis[HOP_LENGTH:]
        self._analysis[-HOP_LENGTH:] = hop

        analysis_frame = self._analysis * self._window
        spectrum = np.fft.rfft(analysis_frame, n=N_FFT)
        enhanced = _require_spectrum(self._engine.enhance(spectrum), "spectral engine output")
        synthesis_frame = np.fft.irfft(enhanced, n=N_FFT)
        synthesis_frame = synthesis_frame.astype(np.float32, copy=False)
        if not np.all(np.isfinite(synthesis_frame)):
            raise GtcrnFrameError("inverse FFT produced NaN or infinity")

        self._overlap_add[:-HOP_LENGTH] = self._overlap_add[HOP_LENGTH:]
        self._overlap_add[-HOP_LENGTH:] = 0.0
        self._overlap_add += synthesis_frame * self._window

        if not self._started:
            self._started = True
            return np.zeros(HOP_LENGTH, dtype=np.float32)
        return self._overlap_add[:HOP_LENGTH].copy()

    def process(self, block: np.ndarray) -> np.ndarray:
        """Pipeline-compatible alias for :meth:`process_hop`."""

        return self.process_hop(block)

    def flush(self) -> np.ndarray:
        """Return the final pending hop and reset spectral/WOLA state."""

        if not self._started:
            self.reset()
            return np.empty(0, dtype=np.float32)
        tail = self.process_hop(np.zeros(HOP_LENGTH, dtype=np.float32))
        self.reset()
        return tail


class GtcrnStreamingEnhancer(GtcrnStream):
    """Stable application-facing constructor for the pinned GTCRN artifact.

    ``expected_sha256`` is supplied by the artifact manifest and must match
    the independently pinned official digest in this module.  This prevents a
    caller-controlled digest from silently blessing a different model.
    """

    sample_rate = SAMPLE_RATE
    hop_length = HOP_LENGTH

    def __init__(self, model_path: Path, expected_sha256: str, num_threads: int = 1) -> None:
        if not isinstance(expected_sha256, str):
            raise GtcrnModelError("expected_sha256 must be a hexadecimal string")
        normalised_sha256 = expected_sha256.strip().lower()
        if normalised_sha256 != OFFICIAL_MODEL_SHA256:
            raise GtcrnModelError(
                "manifest GTCRN SHA-256 does not match the pinned official digest: "
                "expected %s, got %s" % (OFFICIAL_MODEL_SHA256, normalised_sha256)
            )

        self.model_path = Path(model_path)
        self.expected_sha256 = normalised_sha256
        engine = GtcrnOnnxSpectralEngine.from_model(self.model_path, num_threads=num_threads)
        super().__init__(engine)


__all__ = [
    "EXPECTED_MODEL_METADATA",
    "GtcrnError",
    "GtcrnFrameError",
    "GtcrnModelError",
    "GtcrnOnnxSpectralEngine",
    "GtcrnStream",
    "GtcrnStreamingEnhancer",
    "HOP_LENGTH",
    "MODEL_INPUTS",
    "MODEL_OUTPUTS",
    "N_FFT",
    "NUM_BINS",
    "OFFICIAL_MODEL_SHA256",
    "OFFICIAL_MODEL_SIZE_BYTES",
    "PassthroughSpectralEngine",
    "SAMPLE_RATE",
    "STATE_SHAPES",
    "WINDOW_LENGTH",
    "periodic_sqrt_hann",
    "validate_official_model",
]
