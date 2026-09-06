"""Command-line interface; no hardware is opened unless explicitly requested."""

from __future__ import annotations

import argparse
import json
import math
import platform
import tempfile
import time
import tomllib
from collections import deque
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf

from .artifacts import default_model_path, fetch, verify
from .config import Settings
from .gtcrn import OFFICIAL_MODEL_SHA256, GtcrnStreamingEnhancer


def load_settings(path: Path | None) -> Settings:
    if path is None:
        return Settings()
    with path.open("rb") as stream:
        return Settings(**tomllib.load(stream))


def environment() -> dict:
    return {
        "system": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "onnxruntime", "soundfile")},
        "evidence_scope": "local software execution; not a Raspberry Pi measurement",
    }


def write_report(report: dict, path: Path | None) -> None:
    data = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
    print(data, end="")


def process_file(input_path: Path, output_path: Path, model_path: Path, settings: Settings) -> dict:
    from .engine import TxController
    from .worker import SynchronousWorker

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files")
    samples, rate = sf.read(input_path, dtype="float32")
    if rate != settings.sample_rate or samples.ndim != 1 or not samples.size:
        raise ValueError("Input must be a nonempty mono 16000 Hz audio file")
    enhancer = GtcrnStreamingEnhancer(model_path, expected_sha256=OFFICIAL_MODEL_SHA256)
    worker = SynchronousWorker(enhancer, settings=settings)
    controller = TxController(enhancer, settings=settings, worker=worker)
    hop = settings.frame_samples
    count = math.ceil(len(samples) / hop)
    output = np.zeros(count * hop, dtype=np.float32)
    try:
        for seq in range(count + 2):
            block = np.zeros(hop, dtype=np.float32)
            piece = samples[seq * hop : (seq + 1) * hop]
            block[: len(piece)] = piece
            result = controller.process(block, now=seq * settings.frame_seconds)
            if 0 <= result.sequence < count:
                output[result.sequence * hop : (result.sequence + 1) * hop] = result.audio
    finally:
        controller.close()
    summary = controller.summary()
    ceiling = math.floor(32768 * 10 ** (settings.ceiling_dbfs / 20))
    pcm = np.clip(np.rint(output[: len(samples)] * 32768), -ceiling, ceiling).astype(np.int16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, suffix=".wav", delete=False
        ) as stream:
            temporary = Path(stream.name)
        sf.write(temporary, pcm, rate, subtype="PCM_16", format="WAV")
        temporary.replace(output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        **environment(),
        "mode": "offline",
        "settings": settings.to_dict(),
        "model": verify(model_path),
        "samples": len(samples),
        "sample_rate": rate,
        "scheduling_delay_removed_samples": 2 * hop,
        "note": "Deterministic offline processing; original startup protection remains. No real-time deadline claim.",
        "pipeline": summary,
    }


def benchmark(model_path: Path, frames: int, warmup: int, settings: Settings) -> dict:
    if frames < 1 or warmup < 0:
        raise ValueError("frames must be positive and warmup nonnegative")
    enhancer = GtcrnStreamingEnhancer(model_path, expected_sha256=OFFICIAL_MODEL_SHA256)
    block = (0.1 * np.sin(np.arange(256) * 0.11)).astype(np.float32)
    for _ in range(warmup):
        enhancer.process(block)
    recent: deque[float] = deque(maxlen=settings.telemetry_capacity)
    total = maximum = 0.0
    misses = 0
    for _ in range(frames):
        start = time.perf_counter()
        enhancer.process(block)
        elapsed = (time.perf_counter() - start) * 1000
        recent.append(elapsed)
        total += elapsed
        maximum = max(maximum, elapsed)
        misses += elapsed >= settings.model_deadline_ms
    values = np.asarray(recent)
    return {
        **environment(),
        "model": verify(model_path),
        "settings": settings.to_dict(),
        "frames": frames,
        "mean_ms": total / frames,
        "max_ms": maximum,
        "deadline_misses": misses,
        "deadline_ms": settings.model_deadline_ms,
        "model_rtf": total / (frames * settings.frame_ms),
        "recent_window_frames": len(values),
        "recent_window_p50_ms": float(np.percentile(values, 50)),
        "recent_window_p99_ms": float(np.percentile(values, 99)),
        "scope": "GTCRN + WOLA wall time only; device/network and protection timing excluded",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tse", description="Streaming speech enhancement and local-network audio"
    )
    parser.add_argument("--config", type=Path, help="TOML settings override")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="ONNX model (default: TSE_MODEL_PATH or user cache)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    model = commands.add_parser("model", help="download or verify the pinned model")
    model.add_argument("operation", choices=("fetch", "verify"))
    model.add_argument("--force", action="store_true")
    process = commands.add_parser("process", help="process a mono 16 kHz file offline")
    process.add_argument("input", type=Path)
    process.add_argument("output", type=Path)
    process.add_argument("--report", type=Path)
    commands.add_parser("devices", help="list PortAudio devices")
    bench = commands.add_parser("benchmark", help="measure local model inference")
    bench.add_argument("--frames", type=int, default=10000)
    bench.add_argument("--warmup", type=int, default=100)
    bench.add_argument("--report", type=Path)
    peer = commands.add_parser("peer", help="run two-way audio with one peer; requires a headset")
    peer.add_argument("--bind", required=True, help="local IPv4:port")
    peer.add_argument("--peer", required=True, help="remote IPv4:port")
    peer.add_argument("--input-device", type=int, required=True)
    peer.add_argument("--output-device", type=int, required=True)
    peer.add_argument("--duration", type=float)
    peer.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config)
        model_path = args.model_path or default_model_path()
        if args.command == "model":
            report = (
                fetch(model_path, force=args.force)
                if args.operation == "fetch"
                else verify(model_path)
            )
            if args.operation == "verify":
                GtcrnStreamingEnhancer(model_path, expected_sha256=OFFICIAL_MODEL_SHA256)
                report["onnx_contract"] = "passed"
            write_report(report, None)
        elif args.command == "process":
            if args.report and args.report.resolve() in (
                args.input.resolve(),
                args.output.resolve(),
            ):
                raise ValueError("Report must not overwrite input or output audio")
            write_report(process_file(args.input, args.output, model_path, settings), args.report)
        elif args.command == "benchmark":
            write_report(benchmark(model_path, args.frames, args.warmup, settings), args.report)
        elif args.command == "devices":
            from .live import list_devices

            print(list_devices())
        else:
            from .live import run_peer

            write_report(run_peer(args, model_path, settings), args.report)
        return 0
    except (ValueError, TypeError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"tse: {exc}\n")
