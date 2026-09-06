"""PortAudio adapter: capture/control and playout never execute inference or I/O."""

from __future__ import annotations

import ipaddress
import math
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .config import Settings
from .engine import TxController
from .gtcrn import OFFICIAL_MODEL_SHA256, GtcrnStreamingEnhancer
from .guard import PeakLimiter
from .network import UdpPeer


def audio_module():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError("Install the [audio] extra and the system PortAudio library") from exc
    return sd


def list_devices():
    return audio_module().query_devices()


def endpoint(value: str, *, remote: bool = False) -> tuple[str, int]:
    try:
        host, port_text = value.rsplit(":", 1)
        address = ipaddress.IPv4Address(host)
        port = int(port_text)
    except (ValueError, TypeError) as exc:
        raise ValueError("Use a numeric IPv4:port endpoint") from exc
    if (
        not 1 <= port <= 65535
        or address.is_multicast
        or address == ipaddress.IPv4Address("255.255.255.255")
        or (remote and address.is_unspecified)
    ):
        raise ValueError("Use a unicast IPv4 address and port 1..65535")
    return str(address), port


def run_peer(args, model_path: Path, settings: Settings) -> dict:
    from .cli import environment

    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        raise ValueError("duration must be finite and positive")
    bind, peer = endpoint(args.bind), endpoint(args.peer, remote=True)
    if bind == peer:
        raise ValueError("Local and remote endpoints must differ")
    sd = audio_module()
    sd.check_input_settings(device=args.input_device, channels=1, dtype="float32", samplerate=16000)
    sd.check_output_settings(
        device=args.output_device, channels=1, dtype="float32", samplerate=16000
    )
    controller = TxController(
        GtcrnStreamingEnhancer(model_path, expected_sha256=OFFICIAL_MODEL_SHA256), settings=settings
    )
    transport = None
    stop = threading.Event()
    callback_errors: list[str] = []
    stats = {"callbacks": 0, "audio_status_events": 0}
    callback_times = deque(maxlen=settings.telemetry_capacity)
    callback_total = 0.0
    callback_maximum = 0.0
    callback_misses = 0
    device_latency = None
    limiter = PeakLimiter(settings.ceiling_dbfs, settings.sample_rate, settings.limiter_release_ms)

    def callback(indata, outdata, frames, time_info, status):
        nonlocal callback_total, callback_maximum, callback_misses
        callback_started = time.perf_counter()
        del time_info
        outdata.fill(0)
        stats["callbacks"] += 1
        try:
            if frames != settings.frame_samples or indata.shape != (frames, 1):
                raise ValueError("Audio device changed the fixed frame/channel contract")
            now = time.monotonic()
            result = controller.process(indata[:, 0].copy(), now=now, overload=bool(status))
            if result.sequence >= 0:
                transport.submit(result.sequence, result.audio)
            received = transport.playout(now)
            if status:
                stats["audio_status_events"] += 1
                limiter.reset()
            else:
                outdata[:, 0] = limiter.process(received)
        except Exception as exc:
            if not callback_errors:
                callback_errors.append(type(exc).__name__ + ": " + str(exc))
            outdata.fill(0)
            stop.set()
        finally:
            elapsed = time.perf_counter() - callback_started
            callback_times.append(elapsed)
            callback_total += elapsed
            callback_maximum = max(callback_maximum, elapsed)
            callback_misses += elapsed >= settings.frame_seconds

    try:
        transport = UdpPeer(
            bind,
            peer,
            jitter_target_frames=settings.jitter_target_frames,
            jitter_max_frames=settings.jitter_max_frames,
            ceiling_dbfs=settings.ceiling_dbfs,
        )
        transport.start()
        with sd.Stream(
            samplerate=16000,
            blocksize=256,
            device=(args.input_device, args.output_device),
            channels=(1, 1),
            dtype="float32",
            latency="low",
            callback=callback,
        ) as stream:
            started = time.monotonic()
            device_latency = stream.latency
            while not stop.wait(0.05):
                if args.duration is not None and time.monotonic() - started >= args.duration:
                    break
                if not stream.active:
                    raise RuntimeError("Audio stream stopped unexpectedly")
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if isinstance(exc, getattr(sd, "PortAudioError", ())):
            raise RuntimeError(f"Audio device error: {exc}") from exc
        raise
    finally:
        try:
            if transport is not None:
                transport.close()
        finally:
            close_result = controller.close()
    if callback_errors:
        raise RuntimeError(callback_errors[0])
    if not close_result.get("worker_stopped", False):
        raise RuntimeError("Inference worker did not stop; exit this process before restarting")
    stats.update(
        {
            "callback_mean_ms": callback_total / max(1, stats["callbacks"]) * 1000,
            "callback_max_ms": callback_maximum * 1000,
            "callback_deadline_misses": callback_misses,
            "callback_recent_p99_ms": float(np.percentile(callback_times, 99) * 1000)
            if callback_times
            else 0,
            "callback_timing_samples": len(callback_times),
            "callback_timing_sample_capacity": settings.telemetry_capacity,
        }
    )
    return {
        **environment(),
        "mode": "live",
        "settings": settings.to_dict(),
        "audio": stats,
        "portaudio_reported_latency_seconds": None
        if device_latency is None
        else list(device_latency),
        "pipeline": controller.summary(),
        "shutdown": close_result,
        "network": {} if transport is None else transport.summary(),
        "note": "PortAudio latency is a device estimate, not measured end-to-end latency. No speech is recorded.",
    }
