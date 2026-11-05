"""Run the repository's unmodified `run_peer` with a fake audio device and telemetry.

Instrumentation is attached from outside (subclasses and module attribute
injection); repository source files are not changed.

Per-second CSV columns: wall time, per-thread CPU %, callback timing, inference
latency, jitter-buffer level, counters. Per-frame logs: capture time by source
sequence (sender side) and playout time by sequence (receiver side), so the
software path latency can be computed across two processes on one host
(shared CLOCK_MONOTONIC).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fake_sd  # noqa: E402

sys.modules["sounddevice"] = fake_sd

from tactical_speech_enhancement import engine as tse_engine  # noqa: E402
from tactical_speech_enhancement import live as tse_live  # noqa: E402
from tactical_speech_enhancement import network as tse_network  # noqa: E402
from tactical_speech_enhancement.config import Settings  # noqa: E402

CLK_TCK = os.sysconf("SC_CLK_TCK")
REGISTRY: dict = {}


class InstrumentedController(tse_engine.TxController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        REGISTRY["controller"] = self


class InstrumentedPeer(tse_network.UdpPeer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        REGISTRY["peer"] = self
        self.play_log = []      # (now, played_seq or -1, level_before, level_after)
        self.submit_log = []    # (now, seq)

    def submit(self, sequence, audio):
        self.submit_log.append((time.monotonic(), sequence))
        return super().submit(sequence, audio)

    def playout(self, now=None):
        jb = self._jitter
        with self._jitter_lock:
            played_before = jb._counters["played"]
            level_before = len(jb._frames)
        out = super().playout(now)
        with self._jitter_lock:
            played_after = jb._counters["played"]
            level_after = len(jb._frames)
            seq = jb._next_sequence - 1 if (played_after > played_before and jb._next_sequence is not None) else -1
        self.play_log.append((now if now is not None else time.monotonic(), seq, level_before, level_after))
        return out


def thread_cpu_ticks() -> dict:
    result = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/stat") as f:
                parts = f.read().rsplit(")", 1)[1].split()
            result[int(tid)] = int(parts[11]) + int(parts[12])  # utime + stime
        except (FileNotFoundError, ProcessLookupError, IndexError):
            pass
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bind", required=True)
    ap.add_argument("--peer", required=True)
    ap.add_argument("--ppm", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--input-gain", type=float, default=1.0)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--buffer-periods", type=int, default=3)
    ap.add_argument("--infer-extra-ms", type=float, default=0.0,
                    help="extra time added to each model hop to emulate a slower CPU")
    ap.add_argument("--infer-extra-mode", choices=["busy", "sleep"], default="busy")
    ap.add_argument("--conceal-frames", type=int, default=0)
    ap.add_argument("--protection-mode", choices=["mute", "duck"], default="mute")
    args = ap.parse_args()
    if args.infer_extra_ms > 0:
        base_cls = tse_live.GtcrnStreamingEnhancer
        extra = args.infer_extra_ms / 1000.0

        class SlowEnhancer(base_cls):
            def process_hop(self, samples):
                out = super().process_hop(samples)
                if args.infer_extra_mode == "sleep":   # native op that releases the GIL
                    time.sleep(extra)
                else:                                  # Python-level compute holding the GIL
                    end = time.perf_counter() + extra
                    while time.perf_counter() < end:
                        pass
                return out

        tse_live.GtcrnStreamingEnhancer = SlowEnhancer
    args.outdir.mkdir(parents=True, exist_ok=True)

    audio, rate = sf.read(args.input, dtype="float32")
    assert rate == 16000 and audio.ndim == 1
    audio = (audio * args.input_gain).astype(np.float32)
    n_blocks = len(audio) // 256
    audio = audio[: n_blocks * 256].reshape(n_blocks, 256)

    capture_times = []   # index = callback ordinal == controller input sequence
    tick_log = []

    def provider(ordinal: int) -> np.ndarray:
        return audio[ordinal % n_blocks]

    def on_tick(rec: dict) -> None:
        capture_times.append(rec["start"])
        tick_log.append((rec["ordinal"], rec["deadline"], rec["woke"], rec["start"], rec["end"], rec["status"]))

    fake_sd.CONFIG.update(ppm=args.ppm, input_provider=provider, on_tick=on_tick,
                          buffer_periods=args.buffer_periods)
    tse_live.TxController = InstrumentedController
    tse_live.UdpPeer = InstrumentedPeer

    stop = threading.Event()
    per_second = []

    def sampler() -> None:
        prev_ticks = thread_cpu_ticks()
        prev_t = time.monotonic()
        prev_counts = {}
        prev_tick_idx = 0
        while not stop.wait(1.0):
            now = time.monotonic()
            ticks = thread_cpu_ticks()
            names = {t.native_id: t.name for t in threading.enumerate() if t.native_id}
            if fake_sd.STATE["thread_native_id"]:
                names[fake_sd.STATE["thread_native_id"]] = "fake-audio"
            names[os.getpid()] = "MainThread"
            dt = now - prev_t
            cpu = {}
            for tid, value in ticks.items():
                delta = value - prev_ticks.get(tid, value)
                cpu[names.get(tid, f"tid{tid}")] = 100.0 * delta / CLK_TCK / dt
            prev_ticks, prev_t = ticks, now
            ctrl = REGISTRY.get("controller")
            peer = REGISTRY.get("peer")
            row = {"t": now, "cpu": cpu}
            if ctrl is not None:
                w = ctrl.worker
                with w._condition:
                    wc = dict(w._counts)
                    lat = list(w._latencies)[-62:]
                cc = dict(ctrl._counts)
                row["infer_ms_mean_1s"] = float(np.mean(lat) * 1000) if lat else None
                row["infer_ms_max_1s"] = float(np.max(lat) * 1000) if lat else None
                for key in ("deadline_misses", "processed", "queue_replacements", "model_errors",
                            "expired_before_inference", "resets"):
                    row["w_" + key] = wc.get(key, 0) - prev_counts.get("w_" + key, 0)
                    prev_counts["w_" + key] = wc.get(key, 0)
                for key in ("model_results_used", "dry_fallback_frames", "muted_source_frames", "frames"):
                    row["c_" + key] = cc.get(key, 0) - prev_counts.get("c_" + key, 0)
                    prev_counts["c_" + key] = cc.get(key, 0)
                row["guard_anomalies_total"] = ctrl.guard.anomalies
                row["guard_invalid_total"] = ctrl.guard.invalid_frames
            if peer is not None:
                with peer._jitter_lock:
                    jc = dict(peer._jitter._counters)
                    row["jitter_level"] = len(peer._jitter._frames)
                for key in ("lost", "played", "overflow_dropped", "sequence_skipped", "rebuffered", "late",
                            "startup_silence"):
                    row["j_" + key] = jc.get(key, 0) - prev_counts.get("j_" + key, 0)
                    prev_counts["j_" + key] = jc.get(key, 0)
                try:
                    with open("/proc/net/udp") as f:
                        port = peer.bound_address[1]
                        for line in f.readlines()[1:]:
                            fields = line.split()
                            if int(fields[1].split(":")[1], 16) == port:
                                tx_q, rx_q = fields[4].split(":")
                                row["sock_rx_queue_bytes"] = int(rx_q, 16)
                                row["sock_drops"] = int(fields[-1])
                except OSError:
                    pass
            ticks_now = tick_log[prev_tick_idx:]
            prev_tick_idx += len(ticks_now)
            if ticks_now:
                arr = np.asarray([(r[2] - r[1], r[4] - r[3]) for r in ticks_now])
                row["cb_wake_late_ms_max"] = float(arr[:, 0].max() * 1000)
                row["cb_dur_ms_mean"] = float(arr[:, 1].mean() * 1000)
                row["cb_dur_ms_max"] = float(arr[:, 1].max() * 1000)
                row["cb_status_events"] = int(sum(r[5] for r in ticks_now))
            row["xruns_total"] = fake_sd.STATE["xruns"]
            per_second.append(row)

    sampler_thread = threading.Thread(target=sampler, name="sampler", daemon=True)
    sampler_thread.start()

    settings = Settings(jitter_conceal_frames=args.conceal_frames, protection_mode=args.protection_mode)
    run_args = argparse.Namespace(bind=args.bind, peer=args.peer, input_device=0, output_device=0,
                                  duration=args.duration)
    started = time.monotonic()
    report = tse_live.run_peer(run_args, args.model_path, settings)
    stop.set()
    sampler_thread.join(2)
    report["harness"] = {"name": args.name, "ppm": args.ppm, "duration": args.duration,
                         "fake_device": dict(fake_sd.STATE), "started": started}
    (args.outdir / f"{args.name}_report.json").write_text(json.dumps(report, indent=1, default=str))
    peer = REGISTRY["peer"]
    np.save(args.outdir / f"{args.name}_capture_times.npy", np.asarray(capture_times))
    np.save(args.outdir / f"{args.name}_play_log.npy", np.asarray(peer.play_log))
    np.save(args.outdir / f"{args.name}_ticks.npy", np.asarray(tick_log, dtype=float))
    with open(args.outdir / f"{args.name}_per_second.jsonl", "w") as f:
        for row in per_second:
            f.write(json.dumps(row) + "\n")
    print(f"{args.name}: done, callbacks={fake_sd.STATE['callbacks']} xruns={fake_sd.STATE['xruns']}")


if __name__ == "__main__":
    main()
