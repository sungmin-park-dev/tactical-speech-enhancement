"""Launch proxy + two instrumented peers for one condition, then analyse latency."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def analyse(outdir: Path) -> dict:
    result = {}
    for src, dst in (("A", "B"), ("B", "A")):
        cap = np.load(outdir / f"{src}_capture_times.npy")
        play = np.load(outdir / f"{dst}_play_log.npy")
        played = play[play[:, 1] >= 0]
        seq = played[:, 1].astype(int)
        ok = seq < len(cap)
        lat = (played[ok, 0] - cap[seq[ok]]) * 1000.0
        t = played[ok, 0] - played[0, 0]
        if not len(lat):
            result[f"{src}->{dst}"] = {"played": 0}
            continue
        minutes = np.floor(t / 60).astype(int)
        per_min = [float(np.median(lat[minutes == m])) for m in range(minutes.max() + 1)]
        slope = float(np.polyfit(t, lat, 1)[0] * 60) if len(t) > 10 else None
        result[f"{src}->{dst}"] = {
            "played_frames": int(len(lat)),
            "silence_frames": int((play[:, 1] < 0).sum()),
            "latency_ms_p50": float(np.median(lat)), "latency_ms_p95": float(np.percentile(lat, 95)),
            "latency_ms_max": float(lat.max()), "latency_ms_min": float(lat.min()),
            "latency_ms_first_minute_median": per_min[0], "latency_ms_last_minute_median": per_min[-1],
            "latency_ms_per_minute_median": per_min,
            "linear_trend_ms_per_minute": slope,
            "jitter_level_mean": float(play[:, 3].mean()), "jitter_level_max": int(play[:, 3].max()),
        }
        np.save(outdir / f"lat_{src}to{dst}.npy", np.stack([t, lat]))
    for name in ("A", "B"):
        rows = [json.loads(line) for line in open(outdir / f"{name}_per_second.jsonl")]
        threads = {}
        for r in rows[2:]:
            for k, v in r["cpu"].items():
                threads.setdefault(k, []).append(v)
        result[f"{name}_thread_cpu_pct_mean"] = {k: round(float(np.mean(v)), 2) for k, v in threads.items()}
        result[f"{name}_thread_cpu_pct_max"] = {k: round(float(np.max(v)), 2) for k, v in threads.items()}
        rep = json.loads((outdir / f"{name}_report.json").read_text())
        p = rep["pipeline"]
        result[f"{name}_pipeline"] = {
            "frames": p.get("frames"), "model_results_used": p.get("model_results_used", 0),
            "dry_fallback_frames": p.get("dry_fallback_frames", 0),
            "muted_source_frames": p.get("muted_source_frames", 0),
            "guard_anomalies": p["guard"]["anomalies"], "guard_invalid": p["guard"]["invalid_frames"],
            "worker_deadline_misses": p["worker"].get("deadline_misses", 0),
            "worker_latency_mean_ms": p["worker"].get("latency_mean_ms"),
            "worker_latency_max_ms": p["worker"].get("latency_max_ms"),
            "capture_processing_mean_ms": p.get("capture_processing_mean_ms"),
            "capture_processing_max_ms": p.get("capture_processing_max_ms"),
            "callback_mean_ms": rep["audio"].get("callback_mean_ms"),
            "callback_max_ms": rep["audio"].get("callback_max_ms"),
            "callback_deadline_misses": rep["audio"].get("callback_deadline_misses"),
            "audio_status_events": rep["audio"].get("audio_status_events"),
            "fake_xruns": rep["harness"]["fake_device"]["xruns"],
            "jitter": {k: rep["network"]["jitter"].get(k) for k in (
                "accepted", "lost", "played", "late", "overflow_dropped", "sequence_skipped",
                "rebuffered", "high_watermark")},
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--ppm-a", type=float, default=0)
    ap.add_argument("--ppm-b", type=float, default=0)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--base-port", type=int, default=6000)
    ap.add_argument("--proxy-args", default="")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--peer-extra", default="")
    args = ap.parse_args()
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    if not args.analyse_only:
        p = args.base_port
        a, b, s1, s2 = (f"127.0.0.1:{p + i}" for i in range(4))
        procs = []
        proxy = subprocess.Popen([sys.executable, str(HERE / "udp_proxy.py"), "--a", a, "--b", b,
                                  "--s1", s1, "--s2", s2, "--duration", str(args.duration + 8),
                                  "--log", str(out / "proxy.json")] + args.proxy_args.split())
        time.sleep(0.5)
        for name, bind, peer, ppm in (("A", a, s1, args.ppm_a), ("B", b, s2, args.ppm_b)):
            procs.append(subprocess.Popen([
                sys.executable, str(HERE / "peer_harness.py"), "--name", name, "--bind", bind,
                "--peer", peer, "--ppm", str(ppm), "--duration", str(args.duration),
                "--input", str(args.input), "--outdir", str(out), "--model-path", str(args.model_path)]
                + args.peer_extra.split()))
        for proc in procs:
            proc.wait()
        proxy.wait()
    summary = analyse(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    brief = {k: {kk: vv for kk, vv in v.items() if kk != "latency_ms_per_minute_median"}
             if isinstance(v, dict) and "->" in k else v for k, v in summary.items()}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
