"""Per-frame inference timing on the target device (Raspberry Pi 5).

Measures, for each model and thread count, the distribution of per-hop compute
time and the real-time factor (RTF = compute time / audio time), while logging
SoC temperature, ARM clock and throttling flags (vcgencmd, if present).

Usage on the Pi (inside the project venv; torch only needed for Demucs):
  python model_rtf.py --gtcrn ~/.cache/tactical-speech-enhancement/gtcrn_simple.onnx \
      --dtln-dir ./dtln --demucs-dir ./fb_denoiser --demucs-weights ./weights \
      --seconds 60 --threads 1 2 4 --out rtf_pi5.json

Report the JSON together with: Pi model/revision, OS image, kernel, Python,
onnxruntime/torch versions, cooling (fan/heatsink/case), power supply.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def soc_state() -> dict:
    out = {}
    if shutil.which("vcgencmd"):
        for key, cmd in (("temp", ["measure_temp"]), ("arm_hz", ["measure_clock", "arm"]),
                         ("throttled", ["get_throttled"])):
            try:
                out[key] = subprocess.check_output(["vcgencmd", *cmd], text=True).strip()
            except Exception as exc:  # noqa: BLE001
                out[key] = f"error: {exc}"
    try:
        out["scaling_governor"] = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        pass
    return out


def summarise(times_ms: np.ndarray, hop_ms: float) -> dict:
    return {
        "frames": int(len(times_ms)), "hop_ms": hop_ms,
        "mean_ms": float(times_ms.mean()), "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)), "p99_ms": float(np.percentile(times_ms, 99)),
        "max_ms": float(times_ms.max()), "rtf_mean": float(times_ms.mean() / hop_ms),
        "frames_over_hop": int((times_ms >= hop_ms).sum()),
        "first_vs_last_minute_mean_ms": [float(times_ms[: int(60000 / hop_ms)].mean()),
                                         float(times_ms[-int(60000 / hop_ms):].mean())],
    }


def bench_gtcrn(model: Path, threads: int, seconds: float) -> dict:
    from tactical_speech_enhancement.gtcrn import OFFICIAL_MODEL_SHA256, GtcrnStreamingEnhancer
    enh = GtcrnStreamingEnhancer(model, expected_sha256=OFFICIAL_MODEL_SHA256, num_threads=threads)
    rng = np.random.default_rng(0)
    n = int(seconds * 16000 / 256)
    blocks = (0.05 * rng.standard_normal((n, 256))).astype(np.float32)
    for b in blocks[:100]:
        enh.process_hop(b)
    t = np.empty(n)
    for i, b in enumerate(blocks):
        s = time.perf_counter()
        enh.process_hop(b)
        t[i] = (time.perf_counter() - s) * 1000
    return summarise(t, 16.0)


def bench_dtln(d: Path, threads: int, seconds: float) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    m1 = ort.InferenceSession(str(d / "model_1.onnx"), so, providers=["CPUExecutionProvider"])
    m2 = ort.InferenceSession(str(d / "model_2.onnx"), so, providers=["CPUExecutionProvider"])
    i1 = {i.name: np.zeros([x if isinstance(x, int) else 1 for x in i.shape], np.float32) for i in m1.get_inputs()}
    i2 = {i.name: np.zeros([x if isinstance(x, int) else 1 for x in i.shape], np.float32) for i in m2.get_inputs()}
    n1 = [i.name for i in m1.get_inputs()]
    n2 = [i.name for i in m2.get_inputs()]
    rng = np.random.default_rng(0)
    n = int(seconds * 16000 / 128)
    buf = np.zeros(512, np.float32)
    t = np.empty(n)
    for k in range(n):
        s = time.perf_counter()
        buf[:-128] = buf[128:]
        buf[-128:] = 0.05 * rng.standard_normal(128)
        F = np.fft.rfft(buf)
        mag = np.abs(F).reshape(1, 1, -1).astype(np.float32)
        i1[n1[0]] = mag
        o1 = m1.run(None, i1)
        i1[n1[1]] = o1[1]
        est = np.fft.irfft(mag * o1[0] * np.exp(1j * np.angle(F))).reshape(1, 1, -1).astype(np.float32)
        i2[n2[0]] = est
        o2 = m2.run(None, i2)
        i2[n2[1]] = o2[1]
        t[k] = (time.perf_counter() - s) * 1000
    return summarise(t, 8.0)


def bench_demucs(src: Path, weights: Path, name: str, threads: int, seconds: float, num_frames: int) -> dict:
    import torch
    sys.path.insert(0, str(src))
    from denoiser.demucs import Demucs, DemucsStreamer
    torch.set_num_threads(threads)
    files = {"dns48": ("dns48-11decc9d8e3f0998.th", 48), "dns64": ("dns64-a7761ff99a7d5bb6.th", 64)}
    fname, hidden = files[name]
    model = Demucs(hidden=hidden, sample_rate=16000)
    model.load_state_dict(torch.load(weights / fname, map_location="cpu", weights_only=True))
    model.eval()
    st = DemucsStreamer(model, num_frames=num_frames)
    hop = st.stride
    rng = np.random.default_rng(0)
    n = int(seconds * 16000 / hop)
    t = np.empty(n)
    with torch.no_grad():
        st.feed(torch.zeros(1, st.total_length))
        for k in range(n):
            x = torch.from_numpy((0.05 * rng.standard_normal(hop)).astype(np.float32))[None]
            s = time.perf_counter()
            st.feed(x)
            t[k] = (time.perf_counter() - s) * 1000
    res = summarise(t, hop / 16.0)
    res["algorithmic_latency_ms"] = st.total_length / 16.0
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtcrn", type=Path)
    ap.add_argument("--dtln-dir", type=Path)
    ap.add_argument("--demucs-dir", type=Path)
    ap.add_argument("--demucs-weights", type=Path)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--out", type=Path, default=Path("rtf_report.json"))
    args = ap.parse_args()
    report = {"platform": platform.platform(), "machine": platform.machine(),
              "python": platform.python_version(), "runs": []}
    try:
        report["cpuinfo_model"] = [ln for ln in Path("/proc/cpuinfo").read_text().splitlines() if "Model" in ln][:1]
    except OSError:
        pass
    jobs = []
    for th in args.threads:
        if args.gtcrn:
            jobs.append(("gtcrn", th, lambda th=th: bench_gtcrn(args.gtcrn, th, args.seconds)))
        if args.dtln_dir:
            jobs.append(("dtln", th, lambda th=th: bench_dtln(args.dtln_dir, th, args.seconds)))
        if args.demucs_dir and args.demucs_weights:
            for name in ("dns48", "dns64"):
                for nf in (1, 2):
                    jobs.append((f"demucs_{name}_f{nf}", th,
                                 lambda th=th, name=name, nf=nf: bench_demucs(args.demucs_dir, args.demucs_weights,
                                                                             name, th, args.seconds, nf)))
    for name, th, fn in jobs:
        before = soc_state()
        res = fn()
        after = soc_state()
        row = {"model": name, "threads": th, **res, "soc_before": before, "soc_after": after}
        report["runs"].append(row)
        print(f"{name:18s} threads={th} mean={res['mean_ms']:.2f}ms p99={res['p99_ms']:.2f}ms "
              f"RTF={res['rtf_mean']:.3f} over_hop={res['frames_over_hop']} {after.get('temp', '')}")
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
