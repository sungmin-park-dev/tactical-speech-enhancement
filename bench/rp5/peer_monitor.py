"""Run the unmodified `tse peer` on real audio hardware with external telemetry.

Adds (without editing repository files):
  * per-second JSONL: per-thread CPU %, inference latency, deadline misses, dry/muted
    frames, jitter-buffer level, socket receive-queue bytes, callback duration,
    PortAudio status flags (input overflow / output underflow, separately),
    SoC temperature/clock/throttle (vcgencmd), and the PortAudio DAC-ADC time gap
  * optional audio taps (int16 WAV): raw mic input, transmitted frame, received playout

Usage on each Pi (same options as `tse peer`, plus):
  python peer_monitor.py --bind IP:PORT --peer IP:PORT --input-device N --output-device M \
      --duration 1800 --outdir run_A --dump-audio

Two-device end-to-end latency needs a common reference; see e2e_latency.py.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from tactical_speech_enhancement import engine as tse_engine
from tactical_speech_enhancement import live as tse_live
from tactical_speech_enhancement import network as tse_network
from tactical_speech_enhancement.artifacts import default_model_path
from tactical_speech_enhancement.config import Settings

CLK_TCK = os.sysconf("SC_CLK_TCK")
REG: dict = {"cb": [], "flags_in": 0, "flags_out": 0, "taps": None}


class Taps:
    def __init__(self, outdir: Path):
        self.q: queue.Queue = queue.Queue(maxsize=2000)
        self.dropped = 0
        self.files = {name: sf.SoundFile(outdir / f"tap_{name}.wav", "w", 16000, 1, "PCM_16")
                      for name in ("mic_raw", "tx_sent", "rx_play")}
        self.thread = threading.Thread(target=self._run, name="tap-writer", daemon=True)
        self.thread.start()

    def put(self, name, block):
        try:
            self.q.put_nowait((name, np.asarray(block, dtype=np.float32).copy()))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            name, block = item
            self.files[name].write(block)

    def close(self):
        self.q.put(None)
        self.thread.join(5)
        for f in self.files.values():
            f.close()


class Controller(tse_engine.TxController):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        REG["controller"] = self

    def process(self, block, *, now=None, overload=False):
        if REG["taps"] is not None:
            REG["taps"].put("mic_raw", block)
        frame = super().process(block, now=now, overload=overload)
        if REG["taps"] is not None and frame.sequence >= 0:
            REG["taps"].put("tx_sent", frame.audio)
        return frame


class Peer(tse_network.UdpPeer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        REG["peer"] = self

    def playout(self, now=None):
        out = super().playout(now)
        if REG["taps"] is not None:
            REG["taps"].put("rx_play", out)
        return out


_RealStream = sd.Stream


class Stream(_RealStream):
    def __init__(self, *a, callback=None, **k):
        def wrapped(indata, outdata, frames, time_info, status):
            t0 = time.monotonic()
            if status.input_overflow:
                REG["flags_in"] += 1
            if status.output_underflow:
                REG["flags_out"] += 1
            gap = None
            try:
                gap = time_info.outputBufferDacTime - time_info.inputBufferAdcTime
            except Exception:  # noqa: BLE001
                pass
            callback(indata, outdata, frames, time_info, status)
            REG["cb"].append((t0, time.monotonic() - t0, gap))
        super().__init__(*a, callback=wrapped, **k)


def thread_ticks():
    res = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            parts = open(f"/proc/self/task/{tid}/stat").read().rsplit(")", 1)[1].split()
            res[int(tid)] = int(parts[11]) + int(parts[12])
        except (OSError, IndexError):
            pass
    return res


def soc():
    if not shutil.which("vcgencmd"):
        return {}
    out = {}
    for k, c in (("temp", ["measure_temp"]), ("arm", ["measure_clock", "arm"]), ("thr", ["get_throttled"])):
        try:
            out[k] = subprocess.check_output(["vcgencmd", *c], text=True).strip()
        except Exception:  # noqa: BLE001
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", required=True)
    ap.add_argument("--peer", required=True)
    ap.add_argument("--input-device", type=int, required=True)
    ap.add_argument("--output-device", type=int, required=True)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--dump-audio", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    sd.Stream = Stream
    tse_live.TxController = Controller
    tse_live.UdpPeer = Peer
    if args.dump_audio:
        REG["taps"] = Taps(args.outdir)
    stop = threading.Event()
    rows = []

    def sampler():
        prev, pt, pc, idx = thread_ticks(), time.monotonic(), {}, 0
        while not stop.wait(1.0):
            now = time.monotonic()
            ticks = thread_ticks()
            names = {t.native_id: t.name for t in threading.enumerate() if t.native_id}
            cpu = {names.get(tid, f"tid{tid}"): round(100 * (v - prev.get(tid, v)) / CLK_TCK / (now - pt), 1)
                   for tid, v in ticks.items()}
            prev, pt = ticks, now
            row = {"t": now, "cpu": cpu, "soc": soc(), "xrun_in_total": REG["flags_in"],
                   "xrun_out_total": REG["flags_out"]}
            cb = REG["cb"][idx:]
            idx += len(cb)
            if cb:
                d = np.array([c[1] for c in cb]) * 1000
                row.update(cb_n=len(cb), cb_ms_mean=float(d.mean()), cb_ms_max=float(d.max()))
                gaps = [c[2] for c in cb if c[2] is not None]
                if gaps:
                    row["portaudio_dac_minus_adc_ms"] = float(np.median(gaps) * 1000)
            c = REG.get("controller")
            if c is not None:
                w = c.worker
                with w._condition:
                    wc = dict(w._counts)
                    lat = list(w._latencies)[-62:]
                row["infer_ms_mean"] = float(np.mean(lat) * 1000) if lat else None
                row["infer_ms_max"] = float(np.max(lat) * 1000) if lat else None
                cc = dict(c._counts)
                for k in ("deadline_misses", "queue_replacements", "model_errors"):
                    row["w_" + k] = wc.get(k, 0) - pc.get("w_" + k, 0)
                    pc["w_" + k] = wc.get(k, 0)
                for k in ("model_results_used", "dry_fallback_frames", "muted_source_frames"):
                    row["c_" + k] = cc.get(k, 0) - pc.get("c_" + k, 0)
                    pc["c_" + k] = cc.get(k, 0)
            p = REG.get("peer")
            if p is not None:
                with p._jitter_lock:
                    row["jitter_level"] = len(p._jitter._frames)
                    jc = dict(p._jitter._counters)
                for k in ("lost", "late", "overflow_dropped", "rebuffered"):
                    row["j_" + k] = jc.get(k, 0) - pc.get("j_" + k, 0)
                    pc["j_" + k] = jc.get(k, 0)
                try:
                    port = p.bound_address[1]
                    for line in open("/proc/net/udp").readlines()[1:]:
                        f = line.split()
                        if int(f[1].split(":")[1], 16) == port:
                            row["sock_rx_queue_bytes"] = int(f[4].split(":")[1], 16)
                            row["sock_drops"] = int(f[-1])
                except OSError:
                    pass
            rows.append(row)

    th = threading.Thread(target=sampler, name="sampler", daemon=True)
    th.start()
    run_args = argparse.Namespace(bind=args.bind, peer=args.peer, input_device=args.input_device,
                                  output_device=args.output_device, duration=args.duration)
    try:
        report = tse_live.run_peer(run_args, default_model_path(), Settings())
    finally:
        stop.set()
        th.join(2)
        if REG["taps"] is not None:
            REG["taps"].close()
        with open(args.outdir / "per_second.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    report["monitor"] = {"xrun_in": REG["flags_in"], "xrun_out": REG["flags_out"],
                         "tap_dropped_blocks": REG["taps"].dropped if REG["taps"] else None}
    (args.outdir / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["monitor"]))


if __name__ == "__main__":
    main()
