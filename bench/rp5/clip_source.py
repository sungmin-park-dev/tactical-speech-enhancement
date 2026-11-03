"""Locate where 'clipping-like' sound around a blast starts: source speaker, mic/ADC, model, output, transducer.

Inputs are WAV files of the same event captured at different taps (any subset):
  --t1 reference capture of the test loudspeaker (high-AOP measurement mic or its electrical input)
  --t2 Pi microphone input before processing   (peer_monitor.py --dump-audio -> tap_mic_raw.wav)
  --t3 model / transmitted output                (tap_tx_sent.wav)
  --t4 receiver playout before the transducer   (tap_rx_play.wav on the peer)
  --t5 transducer acoustic/electrical output     (external recorder)

Per tap and per detected event it reports: flat-top runs at the tap's own peak
(hard clipping signature), crest factor, full-scale sample count, the
high-frequency energy rise in a 100 ms window before and after the event
(audible "crackle"), and broadband spectral flatness. The first tap that shows
a signature absent in the previous tap is the likely origin.

Protocol: repeat the same blast file at 5 playback levels (e.g. -24, -18, -12,
-6, 0 dB) with a fixed mic gain, then at a second mic gain. Clipping that
appears at T1 already -> loudspeaker/amplifier; tracks mic gain at T2 -> mic,
preamp or ADC (or hardware AGC if gain pumps); appears only from T3 -> model or
clamp; appears only at T5 -> playback transducer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


def flat_runs(x: np.ndarray, rel: float = 0.995, min_len: int = 3) -> int:
    peak = np.abs(x).max() + 1e-12
    hot = np.abs(x) >= rel * peak
    runs, n = 0, 0
    for h in hot:
        n = n + 1 if h else 0
        if n == min_len:
            runs += 1
    return runs


def events(x: np.ndarray, fs: int, thr_db: float = 20) -> list[tuple[int, int]]:
    frame = fs // 100
    n = len(x) // frame
    e = 10 * np.log10((x[: n * frame].reshape(n, frame) ** 2).mean(1) + 1e-12)
    base = np.median(e)
    hot = e > base + thr_db
    out, start = [], None
    for i, h in enumerate(hot):
        if h and start is None:
            start = i
        if not h and start is not None:
            out.append((start * frame, i * frame))
            start = None
    return out


def analyse_tap(path: Path) -> dict:
    x, fs = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x[:, 0]
    sos = butter(4, 4000 / (fs / 2), "high", output="sos")
    hf = sosfilt(sos, x)
    res = {"file": str(path), "fs": fs, "peak": float(np.abs(x).max()),
           "fullscale_samples": int((np.abs(x) >= 0.999).sum()), "flat_top_runs": flat_runs(x), "events": []}
    for s, e in events(x, fs):
        w = int(0.1 * fs)
        pre, post = hf[max(0, s - w):s], hf[e:e + w]
        ref = hf[max(0, s - 6 * w):max(0, s - 2 * w)]
        lvl = lambda v: float(10 * np.log10(np.mean(v ** 2) + 1e-14)) if len(v) else None  # noqa: E731
        seg = x[s:e]
        res["events"].append({
            "start_s": s / fs, "end_s": e / fs, "peak": float(np.abs(seg).max()),
            "crest_db": float(20 * np.log10(np.abs(seg).max() / (np.sqrt(np.mean(seg ** 2)) + 1e-12))),
            "flat_top_runs": flat_runs(seg),
            "hf_pre_minus_baseline_db": None if lvl(pre) is None or lvl(ref) is None else lvl(pre) - lvl(ref),
            "hf_post_minus_baseline_db": None if lvl(post) is None or lvl(ref) is None else lvl(post) - lvl(ref)})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("t1", "t2", "t3", "t4", "t5"):
        ap.add_argument(f"--{k}", type=Path)
    ap.add_argument("--out", type=Path, default=Path("clip_source.json"))
    args = ap.parse_args()
    report = {k: analyse_tap(getattr(args, k)) for k in ("t1", "t2", "t3", "t4", "t5") if getattr(args, k)}
    args.out.write_text(json.dumps(report, indent=1))
    for k, r in report.items():
        print(k, "peak", round(r["peak"], 3), "fullscale", r["fullscale_samples"], "flat_runs", r["flat_top_runs"],
              "events", len(r["events"]))


if __name__ == "__main__":
    main()
