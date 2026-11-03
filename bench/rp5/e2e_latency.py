"""Mouth-to-ear latency over a long call, measured with ONE common recorder.

Two Pis have unsynchronised clocks, so their timestamps cannot be subtracted.
Instead, record two channels on a third device (laptop + 2-channel USB audio
interface, or a 2-channel field recorder):

  ch0 (reference): a small loudspeaker plays `make` output next to Pi A's mic;
                   a reference mic (or the loudspeaker's electrical feed) goes to ch0
  ch1 (received):  Pi B's headset output (electrical tap at the headphone jack, or a
                   contact/small mic on the bone-conduction transducer)

The test signal is speech (denoisers pass speech; tones/chirps may be suppressed),
so delay is tracked with windowed GCC-PHAT. Run the call for 30-60 minutes.

  python e2e_latency.py make --speech-dir LibriSpeech/test-clean --minutes 60 --out ref_signal.wav
  python e2e_latency.py analyse --recording two_channel.wav --out latency.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def make(speech_dir: Path, minutes: float, out: Path, fs: int = 48000) -> None:
    files = sorted(speech_dir.rglob("*.flac"))[:400]
    chunks, total = [], 0
    rng = np.random.default_rng(0)
    target = int(minutes * 60 * fs)
    while total < target:
        for f in files:
            x, r = sf.read(f, dtype="float32")
            x = resample_poly(x, fs // 1000, r // 1000).astype(np.float32)
            x *= 0.3 / (np.abs(x).max() + 1e-9)
            gap = np.zeros(int(rng.uniform(0.2, 0.8) * fs), np.float32)
            chunks += [x, gap]
            total += len(x) + len(gap)
            if total >= target:
                break
    sf.write(out, np.concatenate(chunks)[:target], fs, subtype="PCM_16")
    print("wrote", out, minutes, "min")


def gcc_phat(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    k = int(np.argmax(cc))
    return k - max_lag, float(cc[k] / (np.mean(np.abs(cc)) + 1e-12))


def analyse(recording: Path, out: Path, win_s: float = 4.0, hop_s: float = 2.0, max_lag_s: float = 2.0) -> None:
    x, fs = sf.read(recording, dtype="float32")
    assert x.ndim == 2 and x.shape[1] >= 2, "need a 2-channel recording"
    ref, rec = x[:, 0], x[:, 1]
    W, H, L = int(win_s * fs), int(hop_s * fs), int(max_lag_s * fs)
    t, lag_ms, conf = [], [], []
    for s in range(0, len(ref) - W - L, H):
        a = ref[s:s + W]
        b = rec[s:s + W + L]
        if np.sqrt(np.mean(a ** 2)) < 1e-3:
            continue
        k, c = gcc_phat(np.pad(a, (0, L)), b, L)
        t.append(s / fs)
        lag_ms.append(1000 * k / fs)
        conf.append(c)
    t, lag_ms, conf = map(np.asarray, (t, lag_ms, conf))
    ok = (conf > np.percentile(conf, 20)) & (lag_ms > 0)
    t, lag_ms = t[ok], lag_ms[ok]
    slope = float(np.polyfit(t / 60, lag_ms, 1)[0]) if len(t) > 10 else None
    steps = np.flatnonzero(np.abs(np.diff(lag_ms)) > 10)
    res = {"windows": int(len(t)), "latency_ms_median": float(np.median(lag_ms)),
           "latency_ms_p95": float(np.percentile(lag_ms, 95)), "latency_ms_max": float(lag_ms.max()),
           "trend_ms_per_minute": slope,
           "per_10min_median_ms": [float(np.median(lag_ms[(t >= m * 600) & (t < (m + 1) * 600)]))
                                   for m in range(int(t.max() // 600) + 1) if np.any((t >= m * 600) & (t < (m + 1) * 600))],
           "step_changes_over_10ms": [{"t_s": float(t[i + 1]), "from_ms": float(lag_ms[i]), "to_ms": float(lag_ms[i + 1])}
                                      for i in steps[:200]],
           "interpretation": {
               "linear fast (>100 ms/min)": "processing slower than real time with an unbounded queue",
               "linear slow (1-20 ms/min)": "clock drift between capture and playback clocks",
               "step up after network stalls, no recovery": "jitter buffer without convergence (ratchet)",
               "bounded sawtooth": "drift handled by drop/rebuffer (repository behaviour)"}}
    out.write_text(json.dumps(res, indent=1))
    np.save(out.with_suffix(".npy"), np.stack([t, lag_ms]))
    print(json.dumps({k: v for k, v in res.items() if k != "step_changes_over_10ms"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--speech-dir", type=Path, required=True)
    m.add_argument("--minutes", type=float, default=60)
    m.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--recording", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "make":
        make(args.speech_dir, args.minutes, args.out)
    else:
        analyse(args.recording, args.out)
