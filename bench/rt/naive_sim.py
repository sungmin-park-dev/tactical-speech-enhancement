"""Discrete-event model of one direction of a voice link (not real time).

It is NOT a reconstruction of the unavailable 2025 implementation. It shows
which latency signature each failure mechanism produces, so that device logs
can be matched against them:

  policy "fifo"      unbounded queues, never drops (typical first implementation)
  policy "bounded8"  repository behaviour: drop oldest above 8 frames, rebuffer after 8 misses
  policy "adaptive"  bounded8 + slow convergence to the 4-frame target (drop one
                     frame when the level stays above target for 1 s), NetEQ-like idea

Scenarios: baseline, rtf>1 (processing slower than real time), clock drift,
Wi-Fi stalls with burst release.
"""

from __future__ import annotations

import json
from collections import deque

import numpy as np

FRAME = 0.016


def simulate(policy, minutes=30, rtf=0.3, ppm_tx=0.0, ppm_rx=0.0, stall_every=0.0, stall_ms=0.0,
             jitter_ms=2.0, seed=0, target=4, maxf=8):
    rng = np.random.default_rng(seed)
    T = minutes * 60
    n = int(T / FRAME * (1 + ppm_tx * 1e-6))
    cap = np.arange(n) * FRAME / (1 + ppm_tx * 1e-6)          # capture-complete times
    proc_time = FRAME * rtf
    # serial processing thread; fifo => unbounded backlog, otherwise drop if > 2 frames late
    done = np.empty(n)
    t_free = 0.0
    dropped_proc = np.zeros(n, bool)
    for i in range(n):
        start = max(cap[i], t_free)
        if policy != "fifo" and start - cap[i] > 2 * FRAME:
            dropped_proc[i] = True          # repository: stale job replaced / dry fallback
            done[i] = cap[i] + 2 * FRAME     # dry audio still sent on schedule
            continue
        t_free = start + proc_time
        done[i] = t_free
    send = np.maximum(done, cap + 2 * FRAME) if policy != "fifo" else done
    arr = send + 0.001 + rng.exponential(jitter_ms / 1000, n)
    if stall_every > 0:
        k = np.floor(arr / stall_every)
        s0 = k * stall_every + stall_every - stall_ms / 1000
        inside = (arr >= s0) & (arr < s0 + stall_ms / 1000)
        arr = np.where(inside, s0 + stall_ms / 1000, arr)
    arr = np.maximum.accumulate(arr)       # link-layer order preserved
    # receiver playout clock
    order = np.argsort(arr, kind="stable")
    t_play = []
    lat = []
    q = deque()
    level_log = []
    ai = 0
    tick = 0
    rx_period = FRAME / (1 + ppm_rx * 1e-6)

    first_deadline = None
    misses = 0
    above_since = None
    drops = 0
    silences = 0
    while True:
        now = tick * rx_period
        if now > T:
            break
        while ai < n and arr[order[ai]] <= now:
            q.append(order[ai])
            ai += 1
            if first_deadline is None:
                first_deadline = arr[order[ai - 1]] + target * FRAME
            if policy != "fifo":
                while len(q) > maxf:
                    q.popleft()
                    drops += 1
        if first_deadline is None or now < first_deadline:
            tick += 1
            continue
        if policy == "adaptive":
            if len(q) > target:
                above_since = now if above_since is None else above_since
                if now - above_since > 1.0:
                    q.popleft()
                    drops += 1
                    above_since = now
            else:
                above_since = None
        if q:
            i = q.popleft()
            lat.append(now - cap[i])
            t_play.append(now)
            misses = 0
        else:
            silences += 1
            misses += 1
            if policy != "fifo" and misses >= maxf:
                first_deadline = None
                misses = 0
        level_log.append(len(q))
        tick += 1
    lat = np.asarray(lat) * 1000
    t_play = np.asarray(t_play)
    def minute(m):
        sel = (t_play >= m * 60) & (t_play < (m + 1) * 60)
        return float(np.median(lat[sel])) if np.any(sel) else float("nan")

    return {"lat_ms_min1": minute(0), "lat_ms_min5": minute(4), "lat_ms_min10": minute(9),
            "lat_ms_min20": minute(19), "lat_ms_min30": minute(29), "lat_ms_max": float(lat.max()),
            "dropped_frames": int(drops + dropped_proc.sum()), "silence_frames": int(silences),
            "mean_level": float(np.mean(level_log))}


SCENARIOS = {
    "baseline (RTF 0.3)": dict(rtf=0.3),
    "CPU saturation (RTF 1.02)": dict(rtf=1.02),
    "CPU saturation (RTF 1.10)": dict(rtf=1.10),
    "clock drift tx +100 ppm": dict(ppm_tx=100),
    "clock drift tx -100 ppm": dict(ppm_tx=-100),
    "Wi-Fi stall 300 ms / 60 s": dict(stall_every=60, stall_ms=300),
    "Wi-Fi stall 1 s / 120 s": dict(stall_every=120, stall_ms=1000),
}

if __name__ == "__main__":
    rows = []
    for name, kw in SCENARIOS.items():
        for policy in ("fifo", "bounded8", "adaptive"):
            r = simulate(policy, **kw)
            rows.append({"scenario": name, "policy": policy, **r})
            print(f"{name:28s} {policy:9s} " + " ".join(
                f"{k}={v:.0f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    json.dump(rows, open("naive_sim_results.json", "w"), indent=1)
