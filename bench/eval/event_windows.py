"""Fine-grained windows around impulsive events (c04/c05/c06), per system.

Scale is estimated on a speech region before the first event, so blast residue
does not bias it. Reported per window (mean over events, with bootstrap CI):
  err_db  = 10log10(E[out - clean] / E[clean])   (distortion relative to the speech present)
  gain_db = 10log10(E[out] / E[clean]) over speech-active frames only
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SP = Path(os.environ.get("TSE_BENCH_DIR", Path(__file__).resolve().parents[1] / "work"))
TS = SP / "eval/testset"
OUT = SP / "eval/out"
FS = 16000
rng = np.random.default_rng(0)

WINDOWS = {"pre_64_32ms": (-0.064, -0.032, "start"), "pre_32_0ms": (-0.032, 0.0, "start"),
           "post_0_250ms": (0.0, 0.25, "end"), "post_250_2000ms": (0.25, 2.0, "end")}


def ci(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if len(v) < 2:
        return [float("nan")] * 3
    m = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main(systems, conds=("c04_gunshot", "c05_blast_sat", "c06_blast_hdr")):
    meta = json.loads((TS / "meta.json").read_text())
    res = {}
    for cond in conds:
        clips = [c for c in meta["clips"] if c["cond"] == cond]
        for system in systems:
            acc = {k: {"err": [], "gain": []} for k in WINDOWS}
            for c in clips:
                p = OUT / system / cond / f"{c['uid']}.wav"
                if not p.exists():
                    continue
                y = sf.read(p, dtype="float64")[0]
                ref = sf.read(TS / cond / f"{c['uid']}.ref.wav", dtype="float64")[0]
                ev = c.get("events", [])
                if not ev:
                    continue
                a, b = c["speech_start"], max(c["speech_start"] + FS // 4, ev[0]["start"] - FS // 10)
                scale = np.dot(y[a:b], ref[a:b]) / (np.dot(ref[a:b], ref[a:b]) + 1e-12)
                if not np.isfinite(scale) or abs(scale) < 1e-3:
                    scale = 1.0
                for e in ev:
                    for k, (t0, t1, anchor) in WINDOWS.items():
                        base = e["start"] if anchor == "start" else e["end"]
                        s0, s1 = int(base + t0 * FS), int(base + t1 * FS)
                        s0, s1 = max(s0, 0), min(s1, len(ref))
                        if s1 - s0 < 128:
                            continue
                        r, o = ref[s0:s1], y[s0:s1] / scale
                        er = np.sum(r ** 2)
                        if er < 1e-8:
                            continue
                        acc[k]["err"].append(10 * np.log10(np.sum((o - r) ** 2) / er + 1e-12))
                        acc[k]["gain"].append(10 * np.log10(np.sum(o ** 2) / er + 1e-12))
            res[f"{cond}|{system}"] = {k: {"err_db": ci(v["err"]), "gain_db": ci(v["gain"]), "n": len(v["err"])}
                                       for k, v in acc.items()}
    (SP / "eval/event_windows.json").write_text(json.dumps(res, indent=1))
    for key, v in res.items():
        print(key, " | ".join(f"{k}: err {w['err_db'][0]:+.1f} gain {w['gain_db'][0]:+.1f} (n={w['n']})"
                               for k, w in v.items()))


if __name__ == "__main__":
    main(sys.argv[1:])
