"""Observation adding (Iwamoto et al., Interspeech 2022): y = (1-b)*enhanced + b*noisy.

Built from saved outputs, so no model is re-run. Systems written:
  S1d_gtcrn_dry20   from S1_gtcrn_only
  S6d_dns48_dry20   from S6_demucs_dns48
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

SP = Path(os.environ.get("TSE_BENCH_DIR", Path(__file__).resolve().parents[1] / "work"))
TS = SP / "eval/testset"
OUT = SP / "eval/out"
B = 0.2
PAIRS = {"S1d_gtcrn_dry20": "S1_gtcrn_only", "S6d_dns48_dry20": "S6_demucs_dns48",
         "S13d_ulunas_dry20": "S13_ulunas", "S9d_dtln_dry20": "S9_dtln"}
CONDS = {"c00_clean", "c01_pink5", "c02_vehicle0", "c03_babble5", "c04_gunshot", "c05_blast_sat", "c06_blast_hdr"}


def main():
    meta = json.loads((TS / "meta.json").read_text())
    n = 0
    for clip in meta["clips"]:
        if clip["cond"] not in CONDS:
            continue
        x = sf.read(TS / clip["cond"] / f"{clip['uid']}.wav", dtype="float32")[0]
        for new, src in PAIRS.items():
            if (OUT / new / clip["cond"] / f"{clip['uid']}.wav").exists():
                continue
            p = OUT / src / clip["cond"] / f"{clip['uid']}.wav"
            y = sf.read(p, dtype="float32")[0]
            # match the enhanced level before mixing so b is a true proportion
            g = np.dot(y, x) / (np.dot(x, x) + 1e-12)
            g = g if g > 0.05 else 1.0
            z = (1 - B) * y + B * g * x
            d = OUT / new / clip["cond"]
            d.mkdir(parents=True, exist_ok=True)
            sf.write(d / f"{clip['uid']}.wav", z.astype(np.float32), 16000, subtype="FLOAT")
            (d / f"{clip['uid']}.json").write_text(json.dumps({"wall_s": 0.0, "audio_s": len(x) / 16000,
                                                                "dry_mix": B, "source_system": src}))
            n += 1
    print("written", n)


if __name__ == "__main__":
    main()
