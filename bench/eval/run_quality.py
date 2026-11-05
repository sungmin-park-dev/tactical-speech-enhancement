"""Run systems over the test set, then compute metrics (and optionally WER)."""

from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SP = Path(os.environ.get("TSE_BENCH_DIR", Path(__file__).resolve().parents[1] / "work"))
TS = SP / "eval/testset"
OUTROOT = SP / "eval/out"
AEC_ONLY = {"S10_aec_gtcrn", "S11_aec_only"}
ECHO_CONDS = {"c10_echo_only", "c11_doubletalk"}
IMPULSE_CONDS = {"c04_gunshot", "c05_blast_sat", "c06_blast_hdr", "c07_speech_loud", "c08_speech_clip"}
DRY_CONDS = {"c00_clean", "c01_pink5", "c02_vehicle0", "c03_babble5", "c04_gunshot", "c05_blast_sat", "c06_blast_hdr"}
SYSTEM_CONDS = {"S10_aec_gtcrn": ECHO_CONDS, "S11_aec_only": ECHO_CONDS, "S1d_gtcrn_dry20": DRY_CONDS, "S6d_dns48_dry20": DRY_CONDS,
                "S13d_ulunas_dry20": DRY_CONDS, "S9d_dtln_dry20": DRY_CONDS,
                "S6b_demucs_dns48_offline": IMPULSE_CONDS | {"c00_clean", "c01_pink5"}}


def _init():
    import torch
    torch.set_num_threads(1)


def run_one(task):
    system, clip = task
    import systems as S
    cond, uid = clip["cond"], clip["uid"]
    out_path = OUTROOT / system / cond / f"{uid}.wav"
    if out_path.exists():
        return None
    x, fs = sf.read(TS / cond / f"{uid}.wav", dtype="float32")
    kw = {}
    far = TS / cond / f"{uid}.far.wav"
    if far.exists():
        kw["far"] = sf.read(far, dtype="float32")[0]
    t0 = time.perf_counter()
    y, diag = S.SYSTEMS[system](x, **kw)
    dt = time.perf_counter() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, np.asarray(y, dtype=np.float32), fs, subtype="FLOAT")
    diag = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in diag.items()}
    (out_path.with_suffix(".json")).write_text(json.dumps({"wall_s": dt, "audio_s": len(x) / fs, **diag}))
    return system, cond, uid, dt


def metric_one(task):
    system, clip = task
    import metrics as M
    cond, uid = clip["cond"], clip["uid"]
    y = sf.read(OUTROOT / system / cond / f"{uid}.wav", dtype="float32")[0].astype(np.float64)
    x = sf.read(TS / cond / f"{uid}.wav", dtype="float32")[0].astype(np.float64)
    ref = sf.read(TS / cond / f"{uid}.ref.wav", dtype="float32")[0].astype(np.float64)
    diag = json.loads((OUTROOT / system / cond / f"{uid}.json").read_text())
    row = {"system": system, "cond": cond, "uid": uid, **{f"diag_{k}": v for k, v in diag.items()}}
    s0, s1 = clip["speech_start"], clip["speech_end"]
    if cond == "c10_echo_only":
        echo = sf.read(TS / cond / f"{uid}.echo.wav", dtype="float32")[0].astype(np.float64)
        row["erle_db"] = M.erle_db(x, y, s0, s1)
        row["erle_db_2nd_half"] = M.erle_db(x, y, (s0 + s1) // 2, s1)
        row["residual_echo_dbfs"] = float(10 * np.log10(np.mean(y[s0:s1] ** 2) + 1e-12))
        row["mic_echo_dbfs"] = float(10 * np.log10(np.mean(echo[s0:s1] ** 2) + 1e-12))
        return row
    row["si_sdr"] = float(M.si_sdr(y, ref))
    row["pesq_wb"] = M.pesq_wb(y, ref)
    row["estoi"] = M.estoi(y, ref)
    row["speech_loss_ratio"] = M.speech_loss_ratio(y, ref)
    row["out_peak"] = float(np.abs(y).max())
    row["out_fullscale_samples"] = int((np.abs(y) >= 0.999).sum())
    # flat-top runs (>=3 consecutive samples within 0.5% of the output's own peak): clipping signature
    hot = np.abs(y) >= 0.995 * (np.abs(y).max() + 1e-12)
    runs = np.diff(np.concatenate([[0], hot.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
    row["flat_top_runs"] = int(np.sum((ends - starts) >= 3))
    if "events" in clip:
        row.update(M.window_metrics(y, ref, clip["events"]))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--conds", nargs="*", default=None)
    ap.add_argument("--stage", choices=["run", "metrics", "both"], default="both")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="metrics")
    args = ap.parse_args()
    meta = json.loads((TS / "meta.json").read_text())
    clips = [c for c in meta["clips"] if args.conds is None or c["cond"] in args.conds]
    if args.limit:
        seen = {}
        lim = []
        for c in clips:
            seen[c["cond"]] = seen.get(c["cond"], 0) + 1
            if seen[c["cond"]] <= args.limit:
                lim.append(c)
        clips = lim
    tasks = []
    for s in args.systems:
        for c in clips:
            if s in SYSTEM_CONDS and c["cond"] not in SYSTEM_CONDS[s]:
                continue
            tasks.append((s, c))
    if args.stage in ("run", "both"):
        t0 = time.time()
        with Pool(args.workers, initializer=_init) as pool:
            for i, r in enumerate(pool.imap_unordered(run_one, tasks, chunksize=2)):
                if i % 100 == 0:
                    print("run", i, len(tasks), round(time.time() - t0, 1), flush=True)
    if args.stage in ("metrics", "both"):
        out = SP / f"eval/{args.tag}.jsonl"
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                r = json.loads(line)
                done.add((r["system"], r["cond"], r["uid"]))
        todo = [t for t in tasks if (t[0], t[1]["cond"], t[1]["uid"]) not in done]
        n = 0
        with Pool(args.workers) as pool, open(out, "a") as f:
            for r in pool.imap_unordered(metric_one, todo, chunksize=4):
                f.write(json.dumps(r) + "\n")
                f.flush()
                n += 1
        print("metrics rows", n, "->", out)


if __name__ == "__main__":
    main()
