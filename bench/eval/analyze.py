"""Aggregate metrics with paired bootstrap 95% CIs over utterances."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

SP = Path(os.environ.get("TSE_BENCH_DIR", Path(__file__).resolve().parents[1] / "work"))
rng = np.random.default_rng(0)
B = 2000


def load(path):
    p = SP / "eval" / path
    if not p.exists():
        return []
    seen, rows = set(), []
    for line in p.read_text().splitlines():
        r = json.loads(line)
        key = (r["system"], r["cond"], r["uid"])
        if key in seen:      # resumable/parallel runs may write a row twice
            continue
        seen.add(key)
        rows.append(r)
    return rows


def boot_mean(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float("nan"),) * 3
    idx = rng.integers(0, len(v), (B, len(v)))
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_wer(err, words):
    err, words = np.asarray(err, float), np.asarray(words, float)
    idx = rng.integers(0, len(err), (B, len(err)))
    w = err[idx].sum(1) / words[idx].sum(1)
    return float(err.sum() / words.sum()), float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))


def paired_diff(a: dict, b: dict):
    keys = [k for k in a if k in b and np.isfinite(a[k]) and np.isfinite(b[k])]
    d = np.array([a[k] - b[k] for k in keys])
    if len(d) < 2:
        return (float("nan"),) * 3 + (0,)
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(d)


def main():
    rows = load("metrics.jsonl")
    wer = load("wer.jsonl")
    by = defaultdict(list)
    for r in rows:
        by[(r["cond"], r["system"])].append(r)
    wby = defaultdict(list)
    for r in wer:
        wby[(r["cond"], r["system"])].append(r)
    conds = sorted({r["cond"] for r in rows})
    import re
    systems = sorted({r["system"] for r in rows},
                     key=lambda s: (int(re.match(r"S(\d+)", s).group(1)), s))
    summary = {}
    for c in conds:
        for s in systems:
            rs = by.get((c, s), [])
            if not rs:
                continue
            e = {"n": len(rs)}
            for m in ("si_sdr", "pesq_wb", "estoi", "speech_loss_ratio", "pre_event_distortion_db",
                      "post_event_speech_energy_db", "post_event_si_sdr", "erle_db", "erle_db_2nd_half",
                      "residual_echo_dbfs", "flat_top_runs", "out_peak"):
                vals = [r.get(m, float("nan")) for r in rs]
                if any(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
                    e[m] = boot_mean([v if v is not None else float("nan") for v in vals])
            ws = wby.get((c, s), [])
            if ws:
                e["wer"] = boot_wer([w["errors"] for w in ws], [w["ref_words"] for w in ws])
            e["diag_samples_over_full_scale_mean"] = float(np.mean([r.get("diag_samples_over_full_scale", 0) for r in rs]))
            e["diag_muted_frames_mean"] = float(np.mean([r.get("diag_muted_frames", 0) for r in rs]))
            e["rtf_offline_mean"] = float(np.mean([r["diag_wall_s"] / r["diag_audio_s"] for r in rs]))
            summary[f"{c}|{s}"] = e
    # paired differences vs current pipeline and vs noisy
    diffs = {}
    for c in conds:
        for ref in ("S2_current_pipeline", "S0_noisy"):
            refd = {r["uid"]: r for r in by.get((c, ref), [])}
            for s in systems:
                if s == ref:
                    continue
                cur = {r["uid"]: r for r in by.get((c, s), [])}
                for m in ("si_sdr", "pesq_wb", "estoi", "speech_loss_ratio"):
                    a = {k: v.get(m, float("nan")) for k, v in cur.items()}
                    b = {k: v.get(m, float("nan")) for k, v in refd.items()}
                    diffs[f"{c}|{s}-{ref}|{m}"] = paired_diff(a, b)
                wa = {w["uid"]: w["wer"] for w in wby.get((c, s), [])}
                wb = {w["uid"]: w["wer"] for w in wby.get((c, ref), [])}
                if wa and wb:
                    diffs[f"{c}|{s}-{ref}|wer_utt"] = paired_diff(wa, wb)
    (SP / "eval/summary.json").write_text(json.dumps(summary, indent=1))
    (SP / "eval/diffs.json").write_text(json.dumps(diffs, indent=1))
    # compact markdown table
    lines = []
    for c in conds:
        lines.append(f"\n### {c}\n")
        lines.append("| system | SI-SDR | PESQ-WB | ESTOI | WER % | speech loss % | flat-top runs | extra |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s in systems:
            e = summary.get(f"{c}|{s}")
            if not e:
                continue
            f = lambda k, mul=1, nd=2: ("–" if k not in e or not math.isfinite(e[k][0]) else  # noqa: E731
                                        f"{e[k][0]*mul:.{nd}f} [{e[k][1]*mul:.{nd}f}, {e[k][2]*mul:.{nd}f}]")
            extra = []
            for k, lab in (("erle_db_2nd_half", "ERLE2"), ("post_event_speech_energy_db", "postE"),
                           ("pre_event_distortion_db", "preD")):
                if k in e and math.isfinite(e[k][0]):
                    extra.append(f"{lab} {e[k][0]:.1f}")
            if e.get("diag_samples_over_full_scale_mean"):
                extra.append(f">FS {e['diag_samples_over_full_scale_mean']:.0f}")
            lines.append(f"| {s} | {f('si_sdr', 1, 1)} | {f('pesq_wb')} | {f('estoi')} | {f('wer', 100, 1)} | "
                         f"{f('speech_loss_ratio', 100, 1)} | {e.get('flat_top_runs', [float('nan')])[0]:.1f} | {'; '.join(extra)} |")
    (SP / "eval/tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
