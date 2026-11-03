"""Compact Q2 tables for the report from summary.json / diffs.json / event_windows.json."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

SP = Path(os.environ.get("TSE_BENCH_DIR", Path(__file__).resolve().parents[1] / "work"))
S = json.loads((SP / "eval/summary.json").read_text())
D = json.loads((SP / "eval/diffs.json").read_text())

SYSTEMS = [("S0_noisy", "입력(무처리)"), ("S1_gtcrn_only", "GTCRN 단독"), ("S2_current_pipeline", "현재(GTCRN+가드)"),
           ("S3_guard_only", "가드 단독"), ("S4_mmse_lsa", "고전 MMSE-LSA"), ("S5_softlimit_gtcrn", "감쇠형 보호+GTCRN"),
           ("S6_demucs_dns48", "Demucs dns48 스트리밍"), ("S6b_demucs_dns48_offline", "dns48 오프라인"),
           ("S7_demucs_dns64_offline", "dns64 오프라인"), ("S8_demucs_master64_offline", "master64 오프라인"),
           ("S9_dtln", "DTLN"), ("S10_aec_gtcrn", "AEC+GTCRN"), ("S11_aec_only", "AEC 단독")]
CONDS = [("c00_clean", "깨끗한 음성"), ("c01_pink5", "분홍 5 dB"), ("c02_vehicle0", "차량 0 dB"),
         ("c03_babble5", "다화자 5 dB"), ("c04_gunshot", "총성"), ("c05_blast_sat", "폭음(포화)"),
         ("c06_blast_hdr", "폭음(무포화)"), ("c07_speech_loud", "큰 목소리"), ("c08_speech_clip", "외침(클리핑)"),
         ("c09_reverb", "잔향 0.6 s"), ("c11_doubletalk", "이중발화+반향")]


def v(cond, sys, m, mul=1.0, nd=2):
    e = S.get(f"{cond}|{sys}")
    if not e or m not in e or not math.isfinite(e[m][0]):
        return "–"
    return f"{e[m][0] * mul:.{nd}f}"


def dv(cond, sys, m, ref="S0_noisy", nd=3):
    k = f"{cond}|{sys}-{ref}|{m}"
    if k not in D or not math.isfinite(D[k][0]):
        return "–"
    mean, lo, hi, _ = D[k]
    star = "" if (lo <= 0 <= hi) else "*"
    return f"{mean:+.{nd}f}{star}"


def main():
    out = []
    out.append("**표 A. ESTOI (평균) — 높을수록 명료. 괄호 안은 입력 대비 쌍체 차이, *는 95% CI가 0을 포함하지 않음**\n")
    hdr = "| 조건 | " + " | ".join(n for _, n in SYSTEMS if _ not in ("S10_aec_gtcrn", "S11_aec_only")) + " |"
    out.append(hdr)
    out.append("|" + "---|" * (hdr.count("|") - 1))
    for c, cn in CONDS:
        cells = []
        for s, _ in SYSTEMS:
            if s in ("S10_aec_gtcrn", "S11_aec_only"):
                continue
            val = v(c, s, "estoi")
            if val != "–" and s != "S0_noisy":
                d = dv(c, s, "estoi")
                val = f"{val} ({d})"
            cells.append(val)
        out.append(f"| {cn} | " + " | ".join(cells) + " |")
    out.append("\n**표 B. WER % (발화 묶음 전체 오류/단어, faster-whisper base.en)**\n")
    out.append(hdr)
    out.append("|" + "---|" * (hdr.count("|") - 1))
    for c, cn in CONDS:
        cells = [v(c, s, "wer", 100, 1) for s, _ in SYSTEMS if s not in ("S10_aec_gtcrn", "S11_aec_only")]
        out.append(f"| {cn} | " + " | ".join(cells) + " |")
    out.append("\n**표 C. PESQ-WB (평균)**\n")
    out.append(hdr)
    out.append("|" + "---|" * (hdr.count("|") - 1))
    for c, cn in CONDS:
        cells = [v(c, s, "pesq_wb") for s, _ in SYSTEMS if s not in ("S10_aec_gtcrn", "S11_aec_only")]
        out.append(f"| {cn} | " + " | ".join(cells) + " |")
    out.append("\n**표 D. 음성 손실률 % (충격·포화 조건)**\n")
    out.append(hdr)
    out.append("|" + "---|" * (hdr.count("|") - 1))
    for c, cn in CONDS[4:9]:
        cells = [v(c, s, "speech_loss_ratio", 100, 1) for s, _ in SYSTEMS if s not in ("S10_aec_gtcrn", "S11_aec_only")]
        out.append(f"| {cn} | " + " | ".join(cells) + " |")
    out.append("\n**표 E. 반향 (c10 단일 발화 ERLE, 수렴 후반부 dB / c11 이중발화 ESTOI·WER)**\n")
    out.append("| 시스템 | ERLE 후반 dB | 이중발화 ESTOI | 이중발화 WER % |")
    out.append("|---|---|---|---|")
    for s, n in SYSTEMS:
        e = S.get(f"c10_echo_only|{s}")
        if not e:
            continue
        out.append(f"| {n} | {e['erle_db_2nd_half'][0]:.1f} | {v('c11_doubletalk', s, 'estoi')} | "
                   f"{v('c11_doubletalk', s, 'wer', 100, 1)} |")
    out.append("\n**표 F. Demucs 출력 전체 척도 초과(공식 live.py는 ±1로 하드 클램프) — 클립당 평균 샘플 수**\n")
    out.append("| 조건 | dns48 스트리밍 |")
    out.append("|---|---|")
    for c, cn in CONDS:
        e = S.get(f"{c}|S6_demucs_dns48")
        if e:
            out.append(f"| {cn} | {e.get('diag_samples_over_full_scale_mean', float('nan')):.1f} |")
    out.append("\n**표 G. 원음 20% 혼합(observation adding) 효과 — ESTOI / PESQ / WER %**\n")
    out.append("| 조건 | 입력 | GTCRN | GTCRN+원음 20% | dns48 스트리밍 | dns48+원음 20% |")
    out.append("|---|---|---|---|---|---|")
    for c, cn in CONDS[:7]:
        cells = []
        for s in ("S0_noisy", "S1_gtcrn_only", "S1d_gtcrn_dry20", "S6_demucs_dns48", "S6d_dns48_dry20"):
            cells.append(f"{v(c, s, 'estoi')} / {v(c, s, 'pesq_wb')} / {v(c, s, 'wer', 100, 1)}")
        out.append(f"| {cn} | " + " | ".join(cells) + " |")
    (SP / "eval/q2_tables.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
