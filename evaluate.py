"""
evaluate.py
평가 코드: PESQ / STOI / SI-SNR

실험 설계서 v3 §7.1:
  지표: PESQ / STOI / SI-SNR
  100 조건 (환경 × 포화율 × 백본 × Ablation)
  3 seeds 평균 ± std
  핵심 비교: paired t-test (A-soft vs B)
"""

import os
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
from scipy import stats


# ─────────────────────────────────────────────
# 개별 지표 함수
# ─────────────────────────────────────────────

def compute_si_snr(enhanced: np.ndarray, clean: np.ndarray, eps: float = 1e-8) -> float:
    """SI-SNR (스칼라, 높을수록 좋음)."""
    enhanced = enhanced - enhanced.mean()
    clean    = clean    - clean.mean()
    dot = np.dot(enhanced, clean)
    s_target_power = np.dot(clean, clean) + eps
    s_target = (dot / s_target_power) * clean
    e_noise  = enhanced - s_target
    ratio    = np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + eps)
    return float(10 * np.log10(ratio + eps))


def compute_pesq(
    enhanced: np.ndarray,
    clean: np.ndarray,
    sample_rate: int = 16000,
    mode: str = 'wb',
) -> float:
    """
    PESQ (Perceptual Evaluation of Speech Quality).
    pesq 패키지 필요: pip install pesq
    mode: 'wb' (wideband, 16kHz) | 'nb' (narrowband, 8kHz)
    반환: PESQ MOS 점수 (1.0 ~ 4.5)
    """
    try:
        from pesq import pesq as pesq_fn, NoUtterancesError
        # pesq는 float32 배열 필요
        ref = np.asarray(clean, dtype=np.float32)
        deg = np.asarray(enhanced, dtype=np.float32)
        # 길이 맞추기
        min_len = min(len(ref), len(deg))
        return float(pesq_fn(sample_rate, ref[:min_len], deg[:min_len], mode))
    except ImportError:
        warnings.warn('pesq 패키지 없음. pip install pesq')
        return float('nan')
    except Exception as e:
        warnings.warn(f'PESQ 계산 실패: {e}')
        return float('nan')


def compute_stoi(
    enhanced: np.ndarray,
    clean: np.ndarray,
    sample_rate: int = 16000,
    extended: bool = False,
) -> float:
    """
    STOI (Short-Time Objective Intelligibility).
    pystoi 패키지 필요: pip install pystoi
    반환: 0~1 (높을수록 명료도 높음)
    """
    try:
        from pystoi import stoi as stoi_fn
        ref = np.asarray(clean, dtype=np.float64)
        deg = np.asarray(enhanced, dtype=np.float64)
        min_len = min(len(ref), len(deg))
        return float(stoi_fn(ref[:min_len], deg[:min_len], sample_rate, extended=extended))
    except ImportError:
        warnings.warn('pystoi 패키지 없음. pip install pystoi')
        return float('nan')
    except Exception as e:
        warnings.warn(f'STOI 계산 실패: {e}')
        return float('nan')


def compute_metrics(
    enhanced: np.ndarray,
    clean: np.ndarray,
    sample_rate: int = 16000,
    compute_pesq_flag: bool = True,
    compute_stoi_flag: bool = True,
) -> Dict[str, float]:
    """세 지표를 한번에 계산."""
    result = {'si_snr': compute_si_snr(enhanced, clean)}
    if compute_pesq_flag:
        result['pesq'] = compute_pesq(enhanced, clean, sample_rate)
    if compute_stoi_flag:
        result['stoi'] = compute_stoi(enhanced, clean, sample_rate)
    return result


# ─────────────────────────────────────────────
# 배치 평가
# ─────────────────────────────────────────────

def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    sample_rate: int = 16000,
    ablation_id: str = 'A-soft',
    mask_type: str = 'soft',
    max_batches: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    모델 평가 루프.

    dataloader: SpeechEnhancementDataset 기반 DataLoader
                batch = {'bc', 'ac', 'clean', 'mask'}
    반환: {'pesq_mean', 'pesq_std', 'stoi_mean', 'stoi_std',
           'si_snr_mean', 'si_snr_std'}
    """
    model.eval()
    all_metrics = {'pesq': [], 'stoi': [], 'si_snr': []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            bc    = batch['bc'].to(device)      # (B, T)
            ac    = batch['ac'].to(device)
            clean = batch['clean'].to(device)
            mask  = batch['mask'].to(device)    # (B, T, F)

            # 모델 추론 (모델 인터페이스 통일 가정)
            enhanced = model(bc=bc, ac=ac, mask=mask)   # (B, T)

            # 배치별 지표 계산
            B = clean.shape[0]
            for i in range(B):
                enh_np = enhanced[i].cpu().numpy().astype(np.float32)
                cln_np = clean[i].cpu().numpy().astype(np.float32)
                m = compute_metrics(enh_np, cln_np, sample_rate)
                for k, v in m.items():
                    all_metrics[k].append(v)

            if verbose and batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(dataloader)}'
                      f'  SI-SNR={np.nanmean(all_metrics["si_snr"]):.2f}')

    # 통계 집계
    result = {}
    for k, vals in all_metrics.items():
        arr = np.array([v for v in vals if not np.isnan(v)])
        if len(arr) == 0:
            result[f'{k}_mean'] = float('nan')
            result[f'{k}_std']  = float('nan')
        else:
            result[f'{k}_mean'] = float(np.mean(arr))
            result[f'{k}_std']  = float(np.std(arr))

    return result


# ─────────────────────────────────────────────
# 100 조건 평가 매트릭스
# ─────────────────────────────────────────────

def evaluate_full_matrix(
    results_dir: str,
    save_csv: str = 'results/evaluation_matrix.csv',
) -> None:
    """
    results_dir 내의 저장된 결과 .npz 파일들을 읽어
    실험 설계서 §7.1의 100 조건 매트릭스 CSV로 정리.

    결과 파일 구조 (train.py에서 저장):
      results_dir/{backbone}/{ablation}/seed{N}/{env}_{sat_level}_metrics.npz
    """
    try:
        import pandas as pd
    except ImportError:
        warnings.warn('pandas 필요: pip install pandas')
        return

    rows = []
    for npz_path in sorted(Path(results_dir).rglob('*_metrics.npz')):
        parts = npz_path.parts
        try:
            # 경로에서 메타 추출
            backbone = [p for p in parts if p in ('dpcrn', 'unet')][0]
            ablation = [p for p in parts if p.startswith('A-') or p in ('B', 'C')][0]
            seed     = int([p for p in parts if p.startswith('seed')][0].replace('seed', ''))
            stem     = npz_path.stem  # e.g., 'military_S2_metrics'
            env, sat_level = stem.split('_')[:2]
        except Exception:
            continue

        data = np.load(npz_path)
        row = {
            'backbone': backbone, 'ablation': ablation,
            'seed': seed, 'env': env, 'sat_level': sat_level,
        }
        for k in data.files:
            row[k] = float(data[k])
        rows.append(row)

    if not rows:
        print(f'결과 파일 없음: {results_dir}')
        return

    df = pd.DataFrame(rows)
    # seed 평균 집계
    group_cols = ['backbone', 'ablation', 'env', 'sat_level']
    agg = df.groupby(group_cols).agg(
        pesq_mean=('pesq_mean', 'mean'), pesq_std=('pesq_mean', 'std'),
        stoi_mean=('stoi_mean', 'mean'), stoi_std=('stoi_mean', 'std'),
        si_snr_mean=('si_snr_mean', 'mean'), si_snr_std=('si_snr_mean', 'std'),
    ).reset_index()

    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    agg.to_csv(save_csv, index=False)
    print(f'평가 매트릭스 저장: {save_csv}')
    print(agg.to_string())


# ─────────────────────────────────────────────
# 통계 검정 (paired t-test)
# ─────────────────────────────────────────────

def paired_ttest(
    scores_a: List[float],
    scores_b: List[float],
    alpha: float = 0.05,
    label_a: str = 'A-soft',
    label_b: str = 'B',
    metric: str = 'PESQ',
) -> Dict:
    """
    실험 설계서 §7.5: 핵심 비교(A-soft vs B) paired t-test.

    반환: {'t_stat', 'p_value', 'significant', 'mean_diff'}
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    t_stat, p_value = stats.ttest_rel(a, b)
    significant = bool(p_value < alpha)

    result = {
        't_stat': float(t_stat),
        'p_value': float(p_value),
        'significant': significant,
        'mean_diff': float(np.mean(a) - np.mean(b)),
        'mean_a': float(np.mean(a)),
        'mean_b': float(np.mean(b)),
    }

    print(f'\n=== Paired t-test: {label_a} vs {label_b} [{metric}] ===')
    print(f'  {label_a}: {result["mean_a"]:.4f}')
    print(f'  {label_b}: {result["mean_b"]:.4f}')
    print(f'  차이:     {result["mean_diff"]:+.4f}')
    print(f'  t={result["t_stat"]:.4f}, p={result["p_value"]:.4f}')
    print(f'  유의 (p<{alpha}): {"✅ YES" if significant else "❌ NO"}')

    return result


# ─────────────────────────────────────────────
# 포화율별 세그먼트 지표 계산 유틸
# ─────────────────────────────────────────────

SATURATION_LEVELS = {
    'S0': (0.00, 0.05),    # 0%
    'S1': (0.05, 0.35),    # ~30%
    'S2': (0.30, 0.55),    # ~50%
    'S3': (0.55, 0.75),    # ~70%
    'S4': (0.75, 1.01),    # ~100%
}


def bucket_by_saturation(
    metrics_list: List[Dict],
    clip_ratios: List[float],
) -> Dict[str, List]:
    """
    각 샘플의 clip_ratio를 기준으로 포화율 구간별로 지표 분류.

    metrics_list: [{'pesq': ..., 'stoi': ..., 'si_snr': ...}, ...]
    clip_ratios : 각 샘플의 클리핑 비율

    반환: {'S0': [{'pesq':..., ...}, ...], 'S1': [...], ...}
    """
    bucketed = {k: [] for k in SATURATION_LEVELS}
    for m, cr in zip(metrics_list, clip_ratios):
        for level, (lo, hi) in SATURATION_LEVELS.items():
            if lo <= cr < hi:
                bucketed[level].append(m)
                break
    return bucketed


# ─────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────

if __name__ == '__main__':
    sr = 16000
    t = np.linspace(0, 4, sr * 4, endpoint=False)
    clean = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    enhanced = clean + np.random.randn(len(clean)).astype(np.float32) * 0.01

    m = compute_metrics(enhanced, clean, sr)
    print(f'SI-SNR: {m["si_snr"]:.2f} dB')
    print(f'PESQ:   {m.get("pesq", "N/A")}')
    print(f'STOI:   {m.get("stoi", "N/A")}')

    # t-test 예시
    scores_a = [3.5, 3.6, 3.7, 3.8, 3.6]
    scores_b = [3.0, 3.1, 3.2, 3.0, 3.1]
    paired_ttest(scores_a, scores_b, metric='PESQ')
    print('\nevaluate.py 단위 테스트 완료!')
