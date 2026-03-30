"""
saturation.py
포화(포화/클리핑) 시뮬레이터 + 3유형 포화 마스크 생성기

실험 설계서 v3 §4.1 / §4.3 사양:
  포화 시뮬 :
    - 하드 클리핑 (Hard clipping)
    - 소프트 클리핑 (Soft / tanh)
    - 회복 시간 포함 (지수 감쇠, τ=2~5ms)
    - 다항식 비선형

  포화 마스크 3유형:
    1. Hard Mask  — 포화 프레임 전체 freq bin = 1, 나머지 0 (이진)
    2. Soft Mask  — 프레임 내 클리핑 비율 → sigmoid
    3. Parametric Mask — Soft Mask + 포화 종료 후 지수 감쇠 tail
"""

import numpy as np
from scipy.special import expit as sigmoid  # numerically stable
from typing import Literal, Optional, Tuple


# ─────────────────────────────────────────────
# 포화 시뮬레이터
# ─────────────────────────────────────────────

def _hard_clip(signal: np.ndarray, clip_level: float) -> np.ndarray:
    """단순 하드 클리핑."""
    return np.clip(signal, -clip_level, clip_level)


def _soft_clip_tanh(signal: np.ndarray, clip_level: float) -> np.ndarray:
    """tanh 기반 소프트 클리핑."""
    alpha = 1.0 / clip_level  # stretch factor
    return np.tanh(alpha * signal) * clip_level


def _poly_nonlinear(signal: np.ndarray, clip_level: float, order: int) -> np.ndarray:
    """
    다항식 비선형: odd-order polynomial saturation.
    signal을 [-clip_level, clip_level]로 정규화 후 x - (1/order)*x^order 형태.
    """
    x = np.clip(signal / (clip_level + 1e-8), -1.0, 1.0)
    out = x
    for k in range(3, order + 1, 2):
        out = out - (1.0 / k) * x ** k
    # 원래 스케일 복원 후 클리핑
    out = out * clip_level
    return np.clip(out, -clip_level, clip_level)


def apply_recovery_time(
    clipped: np.ndarray,
    original: np.ndarray,
    sample_rate: int = 16000,
    tau_ms: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    클리핑 종료 후 지수 감쇠 회복 시뮬:
    클리핑 끝점에서 original → clipped 사이를 지수 envelope로 보간.
    """
    if rng is None:
        rng = np.random.default_rng()

    output = clipped.copy()
    tau = tau_ms / 1000.0  # sec
    tau_samples = int(tau * sample_rate * 3)  # 3τ로 회복 완료 가정

    abs_out = np.abs(output)
    max_val = np.max(abs_out)
    if max_val < 1e-8:
        return output

    # 클리핑 구간 검출 (95% 기준)
    threshold = max_val * 0.95
    is_clipped = abs_out >= threshold

    # 클리핑 종료 지점 찾기
    end_points = np.where(np.diff(is_clipped.astype(int)) == -1)[0]

    for ep in end_points:
        end = min(ep + tau_samples, len(output))
        n_tail = end - ep
        if n_tail <= 0:
            continue
        t = np.arange(n_tail) / sample_rate
        # 회복 계수: 클리핑 영향이 지수적으로 감쇠
        recovery = np.exp(-t / tau)
        # clipped 레벨에서 original 방향으로 보간 (blend)
        orig_tail = original[ep:end]
        clip_tail = output[ep:end]
        output[ep:end] = clip_tail * recovery + orig_tail * (1.0 - recovery)

    return output


def saturate(
    signal: np.ndarray,
    clip_level: float = 0.7,
    mode: Literal['hard', 'soft', 'poly'] = 'soft',
    poly_order: int = 5,
    apply_recovery: bool = True,
    sample_rate: int = 16000,
    tau_recovery_ms: float = 3.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    음성 신호에 포화를 적용.

    Parameters
    ----------
    signal        : (N,) 원본 신호
    clip_level    : 클리핑 임계값 (신호 절대값 기준)
    mode          : 'hard' | 'soft' | 'poly'
    poly_order    : 다항식 차수 (mode='poly' 시)
    apply_recovery: 클리핑 종료 후 회복 시간 적용 여부
    tau_recovery_ms: 회복 시간 상수 (ms)

    Returns
    -------
    saturated : (N,) 포화 적용 신호
    original  : (N,) 원본 (마스크 계산용 참조)
    """
    signal = np.asarray(signal, dtype=np.float64)

    if mode == 'hard':
        out = _hard_clip(signal, clip_level)
    elif mode == 'soft':
        out = _soft_clip_tanh(signal, clip_level)
    elif mode == 'poly':
        out = _poly_nonlinear(signal, clip_level, poly_order)
    else:
        raise ValueError(f'Unknown saturation mode: {mode}')

    if apply_recovery:
        out = apply_recovery_time(out, signal, sample_rate, tau_recovery_ms, rng)

    return out.astype(np.float32), signal.astype(np.float32)


# ─────────────────────────────────────────────
# 클리핑 검출
# ─────────────────────────────────────────────

def detect_clipping(
    signal: np.ndarray,
    threshold_ratio: float = 0.95,
    consecutive: int = 3,
) -> np.ndarray:
    """
    시간 영역 클리핑 검출.

    threshold_ratio : full-scale의 몇 배를 임계값으로 볼지
    consecutive     : 연속 N 샘플 이상이어야 포화 판정

    반환: (N,) bool array — True = 포화 구간
    """
    fullscale = np.max(np.abs(signal))
    if fullscale < 1e-8:
        return np.zeros(len(signal), dtype=bool)

    threshold = fullscale * threshold_ratio
    raw_clipped = np.abs(signal) >= threshold

    if consecutive <= 1:
        return raw_clipped

    # 연속 N개 필터 (1D morphological erosion 근사)
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(raw_clipped.astype(float), size=consecutive)
    clipped_regions = smoothed >= (consecutive - 1) / consecutive
    return clipped_regions


# ─────────────────────────────────────────────
# STFT 프레임 매핑 유틸리티
# ─────────────────────────────────────────────

def _sample_range_of_frame(
    frame_idx: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_samples: int,
) -> Tuple[int, int]:
    """프레임 인덱스 → 시간 영역 샘플 범위 (start, end)."""
    center = frame_idx * hop_length + win_length // 2
    start = max(0, center - win_length // 2)
    end = min(n_samples, center + win_length // 2)
    return start, end


def _n_frames(n_samples: int, hop_length: int, n_fft: int) -> int:
    return 1 + (n_samples - n_fft) // hop_length


# ─────────────────────────────────────────────
# 포화 마스크 3유형
# ─────────────────────────────────────────────

def hard_mask(
    clipped_regions: np.ndarray,
    n_samples: int,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """
    Hard Mask: 포화 구간이 포함된 프레임 전체 freq bin = 1, 아닌 곳 0.

    반환: (n_frames, n_freq) float32, n_freq = n_fft//2 + 1
    """
    n_freq = n_fft // 2 + 1
    n_f = _n_frames(n_samples, hop_length, n_fft)
    mask = np.zeros((n_f, n_freq), dtype=np.float32)

    for fi in range(n_f):
        s, e = _sample_range_of_frame(fi, n_fft, hop_length, win_length, n_samples)
        if np.any(clipped_regions[s:e]):
            mask[fi, :] = 1.0

    return mask


def soft_mask(
    clipped_regions: np.ndarray,
    n_samples: int,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    temperature: float = 0.1,
) -> np.ndarray:
    """
    Soft Mask: 프레임 내 클리핑 비율 → sigmoid(ratio / temperature).

    반환: (n_frames, n_freq) float32
    """
    n_freq = n_fft // 2 + 1
    n_f = _n_frames(n_samples, hop_length, n_fft)
    mask = np.zeros((n_f, n_freq), dtype=np.float32)

    for fi in range(n_f):
        s, e = _sample_range_of_frame(fi, n_fft, hop_length, win_length, n_samples)
        clip_ratio = np.mean(clipped_regions[s:e].astype(float))
        val = float(sigmoid(clip_ratio / temperature))
        mask[fi, :] = val

    return mask


def parametric_mask(
    clipped_regions: np.ndarray,
    n_samples: int,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    temperature: float = 0.1,
    tau_recovery_ms: float = 5.0,
) -> np.ndarray:
    """
    Parametric Mask: Soft Mask + 포화 종료 후 지수 감쇠 tail.
    τ_recovery_ms 기준 회복 시간 반영.

    반환: (n_frames, n_freq) float32
    """
    n_f = _n_frames(n_samples, hop_length, n_fft)
    hop_ms = hop_length / (n_samples / max(n_samples, 1)) * 1000
    # proper hop duration in ms
    hop_ms = hop_length / 16000 * 1000  # TODO: pass sample_rate
    max_tail_frames = int(5 * tau_recovery_ms / hop_ms) + 1

    base = soft_mask(clipped_regions, n_samples, n_fft, hop_length,
                     win_length, temperature)

    # 포화 종료 지점에서 감쇠 tail 추가
    for fi in range(n_f - 1):
        is_sat_now = base[fi, 0] > 0.05
        is_sat_next = (fi + 1 < n_f) and (base[fi + 1, 0] > 0.05)
        if is_sat_now and not is_sat_next:
            # 종료 지점
            for k in range(1, max_tail_frames + 1):
                if fi + k >= n_f:
                    break
                decay = np.exp(-k * hop_ms / tau_recovery_ms)
                tail_val = base[fi, :] * decay
                base[fi + k, :] = np.maximum(base[fi + k, :], tail_val)

    return base.astype(np.float32)


# ─────────────────────────────────────────────
# 통합 클래스
# ─────────────────────────────────────────────

class SaturationSimulator:
    """
    포화 적용 + 3유형 마스크 생성 통합 클래스.

    사용법:
        sim = SaturationSimulator(sample_rate=16000)
        saturated, masks = sim(signal, mode='soft', mask_type='soft')
        # masks: {'hard': ..., 'soft': ..., 'parametric': ...}
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        threshold_ratio: float = 0.95,
        consecutive_samples: int = 3,
        soft_temperature: float = 0.1,
        tau_recovery_ms_range: Tuple[float, float] = (2.0, 5.0),
        rng: Optional[np.random.Generator] = None,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.threshold_ratio = threshold_ratio
        self.consecutive = consecutive_samples
        self.soft_temp = soft_temperature
        self.tau_range = tau_recovery_ms_range
        self.rng = rng or np.random.default_rng()

    def __call__(
        self,
        signal: np.ndarray,
        clip_level: Optional[float] = None,
        sat_mode: Literal['hard', 'soft', 'poly'] = 'soft',
        poly_order: int = 5,
        generate_all_masks: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """
        Parameters
        ----------
        signal      : (N,) 원본 (정규화 권장)
        clip_level  : None이면 rng로 랜덤 선택 (0.3~0.9 range)
        sat_mode    : 클리핑 방식
        generate_all_masks: True면 3유형 마스크 모두 반환

        Returns
        -------
        saturated : (N,) 포화 신호
        info      : {
            'masks': {'hard': (T,F), 'soft': (T,F), 'parametric': (T,F)},
            'clipped_regions': (N,) bool,
            'clip_level': float,
            'tau_recovery_ms': float,
            'sat_mode': str,
        }
        """
        if clip_level is None:
            clip_level = float(self.rng.uniform(0.3, 0.9))

        tau_ms = float(self.rng.uniform(*self.tau_range))

        saturated, original = saturate(
            signal,
            clip_level=clip_level,
            mode=sat_mode,
            poly_order=poly_order,
            apply_recovery=True,
            sample_rate=self.sample_rate,
            tau_recovery_ms=tau_ms,
            rng=self.rng,
        )

        # 클리핑 검출 (포화 후 신호 기준)
        clipped_regions = detect_clipping(
            saturated,
            threshold_ratio=self.threshold_ratio,
            consecutive=self.consecutive,
        )

        n_samples = len(signal)
        masks = {}
        if generate_all_masks:
            masks['hard'] = hard_mask(
                clipped_regions, n_samples, self.n_fft,
                self.hop_length, self.win_length
            )
            masks['soft'] = soft_mask(
                clipped_regions, n_samples, self.n_fft,
                self.hop_length, self.win_length, self.soft_temp
            )
            masks['parametric'] = parametric_mask(
                clipped_regions, n_samples, self.n_fft,
                self.hop_length, self.win_length, self.soft_temp, tau_ms
            )
        else:
            masks['soft'] = soft_mask(
                clipped_regions, n_samples, self.n_fft,
                self.hop_length, self.win_length, self.soft_temp
            )

        return saturated, {
            'masks': masks,
            'clipped_regions': clipped_regions,
            'clip_level': clip_level,
            'tau_recovery_ms': tau_ms,
            'sat_mode': sat_mode,
        }


# ─────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────

if __name__ == '__main__':
    fs = 16000
    duration = 4.0
    rng = np.random.default_rng(42)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # 테스트: 440Hz 정현파
    src = 0.9 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    sim = SaturationSimulator(sample_rate=fs, rng=rng)
    sat_sig, info = sim(src, clip_level=0.5, sat_mode='soft')

    print(f'원본 max: {np.max(np.abs(src)):.3f}')
    print(f'포화 후 max: {np.max(np.abs(sat_sig)):.3f}')
    print(f"클리핑 비율: {info['clipped_regions'].mean():.3f}")
    for k, m in info['masks'].items():
        print(f"  {k} mask — shape: {m.shape}, mean: {m.mean():.4f}, max: {m.max():.4f}")
