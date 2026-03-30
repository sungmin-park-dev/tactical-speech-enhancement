"""
impulse_generator.py
군용 충격음 합성 (총성 / 폭발음)

실험 설계서 v3 §4.2 사양:
  총성(gunshot): 감쇠 정현파 충격파 + 마즐 블래스트 + 잔향
  폭발음(explosion): 저주파 충격파 + 광대역 잔향
"""

import numpy as np
from typing import Optional, Literal, List


def _normalize(x: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """RMS 기준 정규화. 무음이면 원본 반환."""
    rms = np.sqrt(np.mean(x ** 2))
    if rms < 1e-8:
        return x
    return (x / rms) * target_rms


# ─────────────────────────────────────────────
# 총성 합성
# ─────────────────────────────────────────────

def gunshot(
    duration_ms: float = 50.0,
    fs: int = 16000,
    f0: Optional[float] = None,
    tau_attack: Optional[float] = None,
    tau_reverb: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    총성 합성.
    초기 충격파(감쇠 정현파) + 마즐 블래스트(광대역 버스트) + 잔향.

    Parameters
    ----------
    duration_ms : 충격음 본체 길이 (ms)
    fs          : 샘플링 레이트
    f0          : 충격파 주파수 (None이면 1000~4000Hz 랜덤)
    tau_attack  : 충격파 감쇠 시간 (None이면 1~5ms 랜덤)
    tau_reverb  : 잔향 감쇠 시간 (None이면 50~200ms 랜덤)

    반환: 정규화된 총성 신호 (float32)
    """
    if rng is None:
        rng = np.random.default_rng()

    # 랜덤 파라미터
    if f0 is None:
        f0 = float(rng.uniform(1000, 4000))
    if tau_attack is None:
        tau_attack = float(rng.uniform(0.001, 0.005))
    if tau_reverb is None:
        tau_reverb = float(rng.uniform(0.05, 0.2))

    # 충격음 본체
    n_attack = int(duration_ms / 1000 * fs)
    t_attack = np.arange(n_attack) / fs

    # 초기 충격파: 감쇠 정현파
    attack = np.sin(2 * np.pi * f0 * t_attack) * np.exp(-t_attack / tau_attack)

    # 마즐 블래스트: 광대역 버스트 (duration의 처음 ~3ms)
    blast = rng.standard_normal(n_attack) * np.exp(-t_attack / 0.003)

    # 잔향: 지수 감쇠 가우시안 노이즈
    n_reverb = int(tau_reverb * fs * 3)
    t_reverb = np.arange(n_reverb) / fs
    reverb = rng.standard_normal(n_reverb) * np.exp(-t_reverb / tau_reverb)

    full = np.concatenate([attack + blast, reverb])
    return _normalize(full).astype(np.float32)


# ─────────────────────────────────────────────
# 폭발음 합성
# ─────────────────────────────────────────────

def explosion(
    duration_ms: float = 200.0,
    fs: int = 16000,
    f0: Optional[float] = None,
    tau_shock: float = 0.05,
    reverb_duration: float = 0.5,
    tau_reverb: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    폭발음 합성.
    저주파 충격파 + 광대역 잔향.

    Parameters
    ----------
    duration_ms   : 충격파 본체 길이 (ms)
    f0            : 충격파 주파수 (None이면 20~200Hz 랜덤)
    tau_shock     : 충격파 감쇠 (sec)
    reverb_duration: 잔향 길이 (sec)
    tau_reverb    : 잔향 감쇠 (sec)

    반환: 정규화된 폭발음 신호 (float32)
    """
    if rng is None:
        rng = np.random.default_rng()

    if f0 is None:
        f0 = float(rng.uniform(20, 200))

    # 충격파 본체
    n_shock = int(duration_ms / 1000 * fs)
    t_shock = np.arange(n_shock) / fs
    shock = np.sin(2 * np.pi * f0 * t_shock) * np.exp(-t_shock / tau_shock)

    # 광대역 잔향
    n_reverb = int(reverb_duration * fs)
    t_reverb = np.arange(n_reverb) / fs
    reverb = rng.standard_normal(n_reverb) * np.exp(-t_reverb / tau_reverb)

    full = np.concatenate([shock, reverb])
    return _normalize(full).astype(np.float32)


# ─────────────────────────────────────────────
# 다중 충격음 합성 시퀀스 생성
# ─────────────────────────────────────────────

def generate_impulse_noise(
    n_samples: int,
    fs: int = 16000,
    impulse_type: Literal['gunshot', 'explosion', 'random'] = 'random',
    n_events: Optional[int] = None,
    events_per_sec_range: tuple = (0.5, 4.0),
    amplitude_range: tuple = (0.3, 1.0),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    n_samples 길이 신호에 여러 충격음 이벤트를 랜덤 배치.

    Parameters
    ----------
    impulse_type : 'gunshot' | 'explosion' | 'random' (둘 다 혼합)
    n_events     : 이벤트 수 (None이면 events_per_sec_range 기준 랜덤)
    amplitude_range: 각 이벤트 진폭 범위

    반환: (n_samples,) 충격음 신호 (float32)
    """
    if rng is None:
        rng = np.random.default_rng()

    duration = n_samples / fs
    if n_events is None:
        rate = rng.uniform(*events_per_sec_range)
        n_events = max(1, int(rate * duration))

    output = np.zeros(n_samples, dtype=np.float32)

    # 이벤트 시작 위치 (균일 분포)
    start_times = sorted(rng.uniform(0, duration * 0.8, size=n_events))

    for st in start_times:
        # 충격음 유형 선택
        if impulse_type == 'random':
            choice = rng.choice(['gunshot', 'explosion'])
        else:
            choice = impulse_type

        # 충격음 생성
        if choice == 'gunshot':
            event = gunshot(fs=fs, rng=rng)
        else:
            event = explosion(fs=fs, rng=rng)

        # 진폭 스케일
        amp = float(rng.uniform(*amplitude_range))
        event = event * amp

        # 배치
        start_idx = int(st * fs)
        end_idx = min(start_idx + len(event), n_samples)
        event_len = end_idx - start_idx
        output[start_idx:end_idx] += event[:event_len]

    # 최종 클리핑 방지
    peak = np.max(np.abs(output))
    if peak > 1.0:
        output = output / peak

    return output


# ─────────────────────────────────────────────
# 배치 생성 유틸리티
# ─────────────────────────────────────────────

class ImpulseGenerator:
    """
    군용 충격음 합성기 래퍼 클래스.

    사용법:
        gen = ImpulseGenerator(sample_rate=16000)
        noise = gen.sample(n_samples=64000, impulse_ratio=0.5)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        rng: Optional[np.random.Generator] = None,
    ):
        self.fs = sample_rate
        self.rng = rng or np.random.default_rng()

    def sample(
        self,
        n_samples: int,
        impulse_ratio: float = 0.5,
        impulse_type: Literal['gunshot', 'explosion', 'random'] = 'random',
    ) -> np.ndarray:
        """
        impulse_ratio (0~1): 충격음 밀도 (이벤트 수 조정에 사용)
        반환: (n_samples,) 충격음 노이즈
        """
        duration = n_samples / self.fs
        # impulse_ratio × max 4 events/sec
        rate = impulse_ratio * 4.0
        n_events = max(1, int(rate * duration))

        return generate_impulse_noise(
            n_samples=n_samples,
            fs=self.fs,
            impulse_type=impulse_type,
            n_events=n_events,
            rng=self.rng,
        )

    def single_gunshot(self, **kwargs) -> np.ndarray:
        return gunshot(fs=self.fs, rng=self.rng, **kwargs)

    def single_explosion(self, **kwargs) -> np.ndarray:
        return explosion(fs=self.fs, rng=self.rng, **kwargs)


# ─────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────

if __name__ == '__main__':
    fs = 16000
    rng = np.random.default_rng(0)

    gs = gunshot(fs=fs, rng=rng)
    exp = explosion(fs=fs, rng=rng)
    noise = generate_impulse_noise(n_samples=fs * 4, fs=fs, rng=rng)

    print(f'총성 shape: {gs.shape}, max: {np.max(np.abs(gs)):.3f}, RMS: {np.sqrt(np.mean(gs**2)):.4f}')
    print(f'폭발음 shape: {exp.shape}, max: {np.max(np.abs(exp)):.3f}, RMS: {np.sqrt(np.mean(exp**2)):.4f}')
    print(f'충격음 시퀀스 shape: {noise.shape}, max: {np.max(np.abs(noise)):.3f}')
