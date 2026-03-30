"""
bc_simulator.py
골전도(BC) 센서 신호 시뮬레이터

실험 설계서 v3 §4.1 사양:
  - IIR 필터: 2차 피킹 EQ (1.5kHz, Q=2) + 4차 Butterworth LPF (4kHz)
  - 신체 잡음: 심박(1Hz sine, -30dB) + 임의 임펄스(보행, 0.5~10Hz)
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Optional
import warnings


# ─────────────────────────────────────────────
# IIR 필터 설계
# ─────────────────────────────────────────────

def _peaking_eq_coeffs(fc: float, Q: float, gain_db: float, fs: float):
    """
    2차 피킹 EQ 바이쿼드 계수 (Audio EQ Cookbook, Zolzer).
    fc   : 중심 주파수 (Hz)
    Q    : 품질 인자
    gain_db: 부스트/컷 (dB)
    반환: (b, a) — scipy.signal 규약
    """
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    return np.array([b0, b1, b2]) / a0, np.array([1.0, a1 / a0, a2 / a0])


def _butterworth_lpf_coeffs(cutoff: float, order: int, fs: float):
    """4차 Butterworth LPF 계수 (sos 형식)."""
    sos = sp_signal.butter(order, cutoff, btype='low',
                           fs=fs, output='sos')
    return sos


def design_bc_filter(
    sample_rate: int = 16000,
    peak_fc: float = 1500.0,
    peak_Q: float = 2.0,
    peak_gain_db: float = 6.0,
    lpf_cutoff: float = 4000.0,
    lpf_order: int = 4,
):
    """
    BC 센서 주파수 응답 근사 IIR 필터를 설계하고 반환.

    반환값: dict {'b_eq': ..., 'a_eq': ..., 'sos_lpf': ...}
    """
    b_eq, a_eq = _peaking_eq_coeffs(peak_fc, peak_Q, peak_gain_db, sample_rate)
    sos_lpf = _butterworth_lpf_coeffs(lpf_cutoff, lpf_order, sample_rate)
    return {'b_eq': b_eq, 'a_eq': a_eq, 'sos_lpf': sos_lpf}


def apply_bc_filter(
    audio: np.ndarray,
    bc_filter: dict,
    zi_eq=None,
    zi_lpf=None,
):
    """
    BC IIR 필터 적용 (오프라인 / 스트리밍 호환).

    audio    : (N,) float32/float64
    bc_filter: design_bc_filter() 반환값
    zi_*     : streaming 초기 상태 (None이면 0 초기화)
    반환     : (filtered, zi_eq, zi_lpf)  — zi_*는 streaming 모드 시 재사용
    """
    # 피킹 EQ 적용
    if zi_eq is None:
        zi_eq = sp_signal.lfilter_zi(bc_filter['b_eq'], bc_filter['a_eq'])
        zi_eq = zi_eq * audio[0]

    y_eq, zi_eq_out = sp_signal.lfilter(
        bc_filter['b_eq'], bc_filter['a_eq'], audio, zi=zi_eq
    )

    # Butterworth LPF 적용
    if zi_lpf is None:
        zi_lpf = sp_signal.sosfilt_zi(bc_filter['sos_lpf'])
        zi_lpf = zi_lpf * y_eq[0]

    y_lpf, zi_lpf_out = sp_signal.sosfilt(
        bc_filter['sos_lpf'], y_eq, zi=zi_lpf
    )

    return y_lpf, zi_eq_out, zi_lpf_out


# ─────────────────────────────────────────────
# 신체 잡음 생성
# ─────────────────────────────────────────────

def generate_body_noise(
    n_samples: int,
    sample_rate: int = 16000,
    heartbeat_freq: float = 1.0,
    heartbeat_amp_db: float = -30.0,
    footstep_freq_range: tuple = (0.5, 10.0),
    footstep_n_per_sec_range: tuple = (1, 5),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    심박 + 보행 임펄스로 구성된 신체 잡음을 생성.

    반환: (n_samples,) 신체 잡음 신호 (정규화되지 않은 RMS 기준 -30dB 수준)
    """
    if rng is None:
        rng = np.random.default_rng()

    noise = np.zeros(n_samples, dtype=np.float64)
    t = np.arange(n_samples) / sample_rate

    # 1) 심박: 주기성 sine (1Hz)
    hb_amp = 10 ** (heartbeat_amp_db / 20.0)
    heartbeat = hb_amp * np.sin(2 * np.pi * heartbeat_freq * t)
    noise += heartbeat

    # 2) 보행 임펄스
    duration = n_samples / sample_rate
    n_steps = int(rng.uniform(*footstep_n_per_sec_range) * duration)
    step_times = rng.uniform(0, duration, size=n_steps)

    for st in step_times:
        idx = int(st * sample_rate)
        if idx < n_samples:
            # 짧은 감쇠 임펄스 (5ms)
            imp_len = int(0.005 * sample_rate)
            imp_t = np.arange(min(imp_len, n_samples - idx)) / sample_rate
            imp = hb_amp * 0.5 * np.exp(-imp_t / 0.002) * rng.choice([-1, 1])
            noise[idx:idx + len(imp_t)] += imp

    return noise.astype(np.float32)


# ─────────────────────────────────────────────
# 통합 BC 시뮬레이터
# ─────────────────────────────────────────────

class BCSimulator:
    """
    공기전도(AC) 신호로부터 골전도(BC) 시뮬 신호 생성.

    사용법:
        sim = BCSimulator(sample_rate=16000)
        bc_signal = sim(ac_signal)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        peak_fc: float = 1500.0,
        peak_Q: float = 2.0,
        peak_gain_db: float = 6.0,
        lpf_cutoff: float = 4000.0,
        lpf_order: int = 4,
        add_body_noise: bool = True,
        heartbeat_freq: float = 1.0,
        heartbeat_amp_db: float = -30.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.sample_rate = sample_rate
        self.bc_filter = design_bc_filter(
            sample_rate, peak_fc, peak_Q, peak_gain_db, lpf_cutoff, lpf_order
        )
        self.add_body_noise = add_body_noise
        self.heartbeat_freq = heartbeat_freq
        self.heartbeat_amp_db = heartbeat_amp_db
        self.rng = rng or np.random.default_rng()

    def __call__(self, ac_signal: np.ndarray) -> np.ndarray:
        """
        ac_signal: (N,) 정규화된 공기전도 음성
        반환     : (N,) BC 시뮬 신호
        """
        ac_signal = np.asarray(ac_signal, dtype=np.float64)

        # IIR 필터 적용
        bc_filtered, _, _ = apply_bc_filter(ac_signal, self.bc_filter)

        # 신체 잡음 추가
        if self.add_body_noise:
            body = generate_body_noise(
                n_samples=len(bc_filtered),
                sample_rate=self.sample_rate,
                heartbeat_freq=self.heartbeat_freq,
                heartbeat_amp_db=self.heartbeat_amp_db,
                rng=self.rng,
            )
            bc_filtered = bc_filtered + body

        return bc_filtered.astype(np.float32)

    def filter_response(self, n_freqs: int = 512):
        """
        BC 필터 주파수 응답 반환 (분석/디버깅용).
        반환: (frequencies, magnitude_dB)
        """
        # 피킹 EQ
        w_eq, h_eq = sp_signal.freqz(
            self.bc_filter['b_eq'], self.bc_filter['a_eq'],
            worN=n_freqs, fs=self.sample_rate
        )
        # LPF
        w_lpf, h_lpf = sp_signal.sosfreqz(
            self.bc_filter['sos_lpf'], worN=n_freqs, fs=self.sample_rate
        )
        h_total = h_eq * h_lpf
        mag_db = 20 * np.log10(np.abs(h_total) + 1e-12)
        return w_eq, mag_db


# ─────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fs = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # 테스트 신호: 화이트 노이즈
    rng = np.random.default_rng(0)
    test_signal = rng.standard_normal(len(t)).astype(np.float32) * 0.1

    sim = BCSimulator(sample_rate=fs, rng=rng)
    bc_out = sim(test_signal)

    # 주파수 응답 플롯
    freqs, mag_db = sim.filter_response()
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, mag_db)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('BC Simulator Filter Response')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('bc_filter_response.png', dpi=150)
    print(f'[bc_simulator] 출력 신호 shape: {bc_out.shape}, RMS: {np.sqrt(np.mean(bc_out**2)):.4f}')
    print('필터 응답 저장: bc_filter_response.png')
