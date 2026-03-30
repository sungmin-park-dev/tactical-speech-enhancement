"""
noise_mixer.py
소음 혼합 모듈

실험 설계서 v3 §4.2 사양:
  Military: 실제 녹음(Freesound/AudioSet) 50% + 합성 충격음 50%
            SNR -20~+5 dB, 포화율 30~100%
  General : DEMAND + MUSAN + WHAM! 소음
            SNR -10~+15 dB, 포화율 0~50%
"""

import os
import random
import warnings
from pathlib import Path
from typing import Optional, Literal, List, Tuple

import numpy as np
import soundfile as sf


# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def load_audio(
    path: str,
    target_sr: int = 16000,
    target_len: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    오디오 파일 로드 → 모노, 리샘플, 길이 조정.
    target_len: None이면 전체. 지정 시 랜덤 크롭 또는 반복 패딩.
    """
    try:
        audio, sr = sf.read(path, dtype='float32', always_2d=False)
    except Exception as e:
        warnings.warn(f'오디오 로드 실패: {path} — {e}')
        if target_len is not None:
            return np.zeros(target_len, dtype=np.float32)
        return np.zeros(target_sr, dtype=np.float32)

    # 모노 변환
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # 리샘플 (librosa 사용 가능하면 우선, 없으면 scipy)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            from scipy.signal import resample_poly
            gcd = np.gcd(target_sr, sr)
            audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)

    # 길이 조정
    if target_len is not None:
        if rng is None:
            rng = np.random.default_rng()
        if len(audio) >= target_len:
            # 랜덤 크롭
            start = rng.integers(0, len(audio) - target_len + 1)
            audio = audio[start:start + target_len]
        else:
            # 반복 패딩
            repeats = (target_len // len(audio)) + 2
            audio = np.tile(audio, repeats)[:target_len]

    return audio.astype(np.float32)


def mix_at_snr(
    speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> Tuple[np.ndarray, float]:
    """
    목표 SNR로 소음 스케일링 후 혼합.

    반환: (noisy_signal, noise_scale)
    """
    speech_rms = np.sqrt(np.mean(speech ** 2) + 1e-8)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-8)

    target_noise_rms = speech_rms / (10 ** (snr_db / 20.0))
    scale = target_noise_rms / noise_rms

    noisy = speech + scale * noise
    return noisy.astype(np.float32), scale


def scan_audio_files(directory: str, extensions: tuple = ('.wav', '.flac', '.mp3')) -> List[str]:
    """디렉토리 내 오디오 파일 목록 반환."""
    result = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(extensions):
                result.append(os.path.join(root, f))
    return sorted(result)


# ─────────────────────────────────────────────
# 소음 DB 래퍼
# ─────────────────────────────────────────────

class NoiseBank:
    """
    소음 파일 DB 래퍼. 경로 목록에서 랜덤 샘플링.
    """

    def __init__(
        self,
        paths: List[str],
        sample_rate: int = 16000,
        rng: Optional[np.random.Generator] = None,
        name: str = 'unknown',
    ):
        self.paths = paths
        self.sample_rate = sample_rate
        self.rng = rng or np.random.default_rng()
        self.name = name

        if not paths:
            warnings.warn(f'NoiseBank [{name}]: 파일 없음. 가우시안 폴백 사용.')

    def sample(self, n_samples: int) -> np.ndarray:
        """랜덤 소음 클립 반환 (n_samples 길이)."""
        if not self.paths:
            # 폴백: 가우시안 노이즈
            return (self.rng.standard_normal(n_samples) * 0.1).astype(np.float32)

        path = self.rng.choice(self.paths)
        return load_audio(path, self.sample_rate, n_samples, self.rng)

    def __len__(self):
        return len(self.paths)

    @classmethod
    def from_directory(cls, directory: str, **kwargs) -> 'NoiseBank':
        paths = scan_audio_files(directory)
        return cls(paths, name=os.path.basename(directory), **kwargs)


# ─────────────────────────────────────────────
# Military 소음 믹서
# ─────────────────────────────────────────────

class MilitaryNoiseMixer:
    """
    실제 군용 소음 녹음 50% + 합성 충격음 50% 혼합.

    사용법:
        mixer = MilitaryNoiseMixer(real_paths=[...], sample_rate=16000)
        noise = mixer.sample(n_samples=64000, impulse_ratio=0.5)
    """

    def __init__(
        self,
        real_noise_paths: Optional[List[str]] = None,
        real_noise_directory: Optional[str] = None,
        sample_rate: int = 16000,
        real_noise_fraction: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ):
        self.fs = sample_rate
        self.real_frac = real_noise_fraction
        self.rng = rng or np.random.default_rng()

        # 실제 소음 DB
        paths = real_noise_paths or []
        if real_noise_directory and os.path.isdir(real_noise_directory):
            paths += scan_audio_files(real_noise_directory)
        self.real_bank = NoiseBank(paths, sample_rate, self.rng, 'military_real')

        # 충격음 생성기는 noise_mixer가 아닌 pipeline에서 주입
        # (순환 임포트 방지 — lazy import)

    def sample(
        self,
        n_samples: int,
        impulse_ratio: float = 0.5,
        return_components: bool = False,
    ) -> np.ndarray:
        """
        Military 소음 클립 생성.
        impulse_ratio: 충격음 밀도 (0~1)
        """
        from data.impulse_generator import generate_impulse_noise

        noise = np.zeros(n_samples, dtype=np.float32)

        # 실제 소음 성분
        if self.rng.random() < self.real_frac:
            real_noise = self.real_bank.sample(n_samples)
            noise += real_noise * 0.5

        # 합성 충격음 성분
        if self.rng.random() < (1.0 - self.real_frac) or impulse_ratio > 0:
            imp = generate_impulse_noise(
                n_samples=n_samples,
                fs=self.fs,
                impulse_type='random',
                events_per_sec_range=(impulse_ratio * 0.5, impulse_ratio * 4.0 + 0.5),
                rng=self.rng,
            )
            noise += imp * 0.5

        # 피크 정규화
        peak = np.max(np.abs(noise))
        if peak > 1.0:
            noise = noise / peak

        return noise


# ─────────────────────────────────────────────
# General 소음 믹서
# ─────────────────────────────────────────────

class GeneralNoiseMixer:
    """
    DEMAND / MUSAN / WHAM! 소음 DB 혼합.

    사용법:
        mixer = GeneralNoiseMixer(
            demand_dir='data/raw/DEMAND',
            musan_dir='data/raw/MUSAN',
            wham_dir='data/raw/WHAM',
        )
        noise = mixer.sample(n_samples=64000)
    """

    def __init__(
        self,
        demand_dir: Optional[str] = None,
        musan_dir: Optional[str] = None,
        wham_dir: Optional[str] = None,
        sample_rate: int = 16000,
        rng: Optional[np.random.Generator] = None,
    ):
        self.fs = sample_rate
        self.rng = rng or np.random.default_rng()

        # 각 소음 DB 스캔
        self.banks: List[NoiseBank] = []

        for name, directory in [
            ('DEMAND', demand_dir),
            ('MUSAN', musan_dir),
            ('WHAM', wham_dir),
        ]:
            if directory and os.path.isdir(directory):
                bank = NoiseBank.from_directory(directory, sample_rate=sample_rate, rng=self.rng)
                bank.name = name
                self.banks.append(bank)
                print(f'  [{name}] {len(bank)} 파일 로드')
            else:
                # 폴백 빈 은행 (가우시안 폴백 내장)
                self.banks.append(NoiseBank([], sample_rate, self.rng, name))

        if not any(len(b) > 0 for b in self.banks):
            warnings.warn('GeneralNoiseMixer: 모든 소음 DB 비어 있음. 가우시안 폴백 사용.')

    def sample(self, n_samples: int) -> np.ndarray:
        """가용 소음 DB 중 랜덤으로 하나 선택해 샘플링."""
        # 파일이 있는 bank 우선
        available = [b for b in self.banks if len(b) > 0]
        if not available:
            available = self.banks  # 폴백

        bank = self.rng.choice(available)
        return bank.sample(n_samples)


# ─────────────────────────────────────────────
# 통합 소음 믹서
# ─────────────────────────────────────────────

class NoiseMixer:
    """
    환경별 소음 믹서 + SNR 혼합 통합 인터페이스.

    사용법:
        mixer = NoiseMixer(
            env='military',
            real_noise_dir='data/raw/Freesound',
            sample_rate=16000,
        )
        noisy, clean, noise, snr = mixer.mix(speech)
    """

    def __init__(
        self,
        env: Literal['military', 'general'] = 'general',
        real_noise_dir: Optional[str] = None,
        demand_dir: Optional[str] = None,
        musan_dir: Optional[str] = None,
        wham_dir: Optional[str] = None,
        sample_rate: int = 16000,
        snr_range: Optional[Tuple[float, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.env = env
        self.fs = sample_rate
        self.rng = rng or np.random.default_rng()

        # SNR 범위
        if snr_range is not None:
            self.snr_range = snr_range
        elif env == 'military':
            self.snr_range = (-20.0, 5.0)
        else:
            self.snr_range = (-10.0, 15.0)

        if env == 'military':
            self._mixer = MilitaryNoiseMixer(
                real_noise_directory=real_noise_dir,
                sample_rate=sample_rate,
                rng=self.rng,
            )
        else:
            self._mixer = GeneralNoiseMixer(
                demand_dir=demand_dir,
                musan_dir=musan_dir,
                wham_dir=wham_dir,
                sample_rate=sample_rate,
                rng=self.rng,
            )

    def mix(
        self,
        speech: np.ndarray,
        snr_db: Optional[float] = None,
        impulse_ratio: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Parameters
        ----------
        speech       : (N,) 깨끗한 음성
        snr_db       : None이면 ranomg snr_range에서 선택
        impulse_ratio: Military 모드에서만 사용

        Returns
        -------
        noisy  : (N,) 소음 혼합 신호
        clean  : (N,) 원본 음성 (= speech)
        noise  : (N,) 스케일된 소음 신호
        snr_db : 실제 적용된 SNR
        """
        n_samples = len(speech)
        if snr_db is None:
            snr_db = float(self.rng.uniform(*self.snr_range))

        # 소음 샘플링
        if self.env == 'military':
            noise_raw = self._mixer.sample(n_samples, impulse_ratio=impulse_ratio)
        else:
            noise_raw = self._mixer.sample(n_samples)

        noisy, scale = mix_at_snr(speech, noise_raw, snr_db)
        noise_scaled = (noise_raw * scale).astype(np.float32)

        return noisy, speech.astype(np.float32), noise_scaled, snr_db


# ─────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────

if __name__ == '__main__':
    fs = 16000
    rng = np.random.default_rng(42)

    # 더미 음성
    t = np.linspace(0, 4, fs * 4, endpoint=False)
    speech = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    # General 믹서 (DB 없이 폴백)
    gen_mixer = NoiseMixer(env='general', sample_rate=fs, rng=rng)
    noisy, clean, noise, snr = gen_mixer.mix(speech, snr_db=0.0)
    print(f'[General] SNR={snr:.1f}dB, noisy RMS={np.sqrt(np.mean(noisy**2)):.4f}')

    # Military 믹서 (충격음 합성 폴백)
    mil_mixer = NoiseMixer(env='military', sample_rate=fs, rng=rng)
    noisy_m, clean_m, noise_m, snr_m = mil_mixer.mix(speech, snr_db=-5.0, impulse_ratio=0.5)
    print(f'[Military] SNR={snr_m:.1f}dB, noisy RMS={np.sqrt(np.mean(noisy_m**2)):.4f}')
