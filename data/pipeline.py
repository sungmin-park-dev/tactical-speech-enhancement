"""
pipeline.py
합성 데이터 파이프라인 — Military / General 데이터셋 생성 통합

실험 설계서 v3 §4 사양:
  1. LibriSpeech clean 음성 → BC 시뮬 (포화 없음)
  2. LibriSpeech clean 음성 → 소음 혼합 → AC 포화 시뮬 → 마스크 생성
  3. Military / General 환경 설정 + 분포 검증 통계 수집
"""

import os
import yaml
import warnings
from pathlib import Path
from typing import Optional, Literal, Dict, Tuple, List

import numpy as np
import soundfile as sf

from data.bc_simulator import BCSimulator
from data.saturation import SaturationSimulator
from data.noise_mixer import NoiseMixer, scan_audio_files


# ─────────────────────────────────────────────
# 설정 로더
# ─────────────────────────────────────────────

def load_config(config_path: str = 'configs/data_config.yaml') -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 단일 샘플 생성
# ─────────────────────────────────────────────

class SampleGenerator:
    """
    단일 샘플 생성: clean AC → (BC_sim, AC_noisy, masks, metadata).

    흐름:
        clean_speech
          ├─→ BCSimulator → BC_clean (모델 입력, 포화 없음)
          └─→ NoiseMixer → AC_mixed → SaturationSimulator → AC_sat (+ masks)
    """

    def __init__(
        self,
        env: Literal['military', 'general'],
        bc_sim: BCSimulator,
        sat_sim: SaturationSimulator,
        noise_mixer: NoiseMixer,
        sat_probability: float = 0.6,
        sat_mode_weights: Optional[Dict[str, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.env = env
        self.bc_sim = bc_sim
        self.sat_sim = sat_sim
        self.noise_mixer = noise_mixer
        self.rng = rng or np.random.default_rng()

        # saturation 적용 확률
        self.sat_prob = sat_probability

        # 클리핑 방식 가중치 (hard / soft / poly)
        if sat_mode_weights is None:
            self.sat_mode_weights = {'hard': 0.3, 'soft': 0.5, 'poly': 0.2}
        else:
            self.sat_mode_weights = sat_mode_weights

        self._sat_modes = list(self.sat_mode_weights.keys())
        self._sat_probs = np.array(list(self.sat_mode_weights.values()))
        self._sat_probs /= self._sat_probs.sum()

    def generate(
        self,
        clean_speech: np.ndarray,
        impulse_ratio: Optional[float] = None,
    ) -> Dict:
        """
        Parameters
        ----------
        clean_speech : (N,) 정규화된 깨끗한 음성
        impulse_ratio: Military 소음 충격음 밀도 (None이면 환경 기본값)

        Returns
        -------
        dict:
          'bc_signal'     : (N,) BC 시뮬 신호 (모델 입력 BC 채널, 포화 없음)
          'ac_noisy'      : (N,) AC 소음 + 포화 신호 (모델 입력 AC 채널)
          'clean'         : (N,) 깨끗한 음성 (훈련 타깃)
          'masks'         : {'hard': (T,F), 'soft': (T,F), 'parametric': (T,F)}
          'meta'          : {snr_db, sat_applied, sat_mode, clip_level, tau_ms,
                             clip_ratio, impulse_ratio, env}
        """
        n = len(clean_speech)

        # 1. BC 시뮬 (포화 없음)
        bc_signal = self.bc_sim(clean_speech)

        # 2. AC 소음 혼합
        if impulse_ratio is None:
            if self.env == 'military':
                impulse_ratio = float(self.rng.uniform(0.4, 0.6))
            else:
                impulse_ratio = float(self.rng.uniform(0.0, 0.1))

        ac_mixed, clean, noise, snr_db = self.noise_mixer.mix(
            clean_speech, impulse_ratio=impulse_ratio
        )

        # 3. AC에만 포화 적용 (마스크도 AC 기준)
        sat_applied = self.rng.random() < self.sat_prob
        if sat_applied:
            sat_mode = str(self.rng.choice(self._sat_modes, p=self._sat_probs))
            ac_noisy, sat_info = self.sat_sim(ac_mixed, sat_mode=sat_mode)
            masks = sat_info['masks']
            clip_level = sat_info['clip_level']
            tau_ms = sat_info['tau_recovery_ms']
            clip_ratio = float(sat_info['clipped_regions'].mean())
        else:
            ac_noisy = ac_mixed
            sat_mode = 'none'
            clip_level = 0.0
            tau_ms = 0.0
            clip_ratio = 0.0
            n_fft = self.sat_sim.n_fft
            hop = self.sat_sim.hop_length
            n_frames = 1 + max(0, (n - n_fft) // hop)
            n_freq = n_fft // 2 + 1
            zero_mask = np.zeros((n_frames, n_freq), dtype=np.float32)
            masks = {
                'hard': zero_mask.copy(),
                'soft': zero_mask.copy(),
                'parametric': zero_mask.copy(),
            }

        return {
            'bc_signal': bc_signal,
            'ac_noisy':  ac_noisy,
            'clean':     clean,
            'masks':     masks,
            'meta': {
                'env':          self.env,
                'snr_db':       snr_db,
                'sat_applied':  sat_applied,
                'sat_mode':     sat_mode,
                'clip_level':   clip_level,
                'tau_ms':       tau_ms,
                'clip_ratio':   clip_ratio,
                'impulse_ratio': impulse_ratio,
            },
        }


# ─────────────────────────────────────────────
# 데이터셋 빌더
# ─────────────────────────────────────────────

class DatasetBuilder:
    """
    LibriSpeech 화자 기준 분할 → Military / General 합성 데이터셋 빌드.

    사용법:
        builder = DatasetBuilder.from_config('configs/data_config.yaml')
        stats = builder.build(split='train', env='military', n_samples=5000)
    """

    def __init__(
        self,
        speech_files: List[str],
        env: Literal['military', 'general'],
        output_dir: str,
        bc_sim: BCSimulator,
        sat_sim: SaturationSimulator,
        noise_mixer: NoiseMixer,
        sample_rate: int = 16000,
        segment_samples: int = 64000,
        sat_probability: float = 0.6,
        rng: Optional[np.random.Generator] = None,
    ):
        self.speech_files = speech_files
        self.env = env
        self.output_dir = Path(output_dir)
        self.fs = sample_rate
        self.seg_len = segment_samples
        self.rng = rng or np.random.default_rng()

        self.generator = SampleGenerator(
            env=env,
            bc_sim=bc_sim,
            sat_sim=sat_sim,
            noise_mixer=noise_mixer,
            sat_probability=sat_probability,
            rng=self.rng,
        )

    def _load_random_segment(self) -> Optional[np.ndarray]:
        """랜덤 음성 파일에서 세그먼트 크롭."""
        if not self.speech_files:
            warnings.warn('speech_files 비어 있음 — 사인파 폴백 사용')
            t = np.linspace(0, self.seg_len / self.fs, self.seg_len, endpoint=False)
            return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        path = str(self.rng.choice(self.speech_files))
        try:
            audio, sr = sf.read(path, dtype='float32', always_2d=False)
        except Exception as e:
            warnings.warn(f'파일 로드 실패: {path} — {e}')
            return None

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # 리샘플
        if sr != self.fs:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.fs)
            except ImportError:
                from scipy.signal import resample_poly
                import math
                g = math.gcd(self.fs, sr)
                audio = resample_poly(audio, self.fs // g, sr // g).astype(np.float32)

        if len(audio) < self.seg_len:
            # 반복 패딩
            repeats = (self.seg_len // len(audio)) + 2
            audio = np.tile(audio, repeats)

        start = int(self.rng.integers(0, len(audio) - self.seg_len + 1))
        seg = audio[start:start + self.seg_len]

        # RMS 정규화
        rms = np.sqrt(np.mean(seg ** 2))
        if rms > 1e-6:
            seg = (seg / rms) * 0.1

        return seg.astype(np.float32)

    def build(
        self,
        n_samples: int,
        split: str = 'train',
        save_audio: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        n_samples개 샘플 생성 + 저장 + 통계 반환.

        저장 구조:
            output_dir/{env}/{split}/
                {i:06d}_bc.npy
                {i:06d}_ac.npy
                {i:06d}_clean.npy
                {i:06d}_mask_soft.npy
                {i:06d}_mask_hard.npy
                {i:06d}_mask_param.npy
                metadata.npz  (전체 메타 배열)

        반환: 통계 dict (snr_db, clip_ratio, impulse_ratio 분포)
        """
        out_dir = self.output_dir / self.env / split
        out_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            'snr_db': [],
            'clip_ratio': [],
            'impulse_ratio': [],
            'sat_applied': [],
        }

        i = 0
        failures = 0
        while i < n_samples:
            seg = self._load_random_segment()
            if seg is None:
                failures += 1
                if failures > 100:
                    warnings.warn('연속 100회 세그먼트 로드 실패 — 중단')
                    break
                continue

            sample = self.generator.generate(seg)

            if save_audio:
                prefix = out_dir / f'{i:06d}'
                np.save(f'{prefix}_bc.npy', sample['bc_signal'])
                np.save(f'{prefix}_ac.npy', sample['ac_noisy'])
                np.save(f'{prefix}_clean.npy', sample['clean'])
                np.save(f'{prefix}_mask_hard.npy', sample['masks']['hard'])
                np.save(f'{prefix}_mask_soft.npy', sample['masks']['soft'])
                np.save(f'{prefix}_mask_param.npy', sample['masks']['parametric'])

            meta = sample['meta']
            stats['snr_db'].append(meta['snr_db'])
            stats['clip_ratio'].append(meta['clip_ratio'])
            stats['impulse_ratio'].append(meta['impulse_ratio'])
            stats['sat_applied'].append(float(meta['sat_applied']))

            i += 1
            if verbose and (i % 100 == 0):
                print(f'  [{self.env}/{split}] {i}/{n_samples} 생성 완료...')

        # 메타 저장
        np.savez(out_dir / 'metadata.npz', **{k: np.array(v) for k, v in stats.items()})

        # 통계 요약
        summary = {}
        for k, v in stats.items():
            arr = np.array(v)
            summary[k] = {'mean': float(arr.mean()), 'std': float(arr.std()),
                          'min': float(arr.min()), 'max': float(arr.max())}

        if verbose:
            print(f'\n=== [{self.env}/{split}] 데이터셋 통계 ===')
            for k, s in summary.items():
                print(f'  {k}: mean={s["mean"]:.3f}, std={s["std"]:.3f}, '
                      f'min={s["min"]:.3f}, max={s["max"]:.3f}')

        return summary

    @classmethod
    def from_config(
        cls,
        config_path: str,
        env: Literal['military', 'general'],
        speech_files: List[str],
        split: str = 'train',
        rng: Optional[np.random.Generator] = None,
    ) -> 'DatasetBuilder':
        """설정 파일에서 DatasetBuilder 인스턴스 생성."""
        cfg = load_config(config_path)
        fs = cfg['audio']['sample_rate']
        seg = cfg['audio']['segment_samples']

        if rng is None:
            rng = np.random.default_rng(cfg['split']['seed'])

        bc_sim = BCSimulator(
            sample_rate=fs,
            peak_fc=cfg['bc_simulator']['peak_eq']['center_freq'],
            peak_Q=cfg['bc_simulator']['peak_eq']['q_factor'],
            peak_gain_db=cfg['bc_simulator']['peak_eq']['gain_db'],
            lpf_cutoff=cfg['bc_simulator']['lpf']['cutoff_freq'],
            lpf_order=cfg['bc_simulator']['lpf']['order'],
            rng=rng,
        )

        sat_cfg = cfg['saturation']
        sat_sim = SaturationSimulator(
            sample_rate=fs,
            n_fft=cfg['stft']['n_fft'],
            hop_length=cfg['stft']['hop_length'],
            win_length=cfg['stft']['win_length'],
            threshold_ratio=sat_cfg['detection']['threshold_ratio'],
            consecutive_samples=sat_cfg['detection']['consecutive_samples'],
            soft_temperature=sat_cfg['soft_mask_temperature'],
            tau_recovery_ms_range=tuple(sat_cfg['tau_recovery_ms_range']),
            rng=rng,
        )

        env_cfg = cfg[env]
        snr_range = tuple(env_cfg['snr_range'])

        paths_cfg = cfg['paths']
        noise_mixer = NoiseMixer(
            env=env,
            real_noise_dir=paths_cfg.get('freesound_root'),
            demand_dir=paths_cfg.get('demand_root'),
            musan_dir=paths_cfg.get('musan_root'),
            wham_dir=paths_cfg.get('wham_root'),
            sample_rate=fs,
            snr_range=snr_range,
            rng=rng,
        )

        output_dir = paths_cfg['output_root']
        dataset_cfg = cfg['dataset']
        n_per_split = {'train': dataset_cfg['train_samples_per_env'],
                       'val':   dataset_cfg['val_samples_per_env'],
                       'test':  dataset_cfg['test_samples_per_env']}

        sat_prob = (0.6 if env == 'military' else 0.3)

        return cls(
            speech_files=speech_files,
            env=env,
            output_dir=output_dir,
            bc_sim=bc_sim,
            sat_sim=sat_sim,
            noise_mixer=noise_mixer,
            sample_rate=fs,
            segment_samples=seg,
            sat_probability=sat_prob,
            rng=rng,
        )


# ─────────────────────────────────────────────
# 화자 기준 분할 유틸리티
# ─────────────────────────────────────────────

def split_by_speaker(
    speech_files: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    LibriSpeech 경로에서 화자 ID를 추출해 speaker-independent 분할.

    LibriSpeech 경로 형식:
        .../train-clean-100/SPEAKER_ID/CHAPTER_ID/FILE.flac
    """
    # 화자 ID 추출 (경로 기반)
    speaker_files: Dict[str, List[str]] = {}
    for f in speech_files:
        parts = Path(f).parts
        # 화자 ID는 LibriSpeech에서 파일명의 첫 번째 '-' 앞 숫자
        try:
            fname = Path(f).stem  # e.g., '1234-56789-0001'
            speaker_id = fname.split('-')[0]
        except Exception:
            speaker_id = 'unknown'
        speaker_files.setdefault(speaker_id, []).append(f)

    speakers = sorted(speaker_files.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train:n_train + n_val]
    test_speakers = speakers[n_train + n_val:]

    result = {'train': [], 'val': [], 'test': []}
    for sp in train_speakers:
        result['train'].extend(speaker_files[sp])
    for sp in val_speakers:
        result['val'].extend(speaker_files[sp])
    for sp in test_speakers:
        result['test'].extend(speaker_files[sp])

    print(f'화자 기준 분할: train={len(train_speakers)} / val={len(val_speakers)} / test={len(test_speakers)} 화자')
    print(f'  파일 수: train={len(result["train"])} / val={len(result["val"])} / test={len(result["test"])}')
    return result


# ─────────────────────────────────────────────
# CLI 엔트리포인트
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='합성 데이터셋 생성')
    parser.add_argument('--config', default='configs/data_config.yaml')
    parser.add_argument('--env', choices=['military', 'general', 'both'], default='both')
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='all')
    parser.add_argument('--n_train', type=int, default=None, help='train 샘플 수 (기본: config)')
    parser.add_argument('--n_val', type=int, default=None, help='val 샘플 수')
    parser.add_argument('--n_test', type=int, default=None, help='test 샘플 수')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true', help='파일 저장 없이 통계만')
    args = parser.parse_args()

    cfg = load_config(args.config)
    fs = cfg['audio']['sample_rate']

    # LibriSpeech 파일 스캔
    libri_root = cfg['paths']['librispeech_root']
    print(f'LibriSpeech 루트: {libri_root}')
    if os.path.isdir(libri_root):
        all_files = scan_audio_files(libri_root, extensions=('.flac', '.wav'))
        print(f'  총 {len(all_files)}개 파일 발견')
    else:
        warnings.warn(f'LibriSpeech 경로 없음: {libri_root} — 더미 데이터로 실행')
        all_files = []

    # 화자 기준 분할
    split_files = split_by_speaker(
        all_files,
        train_ratio=cfg['split']['train'],
        val_ratio=cfg['split']['val'],
        seed=args.seed,
    )

    # 데이터셋 크기
    ds_cfg = cfg['dataset']
    n_map = {
        'train': args.n_train or ds_cfg['train_samples_per_env'],
        'val':   args.n_val   or ds_cfg['val_samples_per_env'],
        'test':  args.n_test  or ds_cfg['test_samples_per_env'],
    }

    envs = ['military', 'general'] if args.env == 'both' else [args.env]
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]

    for env in envs:
        for sp in splits:
            print(f'\n=== {env.upper()} / {sp.upper()} 생성 시작 ===')
            rng = np.random.default_rng(args.seed)
            builder = DatasetBuilder.from_config(
                config_path=args.config,
                env=env,
                speech_files=split_files[sp],
                split=sp,
                rng=rng,
            )
            summary = builder.build(
                n_samples=n_map[sp],
                split=sp,
                save_audio=not args.dry_run,
                verbose=True,
            )

    print('\n=== 데이터셋 생성 완료 ===')


if __name__ == '__main__':
    main()
