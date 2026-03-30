"""
dataset.py
PyTorch Dataset / DataLoader

실험 설계서 v3 §4:
  - Military / General 환경별 Dataset
  - 화자 기준 분할 (train / val / test)
  - 3유형 마스크 (hard / soft / parametric) 선택 가능
  - On-the-fly 모드 or Pre-generated npy 디렉토리 모드
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Literal, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# ─────────────────────────────────────────────
# Pre-generated NPY 기반 Dataset
# ─────────────────────────────────────────────

class SpeechEnhancementDataset(Dataset):
    """
    pipeline.py로 사전 생성된 .npy 파일을 읽는 Dataset.

    디렉토리 구조:
        data_dir/{env}/{split}/
            {i:06d}_bc.npy         — BC 시뮬 + 포화 신호
            {i:06d}_ac.npy         — AC 소음 혼합 신호
            {i:06d}_clean.npy      — 깨끗한 음성 (타깃)
            {i:06d}_mask_hard.npy  — Hard 마스크 (T, F)
            {i:06d}_mask_soft.npy  — Soft 마스크 (T, F)
            {i:06d}_mask_param.npy — Parametric 마스크 (T, F)
    """

    def __init__(
        self,
        data_dir: str,
        env: Literal['military', 'general'],
        split: Literal['train', 'val', 'test'],
        mask_type: Literal['hard', 'soft', 'parametric'] = 'soft',
        max_samples: Optional[int] = None,
        return_all_masks: bool = False,
    ):
        self.env = env
        self.split = split
        self.mask_type = mask_type
        self.return_all_masks = return_all_masks

        self.root = Path(data_dir) / env / split
        if not self.root.exists():
            raise FileNotFoundError(
                f'Dataset 디렉토리 없음: {self.root}\n'
                f'먼저 python -m data.pipeline --env {env} --split {split} 실행'
            )

        # 파일 인덱스 수집
        self.indices = sorted(
            int(f.stem.split('_')[0])
            for f in self.root.glob('*_bc.npy')
        )
        if max_samples is not None:
            self.indices = self.indices[:max_samples]

        if len(self.indices) == 0:
            raise RuntimeError(f'Dataset 비어 있음: {self.root}')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        i = self.indices[idx]
        prefix = self.root / f'{i:06d}'

        bc = torch.from_numpy(np.load(f'{prefix}_bc.npy'))    # (N,)
        ac = torch.from_numpy(np.load(f'{prefix}_ac.npy'))    # (N,)
        clean = torch.from_numpy(np.load(f'{prefix}_clean.npy'))  # (N,)

        mask_key = {'hard': 'hard', 'soft': 'soft', 'parametric': 'param'}[self.mask_type]
        mask = torch.from_numpy(np.load(f'{prefix}_mask_{mask_key}.npy'))  # (T, F)

        item = {
            'bc': bc,
            'ac': ac,
            'clean': clean,
            'mask': mask,
            'env': self.env,
        }

        if self.return_all_masks:
            item['mask_hard']  = torch.from_numpy(np.load(f'{prefix}_mask_hard.npy'))
            item['mask_soft']  = torch.from_numpy(np.load(f'{prefix}_mask_soft.npy'))
            item['mask_param'] = torch.from_numpy(np.load(f'{prefix}_mask_param.npy'))

        return item


# ─────────────────────────────────────────────
# On-the-fly Dataset (Colab 등 메모리 절약 모드)
# ─────────────────────────────────────────────

class OnTheFlyDataset(Dataset):
    """
    LibriSpeech 음성 파일 목록 + 파이프라인 파라미터를 받아
    __getitem__에서 실시간 합성.

    장점: 디스크 공간 절약, 무한 augmentation
    단점: CPU 사용량 증가 → DataLoader num_workers 조정 필요
    """

    def __init__(
        self,
        speech_files: List[str],
        env: Literal['military', 'general'],
        config_path: str = 'configs/data_config.yaml',
        mask_type: Literal['hard', 'soft', 'parametric'] = 'soft',
        n_samples: int = 5000,
        segment_samples: int = 64000,
        sample_rate: int = 16000,
        seed: int = 42,
    ):
        self.speech_files = speech_files
        self.env = env
        self.mask_type = mask_type
        self.n_samples = n_samples
        self.seg_len = segment_samples
        self.fs = sample_rate
        self.config_path = config_path
        self.seed = seed

        # 파이프라인은 worker 초기화 시 생성 (pickling 이슈 방지)
        self._generator = None

    def _init_generator(self):
        """Worker 프로세스 내에서 lazy 초기화."""
        if self._generator is not None:
            return

        from data.pipeline import DatasetBuilder
        # worker_id 기반 시드 (재현성)
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        rng = np.random.default_rng(seed)

        builder = DatasetBuilder.from_config(
            config_path=self.config_path,
            env=self.env,
            speech_files=self.speech_files,
            rng=rng,
        )
        self._generator = builder.generator
        self._speech_files = self.speech_files
        self._rng = rng

    def _load_segment(self) -> np.ndarray:
        import soundfile as sf
        path = str(self._rng.choice(self._speech_files))
        try:
            audio, sr = sf.read(path, dtype='float32', always_2d=False)
        except Exception:
            t = np.linspace(0, self.seg_len / self.fs, self.seg_len, endpoint=False)
            return (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr != self.fs:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.fs)
            except ImportError:
                pass

        if len(audio) < self.seg_len:
            audio = np.tile(audio, (self.seg_len // len(audio)) + 2)[:self.seg_len]

        start = int(self._rng.integers(0, max(1, len(audio) - self.seg_len)))
        seg = audio[start:start + self.seg_len]
        rms = np.sqrt(np.mean(seg ** 2))
        if rms > 1e-6:
            seg = (seg / rms) * 0.1
        return seg.astype(np.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        self._init_generator()

        seg = self._load_segment()
        sample = self._generator.generate(seg)

        mask_key = self.mask_type
        mask = sample['masks'][mask_key]

        return {
            'bc':    torch.from_numpy(sample['bc_signal']),
            'ac':    torch.from_numpy(sample['ac_noisy']),
            'clean': torch.from_numpy(sample['clean']),
            'mask':  torch.from_numpy(mask),
            'env':   self.env,
        }


# ─────────────────────────────────────────────
# DataLoader 팩토리
# ─────────────────────────────────────────────

def build_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """표준 DataLoader 생성."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )


def build_dataloaders_from_config(
    config_path: str = 'configs/data_config.yaml',
    data_dir: str = 'data/processed',
    mask_type: Literal['hard', 'soft', 'parametric'] = 'soft',
    envs: List[str] = ('military', 'general'),
    batch_size: int = 8,
    num_workers: int = 4,
) -> Dict[str, Dict[str, DataLoader]]:
    """
    설정 파일 기반으로 모든 환경/분할 DataLoader 생성.

    반환:
        {
            'military': {'train': DataLoader, 'val': DataLoader, 'test': DataLoader},
            'general':  {'train': DataLoader, 'val': DataLoader, 'test': DataLoader},
        }
    """
    result = {}
    for env in envs:
        result[env] = {}
        for split in ['train', 'val', 'test']:
            try:
                ds = SpeechEnhancementDataset(
                    data_dir=data_dir,
                    env=env,
                    split=split,
                    mask_type=mask_type,
                )
                shuffle = (split == 'train')
                result[env][split] = build_dataloader(
                    ds, batch_size=batch_size, shuffle=shuffle,
                    num_workers=num_workers
                )
                print(f'  [{env}/{split}] Dataset: {len(ds)} 샘플')
            except FileNotFoundError as e:
                warnings.warn(str(e))
                result[env][split] = None

    return result


# ─────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile

    fs = 16000
    seg = 64000
    n_fft = 512
    hop = 160
    n_frames = 1 + (seg - n_fft) // hop
    n_freq = n_fft // 2 + 1

    # 더미 npy 파일 생성
    with tempfile.TemporaryDirectory() as tmpdir:
        env = 'general'
        split = 'train'
        d = Path(tmpdir) / env / split
        d.mkdir(parents=True)

        rng = np.random.default_rng(0)
        for i in range(10):
            prefix = d / f'{i:06d}'
            np.save(f'{prefix}_bc.npy', rng.standard_normal(seg).astype(np.float32))
            np.save(f'{prefix}_ac.npy', rng.standard_normal(seg).astype(np.float32))
            np.save(f'{prefix}_clean.npy', rng.standard_normal(seg).astype(np.float32))
            zero_m = np.zeros((n_frames, n_freq), dtype=np.float32)
            np.save(f'{prefix}_mask_hard.npy', zero_m)
            np.save(f'{prefix}_mask_soft.npy', zero_m)
            np.save(f'{prefix}_mask_param.npy', zero_m)

        ds = SpeechEnhancementDataset(tmpdir, env, split, mask_type='soft')
        loader = build_dataloader(ds, batch_size=4, num_workers=0, pin_memory=False)

        batch = next(iter(loader))
        print(f'bc shape:    {batch["bc"].shape}')     # (4, 64000)
        print(f'ac shape:    {batch["ac"].shape}')     # (4, 64000)
        print(f'clean shape: {batch["clean"].shape}')  # (4, 64000)
        print(f'mask shape:  {batch["mask"].shape}')   # (4, T, F)
        print('[dataset] 단위 테스트 통과')
