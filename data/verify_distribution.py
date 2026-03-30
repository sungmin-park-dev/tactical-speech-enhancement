"""
verify_distribution.py
데이터셋 분포 검증 스크립트

사용법:
  python data/verify_distribution.py --data_dir data/processed --env military
  python data/verify_distribution.py --data_dir data/processed --env both --save_dir results/data_stats
"""

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_metadata(data_dir: str, env: str, split: str) -> Optional[dict]:
    """metadata.npz 로드."""
    meta_path = Path(data_dir) / env / split / 'metadata.npz'
    if not meta_path.exists():
        print(f'  메타데이터 없음: {meta_path}')
        return None
    data = np.load(meta_path)
    return {k: data[k] for k in data.files}


def print_stats(name: str, arr: np.ndarray):
    """1차원 배열 통계 출력."""
    print(f'  {name:20s}: mean={arr.mean():.3f}, std={arr.std():.3f}, '
          f'min={arr.min():.3f}, max={arr.max():.3f}, n={len(arr)}')


def plot_distributions(
    metas: dict,
    save_path: Optional[str] = None,
    title: str = '데이터셋 분포 검증',
):
    """
    metas: {'{env}/{split}': metadata_dict} 형식
    """
    keys = list(metas.keys())
    n = len(keys)
    if n == 0:
        print('시각화할 데이터 없음')
        return

    metrics = ['snr_db', 'clip_ratio', 'impulse_ratio', 'sat_applied']
    metric_labels = {
        'snr_db':        'SNR (dB)',
        'clip_ratio':    '클리핑 비율',
        'impulse_ratio': '충격음 밀도',
        'sat_applied':   '포화 적용 비율',
    }

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n, n_metrics, figsize=(5 * n_metrics, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, (label, meta) in enumerate(metas.items()):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            if metric not in meta:
                ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center')
                continue

            arr = meta[metric]
            if metric == 'sat_applied':
                sat_rate = arr.mean() * 100
                ax.bar(['미적용', '적용'], [(1 - arr.mean()) * 100, arr.mean() * 100],
                       color=['#4A90D9', '#E8604C'])
                ax.set_ylabel('비율 (%)')
                ax.set_title(f'{label}\n{metric_labels[metric]} = {sat_rate:.1f}%')
            else:
                ax.hist(arr, bins=30, color='#4A90D9', edgecolor='white', alpha=0.85)
                ax.axvline(arr.mean(), color='#E8604C', linewidth=2, label=f'평균={arr.mean():.2f}')
                ax.set_xlabel(metric_labels[metric])
                ax.set_ylabel('빈도')
                ax.set_title(f'{label}\n{metric_labels[metric]}')
                ax.legend(fontsize=8)

            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'그래프 저장: {save_path}')
    else:
        plt.show()
    plt.close()


def plot_mask_statistics(
    data_dir: str,
    env: str,
    split: str,
    n_samples: int = 50,
    save_path: Optional[str] = None,
):
    """마스크 유형별 점유율 비교."""
    base = Path(data_dir) / env / split
    if not base.exists():
        print(f'디렉토리 없음: {base}')
        return

    hard_means, soft_means, param_means = [], [], []

    files = sorted(base.glob('*_mask_hard.npy'))[:n_samples]
    if not files:
        print('마스크 파일 없음')
        return

    for f in files:
        i_str = f.stem.split('_')[0]
        try:
            h = np.load(base / f'{i_str}_mask_hard.npy').mean()
            s = np.load(base / f'{i_str}_mask_soft.npy').mean()
            p = np.load(base / f'{i_str}_mask_param.npy').mean()
            hard_means.append(h)
            soft_means.append(s)
            param_means.append(p)
        except Exception:
            continue

    if not hard_means:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, name, color in zip(
        axes,
        [hard_means, soft_means, param_means],
        ['Hard Mask', 'Soft Mask', 'Parametric Mask'],
        ['#E8604C', '#4A90D9', '#7ED321'],
    ):
        ax.hist(data, bins=20, color=color, edgecolor='white', alpha=0.85)
        ax.set_title(f'{name}\n평균={np.mean(data):.4f}', fontsize=11)
        ax.set_xlabel('마스크 평균값')
        ax.set_ylabel('빈도')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle(f'마스크 통계 [{env}/{split}]', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'마스크 분포 저장: {save_path}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='데이터셋 분포 검증')
    parser.add_argument('--data_dir', default='data/processed')
    parser.add_argument('--env', choices=['military', 'general', 'both'], default='both')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--save_dir', default='results/data_stats')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    envs = ['military', 'general'] if args.env == 'both' else [args.env]

    # 분포 로드 + 출력
    all_metas = {}
    for env in envs:
        for split in args.splits:
            meta = load_metadata(args.data_dir, env, split)
            key = f'{env}/{split}'
            if meta is not None:
                all_metas[key] = meta
                print(f'\n=== {key} ===')
                for k in ['snr_db', 'clip_ratio', 'impulse_ratio', 'sat_applied']:
                    if k in meta:
                        print_stats(k, meta[k])

    # 전체 분포 시각화
    if all_metas:
        plot_distributions(
            all_metas,
            save_path=str(save_dir / 'dataset_distributions.png'),
            title='합성 데이터셋 분포 검증',
        )

    # 마스크 통계 시각화
    for env in envs:
        for split in args.splits:
            plot_mask_statistics(
                data_dir=args.data_dir,
                env=env,
                split=split,
                save_path=str(save_dir / f'mask_stats_{env}_{split}.png'),
            )

    print(f'\n=== 분포 검증 완료. 저장 위치: {save_dir} ===')


if __name__ == '__main__':
    main()
