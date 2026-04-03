#!/usr/bin/env python3
"""
alpha_search.py
Fine-grained alpha sweep runner around a center value.

Example:
  python scripts/alpha_search.py --center 0.3 --radius 0.06 --step 0.02
  python scripts/alpha_search.py --alphas 0.24 0.27 0.30 0.33 0.36
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / 'train.py'


def _decimal(value: float | str) -> Decimal:
    return Decimal(str(value))


def _quantize(value: Decimal) -> Decimal:
    return value.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)


def build_alpha_grid(center: float, radius: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError('step must be > 0')
    if radius < 0:
        raise ValueError('radius must be >= 0')

    center_d = _decimal(center)
    radius_d = _decimal(radius)
    step_d = _decimal(step)

    start = center_d - radius_d
    stop = center_d + radius_d

    alphas: List[float] = []
    current = start
    while current <= stop + Decimal('1e-12'):
        alphas.append(float(_quantize(current)))
        current += step_d

    return sorted(set(alphas))


def alpha_slug(alpha: float) -> str:
    return f'alpha{alpha:.3f}'


def build_train_command(args: argparse.Namespace, alpha: float, save_dir: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(TRAIN_PY),
        '--config', args.config,
        '--data_config', args.data_config,
        '--ablation', args.ablation,
        '--env', args.env,
        '--alpha', str(alpha),
        '--seed', str(args.seed),
        '--epochs', str(args.epochs),
        '--save_dir', str(save_dir),
    ]

    if args.data_dir is not None:
        cmd.extend(['--data_dir', args.data_dir])
    if args.batch_size is not None:
        cmd.extend(['--batch_size', str(args.batch_size)])
    if args.lr is not None:
        cmd.extend(['--lr', str(args.lr)])

    return cmd


def load_run_summary(save_dir: Path) -> dict | None:
    summary_path = save_dir / 'metrics_summary.json'
    if not summary_path.exists():
        return None
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_table(rows: List[dict]) -> None:
    print()
    print(f'{"alpha":>8} | {"best_val_loss":>14} | {"best_val_sisnr":>15} | {"epoch@loss":>10} | {"epoch@sisnr":>11}')
    print('-' * 72)
    for row in rows:
        best_loss = row.get('best_val_loss')
        best_sisnr = row.get('best_val_sisnr')
        epoch_loss = row.get('best_epoch_by_loss')
        epoch_sisnr = row.get('best_epoch_by_sisnr')
        print(
            f'{row["alpha"]:>8.3f} | '
            f'{best_loss:>14.4f} | '
            f'{best_sisnr:>13.2f} dB | '
            f'{str(epoch_loss):>10} | '
            f'{str(epoch_sisnr):>11}'
        )


def save_aggregate(save_root: Path, rows: List[dict], selection_metric: str, best_row: dict) -> None:
    save_root.mkdir(parents=True, exist_ok=True)

    summary_path = save_root / 'alpha_search_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(
            {
                'selection_metric': selection_metric,
                'best_alpha': best_row['alpha'],
                'best_run': best_row,
                'runs': rows,
            },
            f,
            indent=2,
        )

    csv_path = save_root / 'alpha_search_summary.csv'
    fieldnames = [
        'alpha',
        'best_val_loss',
        'best_epoch_by_loss',
        'best_val_sisnr',
        'best_epoch_by_sisnr',
        'last_val_loss',
        'last_val_sisnr',
        'epochs_completed',
        'save_dir',
    ]
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fine alpha sweep runner around alpha=0.3')
    parser.add_argument('--config', default=str(REPO_ROOT / 'configs/train_config.yaml'))
    parser.add_argument('--data_config', default=str(REPO_ROOT / 'configs/data_config.yaml'))
    parser.add_argument('--data_dir', default=None)
    parser.add_argument('--ablation', default='A-soft')
    parser.add_argument('--env', default='military', choices=['military', 'general', 'mixed'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--save_root', default=str(REPO_ROOT / 'results/dpcrn/alpha_search_fine'))
    parser.add_argument('--selection_metric', choices=['best_val_sisnr', 'best_val_loss'], default='best_val_sisnr')
    parser.add_argument('--skip_existing', action='store_true', help='Reuse existing runs when metrics_summary.json is present')
    parser.add_argument('--dry_run', action='store_true', help='Print planned commands without executing train.py')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--alphas', nargs='+', type=float, default=None)
    group.add_argument('--center', type=float, default=0.3)

    parser.add_argument('--radius', type=float, default=0.06, help='Grid half-width when --alphas is not provided')
    parser.add_argument('--step', type=float, default=0.02, help='Grid step when --alphas is not provided')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    if args.alphas is not None:
        alphas = sorted(set(float(_quantize(_decimal(a))) for a in args.alphas))
    else:
        alphas = build_alpha_grid(args.center, args.radius, args.step)

    print('=== Alpha Sweep Setup ===', flush=True)
    print(f'alphas: {", ".join(f"{a:.3f}" for a in alphas)}', flush=True)
    print(f'selection metric: {args.selection_metric}', flush=True)
    print(f'save root: {save_root}', flush=True)

    rows: List[dict] = []
    for alpha in alphas:
        run_dir = save_root / alpha_slug(alpha)
        existing = load_run_summary(run_dir)

        if existing is not None and args.skip_existing:
            print(f'\n[skip] alpha={alpha:.3f} -> reusing {run_dir}', flush=True)
            row = dict(existing)
            row['alpha'] = float(alpha)
            rows.append(row)
            continue

        cmd = build_train_command(args, alpha, run_dir)
        print(f'\n=== alpha={alpha:.3f} ===', flush=True)
        print(' '.join(cmd), flush=True)

        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            raise SystemExit(f'train.py failed for alpha={alpha:.3f} with exit code {result.returncode}')

        summary = load_run_summary(run_dir)
        if summary is None:
            raise FileNotFoundError(f'metrics_summary.json not found after run: {run_dir}')

        row = dict(summary)
        row['alpha'] = float(alpha)
        rows.append(row)

    if args.dry_run:
        print('\nDry run only. No training was executed.', flush=True)
        return

    if not rows:
        raise SystemExit('No completed runs were collected.')

    rows.sort(key=lambda x: x['alpha'])
    if args.selection_metric == 'best_val_sisnr':
        best_row = max(rows, key=lambda x: x['best_val_sisnr'])
    else:
        best_row = min(rows, key=lambda x: x['best_val_loss'])

    print_table(rows)
    print()
    print(f'>>> best alpha by {args.selection_metric}: {best_row["alpha"]:.3f}', flush=True)

    save_aggregate(save_root, rows, args.selection_metric, best_row)
    print(f'Aggregate summary saved under: {save_root}', flush=True)


if __name__ == '__main__':
    main()
