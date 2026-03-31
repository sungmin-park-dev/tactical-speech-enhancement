"""
train.py
Training loop for DPCRN dual encoder speech enhancement.

Experiment Design v3 Section 6:
  Loss: SI-SNR + alpha * MR-STFT
  Optimizer: AdamW (lr=1e-3, wd=1e-2)
  Scheduler: CosineAnnealingWarmRestarts (T_0=10)
  Epochs: 50 (early stopping patience=10)
  Seeds: 3 per ablation

Usage:
  python train.py --config configs/train_config.yaml --data_config configs/data_config.yaml
  python train.py --ablation A-soft --seed 0 --epochs 50
"""

import os
import sys
import time
import yaml
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.dpcrn_dual import DPCRNDual
from models.loss import TotalLoss, si_snr
from data.dataset import SpeechEnhancementDataset


# =============================================
# Utilities
# =============================================

def set_seed(seed):
    """Reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _create_dummy_data(data_dir, env, n_train, n_val):
    """Create minimal dummy npy files for smoke testing."""
    seg = 64000
    n_fft, hop = 512, 160
    n_frames = 1 + (seg - n_fft) // hop
    n_freq = n_fft // 2 + 1
    rng = np.random.default_rng(0)

    for split, n in [('train', n_train), ('val', n_val)]:
        d = Path(data_dir) / env / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            prefix = d / f'{i:06d}'
            np.save(f'{prefix}_bc.npy', rng.standard_normal(seg).astype(np.float32) * 0.1)
            np.save(f'{prefix}_ac.npy', rng.standard_normal(seg).astype(np.float32) * 0.1)
            np.save(f'{prefix}_clean.npy', rng.standard_normal(seg).astype(np.float32) * 0.1)
            mask = rng.random((n_frames, n_freq)).astype(np.float32)
            np.save(f'{prefix}_mask_hard.npy', (mask > 0.5).astype(np.float32))
            np.save(f'{prefix}_mask_soft.npy', mask)
            np.save(f'{prefix}_mask_param.npy', mask)
    print(f'Dummy data created: {n_train} train + {n_val} val samples')


# =============================================
# Training loop
# =============================================

def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    ablation_id, mask_type, grad_clip, log_interval,
                    epoch, writer=None):
    """Single epoch training."""
    model.train()
    total_loss = 0.0
    total_sisnr = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        bc = batch['bc'].to(device)
        ac = batch['ac'].to(device)
        clean = batch['clean'].to(device)
        mask = batch.get('mask')

        # Mask selection based on ablation
        if ablation_id in ('A-soft', 'A-hard', 'A-param') and mask is not None:
            mask = mask.to(device)
        else:
            mask = None

        # AC input for ablation C
        ac_input = ac if ablation_id != 'C' else None

        # Forward
        enhanced = model(bc, ac_input, mask)

        # Loss
        loss = criterion(enhanced, clean)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Metrics
        with torch.no_grad():
            batch_sisnr = si_snr(enhanced, clean).mean().item()

        total_loss += loss.item()
        total_sisnr += batch_sisnr
        n_batches += 1

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / n_batches
            avg_sisnr = total_sisnr / n_batches
            print(f'  [{batch_idx+1}/{len(dataloader)}] '
                  f'loss={avg_loss:.4f} SI-SNR={avg_sisnr:.2f}dB')

        # TensorBoard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)

    avg_loss = total_loss / max(n_batches, 1)
    avg_sisnr = total_sisnr / max(n_batches, 1)
    return avg_loss, avg_sisnr


@torch.no_grad()
def validate(model, dataloader, criterion, device, ablation_id):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    total_sisnr = 0.0
    n_batches = 0

    for batch in dataloader:
        bc = batch['bc'].to(device)
        ac = batch['ac'].to(device)
        clean = batch['clean'].to(device)
        mask = batch.get('mask')

        if ablation_id in ('A-soft', 'A-hard', 'A-param') and mask is not None:
            mask = mask.to(device)
        else:
            mask = None

        ac_input = ac if ablation_id != 'C' else None
        enhanced = model(bc, ac_input, mask)

        loss = criterion(enhanced, clean)
        batch_sisnr = si_snr(enhanced, clean).mean().item()

        total_loss += loss.item()
        total_sisnr += batch_sisnr
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_sisnr = total_sisnr / max(n_batches, 1)
    return avg_loss, avg_sisnr


# =============================================
# Main
# =============================================

def main():
    parser = argparse.ArgumentParser(description='DPCRN Dual Encoder Training')
    parser.add_argument('--config', default='configs/train_config.yaml')
    parser.add_argument('--data_config', default='configs/data_config.yaml')
    parser.add_argument('--data_dir', default=None,
                        help='Processed data directory (overrides data_config)')

    # Override config values
    parser.add_argument('--backbone', default=None)
    parser.add_argument('--ablation', default=None)
    parser.add_argument('--mask_type', default=None)
    parser.add_argument('--env', default='military',
                        choices=['military', 'general', 'mixed'])
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Override with CLI args
    ablation_id = args.ablation or cfg['model']['ablation_id']
    mask_type = args.mask_type or cfg['model']['mask_type']
    alpha = args.alpha if args.alpha is not None else cfg['loss']['alpha']
    seed = args.seed if args.seed is not None else cfg['training']['seed']
    epochs = args.epochs or cfg['training']['epochs']
    batch_size = args.batch_size or cfg['training']['batch_size']
    lr = args.lr or cfg['optimizer']['lr']
    env = args.env

    set_seed(seed)

    # Save directory
    save_dir = args.save_dir or os.path.join(
        cfg['checkpoint']['save_dir'],
        f'{cfg["model"]["backbone"]}/{ablation_id}/seed{seed}'
    )
    os.makedirs(save_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n=== Training Config ===')
    print(f'Ablation: {ablation_id} | Mask: {mask_type} | Env: {env}')
    print(f'Alpha: {alpha} | Seed: {seed} | Epochs: {epochs}')
    print(f'Batch: {batch_size} | LR: {lr} | Device: {device}')
    print(f'Save: {save_dir}')

    # Data
    data_cfg = load_config(args.data_config)
    data_dir = args.data_dir or data_cfg['paths']['output_root']

    # Map mask_type for ablation
    mask_map = {'A-soft': 'soft', 'A-hard': 'hard', 'A-param': 'parametric'}
    actual_mask_type = mask_map.get(ablation_id, mask_type)

    # Build datasets
    train_data_dir = os.path.join(data_dir, env, 'train')

    if not os.path.isdir(train_data_dir):
        print(f'WARNING: {train_data_dir} not found. Creating dummy data for smoke test.')
        # Create minimal dummy data for smoke testing
        _create_dummy_data(data_dir, env, batch_size * 2, batch_size)

    train_dataset = SpeechEnhancementDataset(
        data_dir=data_dir, env=env, split='train', mask_type=actual_mask_type,
    )
    val_dataset = SpeechEnhancementDataset(
        data_dir=data_dir, env=env, split='val', mask_type=actual_mask_type,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )

    print(f'Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples')

    # Model
    model = DPCRNDual(
        n_fft=cfg['audio']['n_fft'],
        hop_length=cfg['audio']['hop_length'],
        win_length=cfg['audio']['win_length'],
        ablation_id=ablation_id,
        n_dual_path_blocks=cfg['model']['n_dual_path_blocks'],
        lstm_hidden=cfg['model']['lstm_hidden'],
        use_bwe=cfg['model'].get('use_bwe', False),
    ).to(device)

    n_params = count_parameters(model)
    print(f'Model parameters: {n_params:,} (~{n_params/1e6:.1f}M)')

    # Loss
    criterion = TotalLoss(alpha=alpha).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=cfg['optimizer']['weight_decay'],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg['scheduler']['T_0'],
    )

    # TensorBoard
    tb_dir = os.path.join(save_dir, 'tb')
    writer = SummaryWriter(tb_dir)

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f'Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}')

    # Training loop
    patience = cfg['training']['early_stopping_patience']
    grad_clip = cfg['training'].get('grad_clip_norm', 5.0)
    log_interval = cfg['logging']['log_interval']
    patience_counter = 0

    print(f'\n=== Training Start ===')
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Train
        train_loss, train_sisnr = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            ablation_id, mask_type, grad_clip, log_interval,
            epoch, writer,
        )

        # Validate
        val_loss, val_sisnr = validate(
            model, val_loader, criterion, device, ablation_id,
        )

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        dt = time.time() - t0
        print(f'Epoch {epoch+1:3d}/{epochs} ({dt:.0f}s) | '
              f'train_loss={train_loss:.4f} train_SI-SNR={train_sisnr:.2f}dB | '
              f'val_loss={val_loss:.4f} val_SI-SNR={val_sisnr:.2f}dB | '
              f'lr={current_lr:.2e}')

        # TensorBoard
        writer.add_scalar('train/epoch_loss', train_loss, epoch)
        writer.add_scalar('train/epoch_sisnr', train_sisnr, epoch)
        writer.add_scalar('val/epoch_loss', val_loss, epoch)
        writer.add_scalar('val/epoch_sisnr', val_sisnr, epoch)
        writer.add_scalar('lr', current_lr, epoch)

        # Checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_sisnr': val_sisnr,
            'best_val_loss': best_val_loss,
            'config': cfg,
            'ablation_id': ablation_id,
        }

        # Save last
        if cfg['checkpoint'].get('save_last', True):
            torch.save(ckpt, os.path.join(save_dir, 'last.pt'))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt['best_val_loss'] = best_val_loss
            if cfg['checkpoint'].get('save_best', True):
                torch.save(ckpt, os.path.join(save_dir, 'best.pt'))
            print(f'  >> New best val_loss: {best_val_loss:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  >> Early stopping at epoch {epoch+1} '
                      f'(patience={patience})')
                break

    writer.close()
    print(f'\n=== Training Complete ===')
    print(f'Best val_loss: {best_val_loss:.4f}')
    print(f'Checkpoints saved to: {save_dir}')


if __name__ == '__main__':
    main()
