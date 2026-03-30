"""
loss.py
SI-SNR + Multi-Resolution STFT Loss

실험 설계서 v3 §6.1:
  L = L_SI-SNR + α × L_MR-STFT
  α = 0.5 (기본값, 탐색 대상: 0.3 / 0.5 / 1.0)

  MR-STFT 3해상도: (512,50,240) / (1024,120,600) / (2048,240,1200)
  각 해상도: Spectral Convergence + Log Magnitude L1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ─────────────────────────────────────────────
# SI-SNR Loss
# ─────────────────────────────────────────────

def si_snr(
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Scale-Invariant SNR.

    enhanced, clean: (B, T) or (B, 1, T)
    반환: (B,) SI-SNR 값 (높을수록 좋음)
    """
    if enhanced.dim() == 3:
        enhanced = enhanced.squeeze(1)
    if clean.dim() == 3:
        clean = clean.squeeze(1)

    # 평균 제거
    enhanced = enhanced - enhanced.mean(dim=-1, keepdim=True)
    clean = clean - clean.mean(dim=-1, keepdim=True)

    # Signal 투영
    dot = (enhanced * clean).sum(dim=-1, keepdim=True)
    s_target_power = (clean ** 2).sum(dim=-1, keepdim=True) + eps
    s_target = dot * clean / s_target_power           # 투영 성분

    e_noise = enhanced - s_target                      # 잔여 잡음

    # SI-SNR
    ratio = (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def si_snr_loss(enhanced: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """SI-SNR 손실 (최소화용, 부호 반전). 반환: 스칼라."""
    return -si_snr(enhanced, clean).mean()


# ─────────────────────────────────────────────
# STFT 유틸
# ─────────────────────────────────────────────

def _stft_magnitude(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    (B, T) → (B, F, T_frames) magnitude 스펙트로그램.
    """
    window = torch.hann_window(win_length, device=x.device)
    # torch.stft expects (B, T) or (T,)
    S = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
    )  # (B, F, T_frames) complex
    return S.abs()


# ─────────────────────────────────────────────
# MR-STFT Loss
# ─────────────────────────────────────────────

# 기본 3 해상도 (n_fft, hop_length, win_length)
DEFAULT_RESOLUTIONS: List[Tuple[int, int, int]] = [
    (512,  50,  240),
    (1024, 120, 600),
    (2048, 240, 1200),
]


def spectral_convergence_loss(
    enhanced_mag: torch.Tensor,
    clean_mag: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Spectral Convergence: ||S_clean - S_enh||_F / ||S_clean||_F"""
    diff = clean_mag - enhanced_mag
    sc = diff.norm(p='fro', dim=(-2, -1)) / (clean_mag.norm(p='fro', dim=(-2, -1)) + eps)
    return sc.mean()


def log_magnitude_loss(
    enhanced_mag: torch.Tensor,
    clean_mag: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log Magnitude L1: ||log(S_clean) - log(S_enh)||_1"""
    log_diff = torch.log(clean_mag + eps) - torch.log(enhanced_mag + eps)
    return log_diff.abs().mean()


def mr_stft_loss(
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    resolutions: List[Tuple[int, int, int]] = DEFAULT_RESOLUTIONS,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Multi-Resolution STFT Loss.

    enhanced, clean: (B, T)
    반환: 스칼라
    """
    if enhanced.dim() == 3:
        enhanced = enhanced.squeeze(1)
    if clean.dim() == 3:
        clean = clean.squeeze(1)

    total = torch.tensor(0.0, device=enhanced.device)

    for n_fft, hop, win in resolutions:
        enh_mag = _stft_magnitude(enhanced, n_fft, hop, win, eps)
        cln_mag = _stft_magnitude(clean,    n_fft, hop, win, eps)

        sc  = spectral_convergence_loss(enh_mag, cln_mag, eps)
        lm  = log_magnitude_loss(enh_mag, cln_mag, eps)
        total = total + sc + lm

    return total / len(resolutions)


# ─────────────────────────────────────────────
# 통합 손실
# ─────────────────────────────────────────────

class TotalLoss(nn.Module):
    """
    L = L_SI-SNR + alpha × L_MR-STFT

    alpha 탐색 대상: 0.3 / 0.5 / 1.0 (실험 §6.3)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        resolutions: List[Tuple[int, int, int]] = DEFAULT_RESOLUTIONS,
    ):
        super().__init__()
        self.alpha = alpha
        self.resolutions = resolutions

    def forward(
        self,
        enhanced: torch.Tensor,
        clean: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        enhanced : (B, T) or (B, 1, T)
        clean    : (B, T) or (B, 1, T)
        return_components: True면 (total, {'si_snr': ..., 'mr_stft': ...}) 반환

        Returns
        -------
        total loss (스칼라) or (total, components dict)
        """
        l_sisnr  = si_snr_loss(enhanced, clean)
        l_mrstft = mr_stft_loss(enhanced, clean, self.resolutions)
        total    = l_sisnr + self.alpha * l_mrstft

        if return_components:
            return total, {'si_snr': l_sisnr.item(), 'mr_stft': l_mrstft.item(), 'total': total.item()}
        return total


# ─────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import torch
    torch.manual_seed(0)

    B, T = 4, 64000
    clean    = torch.randn(B, T) * 0.1
    enhanced = clean + torch.randn(B, T) * 0.01   # 작은 잡음 추가

    # SI-SNR
    snr_vals = si_snr(enhanced, clean)
    print(f'SI-SNR: {snr_vals}')  # 높은 값 기대

    # MR-STFT
    mrstft = mr_stft_loss(enhanced, clean)
    print(f'MR-STFT Loss: {mrstft:.6f}')

    # 통합 손실
    loss_fn = TotalLoss(alpha=0.5)
    total, comps = loss_fn(enhanced, clean, return_components=True)
    print(f'Total Loss: {total:.6f}  |  components: {comps}')

    # 나쁜 예시 비교
    bad = torch.randn(B, T) * 0.1
    total_bad, comps_bad = loss_fn(bad, clean, return_components=True)
    print(f'Bad Total:  {total_bad:.6f}  |  components: {comps_bad}')
    assert total < total_bad, 'Loss should be lower for better enhancement'
    print('TotalLoss 단위 테스트 통과!')
