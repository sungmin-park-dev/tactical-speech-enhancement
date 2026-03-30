"""
bwe.py
대역 확장(Bandwidth Extension) 서브모듈

실험 설계서 v3 §5.1:
  [BWE] FC(40→256) + FC(256→256) + Conv1D(256→64), ~300K 파라미터
  입력: BC 필터 특성 (40차원 LPC / 저주파 특징)
  출력: 64채널 고해상도 스펙트럼 임베딩 → Decoder에 합산
"""

import torch
import torch.nn as nn
from typing import Optional


class BWESubmodule(nn.Module):
    """
    대역 확장 서브모듈.

    입력 모드:
      'lpc'  : 40차원 LPC 계수 → 고주파 대역 추정 임베딩
      'spec' : 저주파 스펙트럼 bin (n_low_freq) → 전대역 스펙트럼 임베딩

    설계서 기준: FC(40→256) + FC(256→256) + Conv1D(256→64)
    출력: (B, 64, T) — Decoder 특징과 결합
    """

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 256,
        out_channels: int = 64,
        output_seq_len: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        input_dim      : 입력 특징 차원 (LPC: 40, spec: n_low_freq)
        hidden_dim     : 은닉층 차원 (256)
        out_channels   : 출력 채널 (64)
        output_seq_len : None이면 입력 시간 축 그대로 유지
        """
        super().__init__()
        self.input_dim = input_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.PReLU(),          # num_parameters=1: 스칼라 가중치 (Linear 출력 호환)
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
        )

        # Conv1D: (B, 256, T) → (B, 64, T)
        self.conv1d = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1),
            nn.PReLU(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_dim) or (B, input_dim, T)
            시간 × 특징 순서: (B, T, D)로 입력 권장

        Returns
        -------
        (B, out_channels, T)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, T, D) or (1, D, T)

        # 차원 정렬: (B, T, D) 필요
        if x.shape[-1] == self.input_dim:
            pass                             # (B, T, D)
        elif x.shape[1] == self.input_dim:
            x = x.transpose(1, 2)           # (B, D, T) → (B, T, D)

        h = self.fc_layers(x)               # (B, T, hidden)
        h = h.transpose(1, 2)               # (B, hidden, T)
        out = self.conv1d(h)                # (B, out_ch, T)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LPCExtractor(nn.Module):
    """
    BC 신호에서 LPC(Linear Predictive Coding) 계수 추출 → BWE 입력.

    실제 구현: 프레임별 자기상관 기반 LPC (Levinson-Durbin).
    여기서는 근사로 학습 가능한 프로젝션 사용 (빠른 훈련용).
    """

    def __init__(self, n_freq: int = 257, lpc_order: int = 40):
        super().__init__()
        self.lpc_order = lpc_order
        # 스펙트럼 → LPC 계수 근사 투영
        self.proj = nn.Linear(n_freq, lpc_order)

    def forward(self, magnitude_spec: torch.Tensor) -> torch.Tensor:
        """
        magnitude_spec: (B, F, T) or (B, T, F)
        반환: (B, T, lpc_order)
        """
        if magnitude_spec.shape[1] != magnitude_spec.shape[-1]:
            # (B, F, T) → (B, T, F)
            x = magnitude_spec.transpose(1, 2)
        else:
            x = magnitude_spec
        return self.proj(x)   # (B, T, lpc_order)


# ─────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────

if __name__ == '__main__':
    B, T, D = 4, 200, 40
    x = torch.randn(B, T, D)

    bwe = BWESubmodule(input_dim=D, hidden_dim=256, out_channels=64)
    out = bwe(x)
    params = bwe.count_parameters()

    print(f'입력: {x.shape} → 출력: {out.shape}')
    print(f'파라미터 수: {params:,}')  # ~300K 목표
    assert out.shape == (B, 64, T), f'출력 shape 오류: {out.shape}'
    print('BWESubmodule 단위 테스트 통과!')

    # LPCExtractor
    lpc_ext = LPCExtractor(n_freq=257, lpc_order=40)
    spec = torch.rand(B, 257, T)
    lpc = lpc_ext(spec)
    print(f'LPC 출력: {lpc.shape}')  # (B, T, 40)
