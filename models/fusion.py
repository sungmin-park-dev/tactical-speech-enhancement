"""
fusion.py
Magnitude 기반 어텐션 게이트 (Adaptive Fusion)

실험 설계서 v3 §5.1:
  입력: |BC_feat| (magnitude), |AC_feat|, soft_mask
  concat → Conv2D 1×1 (실수) → Sigmoid → weight w
  출력: w × BC_feat + (1-w) × AC_feat  (복소수 곱, 위상 보존)
  ★ 가중치 w는 실수 (0~1)
  ★ 복소수 특징맵에 스칼라 곱 → 위상 보존
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MagnitudeAttentionGate(nn.Module):
    """
    Magnitude 기반 실수 어텐션 가중치 산출 모듈.

    입력 채널 구성:
      - bc_feat:  복소수 특징맵 (B, C, T, F) — real/imag 스택 or 복소 텐서
      - ac_feat:  복소수 특징맵 (B, C, T, F)
      - sat_mask: 포화 마스크  (B, 1, T, F) — hard | soft | parametric 중 하나

    출력:
      - fused:    (B, C, T, F) — w × bc_feat + (1-w) × ac_feat
      - w:        (B, 1, T, F) — 어텐션 가중치 (for 분석)
    """

    def __init__(
        self,
        n_channels: int,
        use_mask: bool = True,
    ):
        """
        Parameters
        ----------
        n_channels : BC/AC 특징 채널 수 (C)
        use_mask   : False이면 마스크 없이 magnitude만 사용 (Ablation B용)
        """
        super().__init__()
        self.use_mask = use_mask

        # 입력: |BC|(C) + |AC|(C) + mask(1) → 가중치
        in_ch = n_channels * 2 + (1 if use_mask else 0)

        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_ch, n_channels // 2, kernel_size=1, bias=True),
            nn.PReLU(n_channels // 2),
            nn.Conv2d(n_channels // 2, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(
        self,
        bc_feat: torch.Tensor,
        ac_feat: torch.Tensor,
        sat_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        bc_feat  : (B, C, T, F)  복소수 or 실수 특징
        ac_feat  : (B, C, T, F)
        sat_mask : (B, 1, T, F) or (B, T, F) — None이면 0으로 대체

        Returns
        -------
        fused: (B, C, T, F)
        w    : (B, 1, T, F)  어텐션 가중치
        """
        # Magnitude 계산 (복소수일 경우 abs, 실수일 경우 abs)
        bc_mag = bc_feat.abs()   # (B, C, T, F)
        ac_mag = ac_feat.abs()

        # 마스크 준비
        if self.use_mask and sat_mask is not None:
            if sat_mask.dim() == 3:                  # (B, T, F)
                sat_mask = sat_mask.unsqueeze(1)     # (B, 1, T, F)
            gate_input = torch.cat([bc_mag, ac_mag, sat_mask], dim=1)
        else:
            gate_input = torch.cat([bc_mag, ac_mag], dim=1)

        w = self.gate_conv(gate_input)               # (B, 1, T, F)

        # 복소수 특징맵에 스칼라 가중치 적용 (위상 보존)
        fused = w * bc_feat + (1.0 - w) * ac_feat   # (B, C, T, F)

        return fused, w


class ConcatFusion(nn.Module):
    """
    Ablation B: 마스크 없이 단순 Concatenation 기반 융합.
    concat(BC, AC) → 1×1 Conv → C 채널로 축소.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(n_channels * 2, n_channels, kernel_size=1, bias=True),
            nn.PReLU(n_channels),
        )

    def forward(
        self,
        bc_feat: torch.Tensor,
        ac_feat: torch.Tensor,
        sat_mask: Optional[torch.Tensor] = None,  # 사용 안 함
    ) -> tuple[torch.Tensor, None]:
        fused = self.proj(torch.cat([bc_feat, ac_feat], dim=1))
        return fused, None


class AdaptiveFusion(nn.Module):
    """
    Ablation 변형을 통합하는 Fusion 모듈 팩토리.

    ablation_id:
      'A-soft'   | 'A-hard' | 'A-param' → MagnitudeAttentionGate (use_mask=True)
      'B'                               → ConcatFusion (use_mask=False)
      'C'                               → Identity (BC only, AC 무시)
    """

    def __init__(self, n_channels: int, ablation_id: str = 'A-soft'):
        super().__init__()
        self.ablation_id = ablation_id

        if ablation_id in ('A-soft', 'A-hard', 'A-param'):
            self.fusion = MagnitudeAttentionGate(n_channels, use_mask=True)
        elif ablation_id == 'B':
            self.fusion = ConcatFusion(n_channels)
        elif ablation_id == 'C':
            self.fusion = None  # BC only
        else:
            raise ValueError(f'Unknown ablation_id: {ablation_id}')

    def forward(
        self,
        bc_feat: torch.Tensor,
        ac_feat: Optional[torch.Tensor] = None,
        sat_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns: (fused_feat, attention_weight or None)
        """
        if self.ablation_id == 'C':
            # BC only: AC 무시
            return bc_feat, None

        assert ac_feat is not None, 'ac_feat required for ablation != C'
        return self.fusion(bc_feat, ac_feat, sat_mask)


# ─────────────────────────────────────────────
# 단위 테스트
# ─────────────────────────────────────────────

if __name__ == '__main__':
    B, C, T, F = 2, 64, 50, 257
    bc = torch.randn(B, C, T, F)
    ac = torch.randn(B, C, T, F)
    mask = torch.rand(B, 1, T, F)

    for ablation in ['A-soft', 'A-hard', 'A-param', 'B', 'C']:
        fusion = AdaptiveFusion(n_channels=C, ablation_id=ablation)
        fused, w = fusion(bc, ac, mask)
        w_str = f'{w.shape}' if w is not None else 'None'
        print(f'[{ablation:8s}] fused={fused.shape}, w={w_str}')
    print('AdaptiveFusion 모든 변형 OK')
