"""
dpcrn_dual.py
DPCRN Dual Encoder - BC/AC fusion speech enhancement model

Experiment Design v3 Section 5.1:
  [BC Encoder] Complex Conv2D x 5 layers (16->32->64->128->256)
  [AC Encoder] Same structure (parameters not shared)
  [Adaptive Fusion] Magnitude-based attention gate
  [Dual-Path Module] BiLSTM(128) + FC, alternating x2
  [Decoder] ConvTranspose2D x 5 layers + Skip connections
  [Output] CRM -> iSTFT -> enhanced waveform
  [BWE] Optional bandwidth extension

Target parameters: ~2M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from models.fusion import AdaptiveFusion
from models.bwe import BWESubmodule, LPCExtractor


# =============================================
# Complex Conv2D blocks (real/imag channel stack)
# =============================================

class ComplexConv2d(nn.Module):
    """
    Complex Conv2D using real/imag channel stacking.

    Input:  (B, 2*C_in,  T, F)  -- [real; imag]
    Output: (B, 2*C_out, T, F') -- [real; imag]

    Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=(2, 3), stride=(1, 2), padding=(0, 1)):
        super().__init__()
        self.conv_rr = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_ri = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_ir = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_ii = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        C = x.shape[1] // 2
        x_real, x_imag = x[:, :C], x[:, C:]
        out_real = self.conv_rr(x_real) - self.conv_ii(x_imag)
        out_imag = self.conv_ri(x_real) + self.conv_ir(x_imag)
        return torch.cat([out_real, out_imag], dim=1)


class ComplexConvTranspose2d(nn.Module):
    """Transpose version of ComplexConv2d (for decoder)."""

    def __init__(self, in_channels, out_channels,
                 kernel_size=(2, 3), stride=(1, 2),
                 padding=(0, 1), output_padding=(0, 1)):
        super().__init__()
        self.tconv_rr = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.tconv_ri = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.tconv_ir = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.tconv_ii = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)

    def forward(self, x):
        C = x.shape[1] // 2
        x_real, x_imag = x[:, :C], x[:, C:]
        out_real = self.tconv_rr(x_real) - self.tconv_ii(x_imag)
        out_imag = self.tconv_ri(x_real) + self.tconv_ir(x_imag)
        return torch.cat([out_real, out_imag], dim=1)


# =============================================
# Encoder / Decoder blocks
# =============================================

class EncoderBlock(nn.Module):
    """ComplexConv2D + BatchNorm + PReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=(2, 3),
                 stride=(1, 2), padding=(0, 1)):
        super().__init__()
        self.conv = ComplexConv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(2 * out_ch)
        self.act = nn.PReLU(2 * out_ch)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """ComplexConvTranspose2D + BatchNorm + PReLU + Skip connection.

    in_ch:   decoder input channels (complex = 2*in_ch total)
    skip_ch: skip connection channels (complex = 2*skip_ch total)
    out_ch:  decoder output channels (complex = 2*out_ch total)
    """

    def __init__(self, in_ch, skip_ch, out_ch,
                 kernel_size=(2, 3), stride=(1, 2),
                 padding=(0, 1), output_padding=(0, 1)):
        super().__init__()
        # Input to tconv: concat of x (2*in_ch) and skip (2*skip_ch)
        # ComplexConvTranspose2d splits by 2 internally
        self.tconv = ComplexConvTranspose2d(
            in_ch + skip_ch, out_ch, kernel_size, stride, padding, output_padding
        )
        self.bn = nn.BatchNorm2d(2 * out_ch)
        self.act = nn.PReLU(2 * out_ch)

    def forward(self, x, skip):
        """
        x:    (B, 2*in_ch,  T, F)
        skip: (B, 2*skip_ch, T, F)
        """
        x = torch.cat([x, skip], dim=1)
        return self.act(self.bn(self.tconv(x)))


# =============================================
# Dual-Path Module
# =============================================

class DualPathBlock(nn.Module):
    """
    Time path (BiLSTM) + Frequency path (FC), alternating.

    Input/Output: (B, C, T, F)
    """

    def __init__(self, channels, hidden_size=128):
        super().__init__()
        # Time path: BiLSTM
        self.time_lstm = nn.LSTM(
            input_size=channels, hidden_size=hidden_size,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.time_fc = nn.Linear(hidden_size * 2, channels)
        self.time_norm = nn.LayerNorm(channels)

        # Frequency path: FC
        self.freq_fc1 = nn.Linear(channels, channels * 2)
        self.freq_act = nn.PReLU()
        self.freq_fc2 = nn.Linear(channels * 2, channels)
        self.freq_norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, T, F = x.shape

        # -- Time path --
        # (B*F, T, C)
        xt = x.permute(0, 3, 2, 1).reshape(B * F, T, C)
        xt_out, _ = self.time_lstm(xt)
        xt_out = self.time_fc(xt_out)
        xt_out = xt_out.reshape(B, F, T, C).permute(0, 3, 2, 1)
        x = self.time_norm((x + xt_out).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # -- Frequency path --
        # (B*T, F, C)
        xf = x.permute(0, 2, 3, 1).reshape(B * T, F, C)
        xf_out = self.freq_fc2(self.freq_act(self.freq_fc1(xf)))
        xf_out = xf_out.reshape(B, T, F, C).permute(0, 3, 1, 2)
        x = self.freq_norm((x + xf_out).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return x


# =============================================
# DPCRN Dual Encoder Model
# =============================================

class DPCRNDual(nn.Module):
    """
    DPCRN Dual Encoder model.

    Usage:
        model = DPCRNDual(ablation_id='A-soft')
        enhanced = model(bc_signal, ac_signal, sat_mask)

    Encoder channels: 1 -> 16 -> 32 -> 64 -> 128 -> 256  (complex: x2)
    Decoder channels: 256 -> 128 -> 64 -> 32 -> 16 -> 1   (symmetric)
    Skip: encoder block i's INPUT (ch[i]) -> decoder block (N-1-i) concat
    """

    ENCODER_CHANNELS = [1, 8, 16, 32, 64, 128]

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        ablation_id: str = 'A-soft',
        n_dual_path_blocks: int = 2,
        lstm_hidden: int = 64,
        use_bwe: bool = False,
        bwe_input_dim: int = 40,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.ablation_id = ablation_id
        self.n_freq = n_fft // 2 + 1

        self.register_buffer('window', torch.hann_window(win_length))

        ch = self.ENCODER_CHANNELS
        N = len(ch) - 1  # 5 encoder blocks

        # -- Dual Encoders (BC + AC, parameters NOT shared) --
        self.bc_encoder = nn.ModuleList([
            EncoderBlock(ch[i], ch[i + 1]) for i in range(N)
        ])
        self.ac_encoder = nn.ModuleList([
            EncoderBlock(ch[i], ch[i + 1]) for i in range(N)
        ])

        # -- Adaptive Fusion (from D2) --
        fusion_channels = 2 * ch[-1]  # 512 (real + imag stacked)
        self.fusion = AdaptiveFusion(n_channels=fusion_channels, ablation_id=ablation_id)

        # -- Dual-Path Module --
        self.dual_path_blocks = nn.ModuleList([
            DualPathBlock(channels=fusion_channels, hidden_size=lstm_hidden)
            for _ in range(n_dual_path_blocks)
        ])

        # -- Decoder (symmetric to encoder) --
        # decoder[0]: in=ch[5]=256, skip=ch[4]=128, out=ch[4]=128
        # decoder[1]: in=ch[4]=128, skip=ch[3]=64,  out=ch[3]=64
        # ...
        # decoder[4]: in=ch[1]=16,  skip=ch[0]=1,   out=ch[0]=1
        # skip_ch[i] = ch[N-1-i] = encoder block (N-1-i)'s input channels
        self.decoder = nn.ModuleList()
        for i in range(N):
            in_c = ch[N - i]        # decoder input: 256, 128, 64, 32, 16
            skip_c = ch[N - 1 - i]  # skip from encoder: 128, 64, 32, 16, 1
            out_c = ch[N - 1 - i]   # decoder output: 128, 64, 32, 16, 1
            self.decoder.append(DecoderBlock(in_c, skip_c, out_c))

        # -- CRM output (Complex Ratio Mask) --
        # decoder final output: 2*ch[0] = 2 channels (real + imag)
        self.crm_conv = nn.Conv2d(2 * ch[0], 2, kernel_size=1)

        # -- BWE (optional) --
        self.use_bwe = use_bwe
        if use_bwe:
            self.lpc_extractor = LPCExtractor(n_freq=self.n_freq, lpc_order=bwe_input_dim)
            self.bwe = BWESubmodule(input_dim=bwe_input_dim, hidden_dim=256, out_channels=64)

    def _stft(self, x):
        """(B, T) -> (B, 2, T_frames, F) -- real/imag stacked."""
        S = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            return_complex=True, center=True, pad_mode='reflect',
        )  # (B, F, T_frames)
        S = S.permute(0, 2, 1)  # (B, T_frames, F)
        return torch.cat([S.real.unsqueeze(1), S.imag.unsqueeze(1)], dim=1)

    def _istft(self, spec, length):
        """(B, 2, T_frames, F) -> (B, T)"""
        S = torch.complex(spec[:, 0], spec[:, 1]).permute(0, 2, 1)
        return torch.istft(
            S, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window, length=length,
        )

    def _encode(self, x, encoder):
        """Encoder forward. skips[i] = block i's INPUT (not output)."""
        skips = []
        for block in encoder:
            skips.append(x)
            x = block(x)
        return x, skips

    def _prepare_mask(self, sat_mask, target_shape):
        """Resize sat_mask to match encoder output shape."""
        if sat_mask is None:
            return None
        if sat_mask.dim() == 3:
            sat_mask = sat_mask.unsqueeze(1)
        _, _, T_t, F_t = target_shape
        return F.interpolate(sat_mask, size=(T_t, F_t), mode='bilinear', align_corners=False)

    def forward(self, bc, ac=None, mask=None):
        """
        bc   : (B, T) BC signal
        ac   : (B, T) AC noisy signal (None for ablation C)
        mask : (B, T_frames, F) saturation mask (None for ablation B/C)
        returns: (B, T) enhanced signal
        """
        orig_len = bc.shape[-1]

        # 1. STFT
        bc_spec = self._stft(bc)
        ac_spec = self._stft(ac) if ac is not None else None

        # 2. Encoder
        bc_feat, bc_skips = self._encode(bc_spec, self.bc_encoder)
        if ac_spec is not None:
            ac_feat, ac_skips = self._encode(ac_spec, self.ac_encoder)
        else:
            ac_feat = None
            ac_skips = [None] * len(bc_skips)

        # 3. Adaptive Fusion
        mask_resized = self._prepare_mask(mask, bc_feat.shape)
        fused, _ = self.fusion(bc_feat, ac_feat, mask_resized)

        # 4. Dual-Path Module
        x = fused
        for dpb in self.dual_path_blocks:
            x = dpb(x)

        # 5. Decoder with skip connections
        # decoder[i] uses skip from bc_skips[N-1-i]
        # bc_skips[j] = encoder block j's input, channels = 2*ch[j]
        N = len(self.decoder)
        for i, dec_block in enumerate(self.decoder):
            skip_idx = N - 1 - i
            bc_skip = bc_skips[skip_idx]
            if ac_skips[skip_idx] is not None:
                skip = (bc_skip + ac_skips[skip_idx]) / 2.0
            else:
                skip = bc_skip

            # Align spatial dimensions (may differ by 1-2 due to conv padding)
            if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
                mt = min(x.shape[2], skip.shape[2])
                mf = min(x.shape[3], skip.shape[3])
                x = x[:, :, :mt, :mf]
                skip = skip[:, :, :mt, :mf]

            x = dec_block(x, skip)

        # 6. CRM (Complex Ratio Mask)
        if x.shape[2] != bc_spec.shape[2] or x.shape[3] != bc_spec.shape[3]:
            x = F.interpolate(x, size=(bc_spec.shape[2], bc_spec.shape[3]),
                              mode='bilinear', align_corners=False)

        crm = torch.tanh(self.crm_conv(x))  # (B, 2, T_f, F)

        # Apply CRM: enhanced = CRM * input_spec (complex multiplication)
        input_spec = ac_spec if ac_spec is not None else bc_spec
        crm_r, crm_i = crm[:, 0:1], crm[:, 1:2]
        spec_r, spec_i = input_spec[:, 0:1], input_spec[:, 1:2]
        enh_r = crm_r * spec_r - crm_i * spec_i
        enh_i = crm_r * spec_i + crm_i * spec_r
        enhanced_spec = torch.cat([enh_r, enh_i], dim=1)

        # 7. iSTFT
        return self._istft(enhanced_spec, orig_len)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================
# Unit test
# =============================================

if __name__ == '__main__':
    torch.manual_seed(42)

    B, T = 2, 64000
    bc = torch.randn(B, T) * 0.1
    ac = torch.randn(B, T) * 0.1

    n_fft, hop = 512, 160
    n_frames = 1 + (T - n_fft) // hop
    n_freq = n_fft // 2 + 1
    mask = torch.rand(B, n_frames, n_freq)

    print(f'Input: bc={bc.shape}, ac={ac.shape}, mask={mask.shape}')
    print(f'STFT: {n_frames} frames x {n_freq} freq bins')
    print()

    for ablation in ['A-soft', 'A-hard', 'A-param', 'B', 'C']:
        model = DPCRNDual(ablation_id=ablation)
        params = model.count_parameters()

        if ablation == 'C':
            out = model(bc, ac=None, mask=None)
        elif ablation == 'B':
            out = model(bc, ac, mask=None)
        else:
            out = model(bc, ac, mask)

        print(f'[{ablation:8s}] output={out.shape}, params={params:,}')

    print(f'\nAll DPCRNDual ablation variants passed forward pass')
    print(f'Parameter count: {params:,} (~{params/1e6:.1f}M)')
