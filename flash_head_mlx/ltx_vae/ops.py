"""
MLX custom operations for LTX VAE — NHWC (channels-last) internal format.

Matches PyTorch diffusers AutoencoderKLLTXVideo / CausalVideoAutoencoder exactly.
"""

import math
from typing import Tuple
import mlx.core as mx
import mlx.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# Format conversion
# ═══════════════════════════════════════════════════════════════════════════════

def ncdhw_to_nhwc(x: mx.array) -> mx.array:
    """PyTorch (N,C,T,H,W) → MLX (N,T,H,W,C)."""
    return mx.transpose(x, (0, 2, 3, 4, 1))


def nhwc_to_ncdhw(x: mx.array) -> mx.array:
    """MLX (N,T,H,W,C) → PyTorch (N,C,T,H,W)."""
    return mx.transpose(x, (0, 4, 1, 2, 3))


# ═══════════════════════════════════════════════════════════════════════════════
# PixelNorm
# ═══════════════════════════════════════════════════════════════════════════════

class PixelNorm(nn.Module):
    """Channel-wise L2 norm (NHWC, dim=-1). No learnable parameters."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return x / mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps)


# ═══════════════════════════════════════════════════════════════════════════════
# CausalConv3d (NHWC native) — matches PT CausalConv3d exactly
# ═══════════════════════════════════════════════════════════════════════════════

class CausalConv3d(nn.Module):
    """
    PT CausalConv3d forward:
      if causal:  replicate first frame (kT-1) times, prepend
      else:       replicate first/last frame (kT-1)//2 times, prepend/append
      spatial:    Conv3d handles padding=(0, h_pad, w_pad) with padding_mode

    MLX: we pad everything manually, then call Conv3d with padding=0.
    Temporal padding is ALWAYS replicate (matching PT). Spatial padding
    is zeros or replicate per spatial_padding_mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0)
        self.kernel_size = kernel_size
        self.causal = causal
        self.spatial_padding_mode = spatial_padding_mode

    def _pad(self, x: mx.array) -> mx.array:
        """Pad NHWC tensor: (N, T, H, W, C)."""
        kT = kH = kW = self.kernel_size

        # Temporal padding: ALWAYS replicate (matching PT CausalConv3d)
        if self.causal:
            pad_t_l, pad_t_r = kT - 1, 0
        else:
            pad_t_l = pad_t_r = kT // 2

        if pad_t_l > 0:
            x = mx.concatenate([mx.repeat(x[:, :1], pad_t_l, axis=1), x], axis=1)
        if pad_t_r > 0:
            x = mx.concatenate([x, mx.repeat(x[:, -1:], pad_t_r, axis=1)], axis=1)

        # Spatial padding
        pad_h = kH // 2
        pad_w = kW // 2

        if self.spatial_padding_mode == "replicate":
            if pad_h > 0:
                x = mx.concatenate([
                    mx.repeat(x[:, :, :1], pad_h, axis=2),
                    x,
                    mx.repeat(x[:, :, -1:], pad_h, axis=2),
                ], axis=2)
            if pad_w > 0:
                x = mx.concatenate([
                    mx.repeat(x[:, :, :, :1], pad_w, axis=3),
                    x,
                    mx.repeat(x[:, :, :, -1:], pad_w, axis=3),
                ], axis=3)
        else:
            # Zero padding
            x = mx.pad(x, [
                (0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0),
            ])
        return x

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(self._pad(x))

    @property
    def weight(self):
        return self.conv.weight

    @weight.setter
    def weight(self, value):
        self.conv.weight = value

    @property
    def bias(self):
        return self.conv.bias

    @bias.setter
    def bias(self, value):
        self.conv.bias = value


# ═══════════════════════════════════════════════════════════════════════════════
# Plain Conv3d 1×1×1 (for conv_shortcut in ResnetBlock3D)
# ═══════════════════════════════════════════════════════════════════════════════

class PlainConv3d(nn.Module):
    """1×1×1 Conv3d — no padding, no causal. Used for resnet shortcuts (make_linear_nd)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)

    @property
    def weight(self):
        return self.conv.weight

    @weight.setter
    def weight(self, value):
        self.conv.weight = value

    @property
    def bias(self):
        return self.conv.bias

    @bias.setter
    def bias(self, value):
        self.conv.bias = value


# ═══════════════════════════════════════════════════════════════════════════════
# ResnetBlock3D (matched to PT)
# ═══════════════════════════════════════════════════════════════════════════════

class ResnetBlock3D(nn.Module):
    """
    PT ResnetBlock3D forward:
      h = norm1(x) → SiLU → conv1(h, causal)
      h = norm2(h) → SiLU → dropout → conv2(h, causal)
      shortcut = norm3(x) → conv_shortcut(x)   [only if in_ch != out_ch]
      out = shortcut + h
    """

    def __init__(self, in_channels: int, out_channels: int,
                 causal: bool = True, spatial_padding_mode: str = "zeros"):
        super().__init__()
        self.norm1 = PixelNorm()
        self.norm2 = PixelNorm()
        self.act = nn.SiLU()
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3,
                                   causal=causal, spatial_padding_mode=spatial_padding_mode)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3,
                                   causal=causal, spatial_padding_mode=spatial_padding_mode)
        self.use_shortcut = in_channels != out_channels
        if self.use_shortcut:
            # PT uses LayerNorm + make_linear_nd (1×1×1 Conv3d, no causal)
            self.norm3 = nn.LayerNorm(in_channels, eps=1e-6)
            self.conv_shortcut = PlainConv3d(in_channels, out_channels)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        if self.use_shortcut:
            x = self.norm3(x)
            x = self.conv_shortcut(x)
        return x + h


# ═══════════════════════════════════════════════════════════════════════════════
# DepthToSpaceUpsample (matches PT DepthToSpaceUpsample without residual)
# ═══════════════════════════════════════════════════════════════════════════════

class DepthToSpaceUpsample(nn.Module):
    """
    PT DepthToSpaceUpsample (non-residual mode):
      conv: CausalConv3d(in, in*8, 3, stride=1)
      depth-to-space: (B, T*2, H*2, W*2, C)
      if stride_t==2: remove first temporal frame
    """

    def __init__(self, channels: int, spatial_padding_mode: str = "zeros",
                 causal: bool = True):
        super().__init__()
        self.conv = CausalConv3d(channels, channels * 8, kernel_size=3,
                                  causal=causal, spatial_padding_mode=spatial_padding_mode)
        self.channels = channels

    def __call__(self, x: mx.array) -> mx.array:
        """x: (N, T, H, W, C) NHWC → (N, T*2-1, H*2, W*2, C) NHWC

        PT: rearrange(x, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)")
        The 4096 channels are grouped as (C=512, pT=2, pH=2, pW=2) where
        C varies slowest (index // 8), pW varies fastest (index % 2).
        In NHWC reshape terms: (N, T, H, W, C, pT, pH, pW) — C before the
        pixel dimensions so C varies slowest in the flattened axis.
        """
        N, T, H, W, C = x.shape
        x = self.conv(x)  # (N, T, H, W, channels*8)

        # conv output channels: (C, pT, pH, pW) — C slowest, pW fastest
        x = mx.reshape(x, (N, T, H, W, C, 2, 2, 2))  # (N,T,H,W, C,pT,pH,pW)
        # → (N, T, pT, H, pH, W, pW, C)
        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
        # → (N, T*2, H*2, W*2, C)  — (T,pT) flattened: T slower, pT faster → t*2+pt ✓
        x = mx.reshape(x, (N, T * 2, H * 2, W * 2, C))

        # Remove first temporal frame (stride_t==2 compensation)
        x = x[:, 1:, :, :, :]
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Strided downsampler (compress_all)
# ═══════════════════════════════════════════════════════════════════════════════

class Downsample3D(nn.Module):
    """PT compress_all: CausalConv3d with stride=(2,2,2)."""
    def __init__(self, channels: int, spatial_padding_mode: str = "zeros"):
        super().__init__()
        self.conv = CausalConv3d(channels, channels, kernel_size=3,
                                  stride=(2, 2, 2), causal=True,
                                  spatial_padding_mode=spatial_padding_mode)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Patchify / Unpatchify (NHWC, spatial-only)
# ═══════════════════════════════════════════════════════════════════════════════

def patchify_nhwc(x: mx.array, patch_size: int = 4) -> mx.array:
    """
    PT patchify(patch_size_hw=4, patch_size_t=1):
      PT channel order: new_c = c*p^2 + (w%p)*p + (h%p)
      i.e. C is slowest-varying, W_patch is middle, H_patch is fastest.
      (N, T, H, W, C) → (N, T, H/p, W/p, C * p^2)

    We flatten (C, pW, pH) in C-slowest order to match PT.
    """
    if patch_size == 1:
        return x
    N, T, H, W, C = x.shape
    p = patch_size
    # (N, T, H/p, pH, W/p, pW, C) → (N, T, H/p, W/p, C, pW, pH) → flatten last 3
    x = mx.reshape(x, (N, T, H // p, p, W // p, p, C))
    x = mx.transpose(x, (0, 1, 2, 4, 6, 5, 3))  # → (N, T, H/p, W/p, C, pW, pH)
    x = mx.reshape(x, (N, T, H // p, W // p, p * p * C))
    return x


def unpatchify_nhwc(x: mx.array, patch_size: int = 4,
                    out_channels: int = 3) -> mx.array:
    """
    PT unpatchify(patch_size_hw=4, patch_size_t=1):
      Reverse of patchify_nhwc.
      (N, T, H_s, W_s, C*p^2) → (N, T, H, W, C)
    """
    if patch_size == 1:
        return x
    N, T, H_s, W_s = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    p = patch_size
    needed_c = out_channels * p * p
    x = x[:, :, :, :, :needed_c]
    # (N, T, H_s, W_s, C, pW, pH) → (N, T, H_s, pH, W_s, pW, C) → (N, T, H, W, C)
    x = mx.reshape(x, (N, T, H_s, W_s, out_channels, p, p))
    x = mx.transpose(x, (0, 1, 2, 6, 3, 5, 4))  # (N, T, H_s, pH, W_s, pW, C)
    x = mx.reshape(x, (N, T, H_s * p, W_s * p, out_channels))
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# Weight conversion: PT NCDHW → MLX NHWC
# ═══════════════════════════════════════════════════════════════════════════════

def pt_to_mlx_conv3d(pt_weight: mx.array) -> mx.array:
    """PyTorch Conv3d (out, in, kT, kH, kW) → MLX (out, kT, kH, kW, in)."""
    return mx.transpose(pt_weight, (0, 2, 3, 4, 1))
