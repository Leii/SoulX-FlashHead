"""
MLX custom operations for LTX VAE — NHWC (channels-last) internal format.

MLX uses channels-last for Conv3d: input (N, T, H, W, C), weight (C_out, kT, kH, kW, C_in).
This module keeps ALL internal tensors in NHWC format for zero-overhead Conv3d calls.
Only the external API (encode/decode) converts to/from PyTorch's NCDHW format.
"""

import math
from typing import Tuple
import mlx.core as mx
import mlx.nn as nn


# =============================================================================
# Format conversion (for external API boundary only)
# =============================================================================

def ncdhw_to_nhwc(x: mx.array) -> mx.array:
    """PyTorch (N,C,T,H,W) → MLX internal (N,T,H,W,C)."""
    return mx.transpose(x, (0, 2, 3, 4, 1))


def nhwc_to_ncdhw(x: mx.array) -> mx.array:
    """MLX internal (N,T,H,W,C) → PyTorch (N,C,T,H,W)."""
    return mx.transpose(x, (0, 4, 1, 2, 3))


# =============================================================================
# Normalization
# =============================================================================

class PixelNorm(nn.Module):
    """Channel-wise L2 norm. Channel is last dim in NHWC."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, T, H, W, C) — channel is dim=-1
        return x / mx.sqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps)


# =============================================================================
# CausalConv3d (NHWC native)
# =============================================================================

class CausalConv3d(nn.Module):
    """
    3D convolution with causal temporal padding, NHWC native.

    MLX Conv3d expects: input (N, T, H, W, C_in), weight (C_out, kT, kH, kW, C_in).
    No transpose overhead — everything stays in NHWC.
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
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self.kernel_size = kernel_size
        self.causal = causal
        self.spatial_padding_mode = spatial_padding_mode

    def _pad(self, x: mx.array) -> mx.array:
        """Pad NHWC tensor: (N, T, H, W, C)."""
        kT = kH = kW = self.kernel_size

        if self.causal:
            pad_t_l, pad_t_r = kT - 1, 0
        else:
            pad_t_l = pad_t_r = kT // 2

        pad_h = kH // 2
        pad_w = kW // 2

        if self.spatial_padding_mode == "replicate":
            if pad_t_l > 0:
                x = mx.concatenate([mx.repeat(x[:, :1], pad_t_l, axis=1), x], axis=1)
            if pad_t_r > 0:
                x = mx.concatenate([x, mx.repeat(x[:, -1:], pad_t_r, axis=1)], axis=1)
            if pad_h > 0:
                x = mx.concatenate([mx.repeat(x[:, :, :1], pad_h, axis=2), x, mx.repeat(x[:, :, -1:], pad_h, axis=2)], axis=2)
            if pad_w > 0:
                x = mx.concatenate([mx.repeat(x[:, :, :, :1], pad_w, axis=3), x, mx.repeat(x[:, :, :, -1:], pad_w, axis=3)], axis=3)
        else:
            x = mx.pad(x, [
                (0, 0),
                (pad_t_l, pad_t_r),
                (pad_h, pad_h),
                (pad_w, pad_w),
                (0, 0),
            ])
        return x

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, T, H, W, C) — NHWC native, no transpose needed!
        x = self._pad(x)
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


# =============================================================================
# ResNet Block
# =============================================================================

class ResnetBlock3D(nn.Module):
    """3D ResNet — NHWC native."""

    def __init__(self, in_channels: int, out_channels: int,
                 causal: bool = True, spatial_padding_mode: str = "replicate"):
        super().__init__()
        self.norm1 = PixelNorm()
        self.norm2 = PixelNorm()
        self.act = nn.SiLU()
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3,
                                   causal=causal, spatial_padding_mode=spatial_padding_mode)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3,
                                   causal=causal, spatial_padding_mode=spatial_padding_mode)
        self.use_shortcut_conv = in_channels != out_channels
        if self.use_shortcut_conv:
            self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1,
                                              causal=causal, spatial_padding_mode=spatial_padding_mode)
            self.norm3 = nn.LayerNorm(in_channels, eps=1e-6)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        if self.use_shortcut_conv:
            x = self.norm3(x)
            x = self.conv_shortcut(x)
        return x + h


# =============================================================================
# Upsample / Downsample
# =============================================================================

class Upsample3D(nn.Module):
    """Depth-to-space (pixel shuffle) 2x upsampling + Conv3d — NHWC native.

    The conv outputs 8x channels (2x per T,H,W dim), then depth-to-space
    rearranges them into spatial dimensions.
    """

    def __init__(self, channels: int, spatial_padding_mode: str = "replicate"):
        super().__init__()
        # Conv outputs 8x channels for 2x depth-to-space in T, H, W
        self.conv = CausalConv3d(channels, channels * 8, kernel_size=3, causal=False,
                                  spatial_padding_mode=spatial_padding_mode)
        self.channels = channels

    def __call__(self, x: mx.array) -> mx.array:
        """x: (N, T, H, W, C) NHWC → (N, T*2-1, H*2, W*2, C) NHWC"""
        x = self.conv(x)  # (N, T, H, W, C*8)

        N, T, H, W, C8 = x.shape
        C = C8 // 8
        # Depth-to-space in NHWC
        # (N, T, H, W, 2(p_T), 2(p_H), 2(p_W), C)
        x = mx.reshape(x, (N, T, H, W, 2, 2, 2, C))
        # Transpose: (N, T, p_T, H, p_H, W, p_W, C)
        x = mx.transpose(x, (0, 1, 4, 2, 5, 3, 6, 7))
        # (N, T*2, H*2, W*2, C)
        x = mx.reshape(x, (N, T * 2, H * 2, W * 2, C))
        # Remove duplicated first temporal frame (compensation for encoder causal pad)
        x = x[:, 1:, :, :, :]
        return x


class Downsample3D(nn.Module):
    """Strided CausalConv3d for 2x downsampling."""

    def __init__(self, channels: int, spatial_padding_mode: str = "replicate"):
        super().__init__()
        self.conv = CausalConv3d(channels, channels, kernel_size=3,
                                  stride=(2, 2, 2), causal=True,
                                  spatial_padding_mode=spatial_padding_mode)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


# =============================================================================
# Patchify / Unpatchify (NHWC)
# =============================================================================

def patchify_nhwc(x: mx.array, patch_size: int = 4) -> mx.array:
    """
    Fold spatial patches into channels. NHWC format.
    (N, T, H, W, C) → (N, T, H/p, W/p, C * p^2)
    Example: (1, 33, 512, 512, 3) → (1, 33, 128, 128, 48)  [3*4*4=48]
    """
    if patch_size == 1:
        return x
    N, T, H, W, C = x.shape
    p = patch_size
    # (N, T, H/p, p, W/p, p, C)
    x = mx.reshape(x, (N, T, H // p, p, W // p, p, C))
    # Transpose: bring p's together with C
    # (N, T, H/p, W/p, p, p, C)
    x = mx.transpose(x, (0, 1, 2, 4, 3, 5, 6))
    # (N, T, H/p, W/p, p * p * C)
    x = mx.reshape(x, (N, T, H // p, W // p, p * p * C))
    return x


def unpatchify_nhwc(x: mx.array, patch_size: int = 4,
                    out_channels: int = 3) -> mx.array:
    """
    Reverse of patchify_nhwc.
    (N, T, H_s, W_s, C*p^2) → (N, T, H_s*p, W_s*p, C)
    """
    if patch_size == 1:
        return x
    N, T, H_s, W_s, C_big = x.shape
    p = patch_size
    # Keep only the needed output channels
    needed_c = out_channels * p * p
    x = x[:, :, :, :, :needed_c]
    # (N, T, H_s, W_s, p, p, out_c)
    x = mx.reshape(x, (N, T, H_s, W_s, p, p, out_channels))
    # Transpose: (N, T, H_s, p, W_s, p, out_c)
    x = mx.transpose(x, (0, 1, 2, 4, 3, 5, 6))
    # (N, T, H, W, out_c)
    x = mx.reshape(x, (N, T, H_s * p, W_s * p, out_channels))
    return x


# =============================================================================
# Attention Block (NHWC)
# =============================================================================

class AttentionBlock(nn.Module):
    """Single-head spatial self-attention. NHWC native."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = PixelNorm()
        self.to_qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.channels = channels

    def __call__(self, x: mx.array) -> mx.array:
        """
        x: (N, T, H, W, C) NHWC
        Attention applied on H*W spatial dims, independently per frame.
        """
        N, T, H, W, C = x.shape
        identity = x

        # Reshape to (N*T, H, W, C) for 2D conv (MLX Conv2d also expects NHWC)
        x_2d = mx.reshape(x, (N * T, H, W, C))
        x_2d = self.norm(x_2d)

        # QKV projection (output: N*T, H, W, 3C)
        qkv = self.to_qkv(x_2d)
        qkv = mx.reshape(qkv, (N * T, H * W, 3, C))
        q = qkv[:, :, 0, :]  # (N*T, H*W, C)
        k = qkv[:, :, 1, :]
        v = qkv[:, :, 2, :]

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(C)
        scores = mx.matmul(q, mx.transpose(k, (0, 2, 1))) * scale
        attn = mx.softmax(scores, axis=-1)
        out = mx.matmul(attn, v)  # (N*T, H*W, C)

        # Reshape back
        out = mx.reshape(out, (N * T, H, W, C))
        out = self.proj(out)  # Conv2d expects NHWC
        out = mx.reshape(out, (N, T, H, W, C))

        return out + identity


# =============================================================================
# Weight conversion
# =============================================================================

def pt_to_mlx_conv3d(pt_weight: mx.array) -> mx.array:
    """
    PyTorch Conv3d (out, in, kT, kH, kW) → MLX (out, kT, kH, kW, in).
    """
    return mx.transpose(pt_weight, (0, 2, 3, 4, 1))
