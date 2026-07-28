"""
MLX CausalVideoAutoencoder — NHWC internal format for native MLX Conv3d performance.

Channel flow (matching diffusers checkpoint):
  Encoder: patchify(4x4) → 48ch → 128 → 256 → 512 → 512 → 128(latent)
  Decoder: 128(latent) → 512 → 512 → 256 → 128 → 48 → unpatchify → 3(RGB)

External API uses NCDHW (PyTorch-compatible), internal uses NHWC for zero-overhead Conv3d.
"""

from typing import List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn

from flash_head_mlx.ltx_vae.ops import (
    CausalConv3d, PixelNorm, ResnetBlock3D, AttentionBlock,
    Upsample3D, Downsample3D,
    patchify_nhwc, unpatchify_nhwc,
    ncdhw_to_nhwc, nhwc_to_ncdhw, pt_to_mlx_conv3d,
)


# =============================================================================
# Building Blocks (NHWC internal)
# =============================================================================

class DownBlock3D(nn.Module):
    """
    Diffusers down block: ResNets → downsampler → conv_out (channel change).

    IMPORTANT: The channel-doubling conv_out happens AFTER the downsampler,
    not before. This matches the diffusers safetensors layout where
    downsampler weights have in_ch == out_ch (no channel change).
    """

    def __init__(self, in_ch: int, out_ch: int, num_resnets: int,
                 has_downsample: bool = True, causal: bool = True,
                 padding_mode: str = "replicate"):
        super().__init__()
        self.resnets = [
            ResnetBlock3D(in_ch, in_ch, causal=causal, spatial_padding_mode=padding_mode)
            for _ in range(num_resnets)
        ]
        # Downsampler operates on in_ch (no channel change)
        self.downsampler = Downsample3D(in_ch, spatial_padding_mode=padding_mode) if has_downsample else None
        # Channel change happens after downsampling
        self.conv_out = (
            ResnetBlock3D(in_ch, out_ch, causal=causal, spatial_padding_mode=padding_mode)
            if out_ch != in_ch else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        if self.conv_out is not None:
            x = self.conv_out(x)
        return x


class UpBlock3D(nn.Module):
    """Optional channel-halving conv + ResNets + optional nearest-upsample."""

    def __init__(self, in_ch: int, out_ch: int, num_resnets: int,
                 has_upsample: bool = True, causal: bool = False,
                 padding_mode: str = "replicate"):
        super().__init__()
        self.conv_in = (
            ResnetBlock3D(in_ch, out_ch, causal=causal, spatial_padding_mode=padding_mode)
            if in_ch != out_ch else None
        )
        res_in = out_ch if in_ch != out_ch else in_ch
        self.resnets = [
            ResnetBlock3D(res_in, res_in, causal=causal, spatial_padding_mode=padding_mode)
            for _ in range(num_resnets)
        ]
        self.upsampler = Upsample3D(res_in, spatial_padding_mode=padding_mode) if has_upsample else None

    def __call__(self, x: mx.array) -> mx.array:
        if self.conv_in is not None:
            x = self.conv_in(x)
        for r in self.resnets:
            x = r(x)
        if self.upsampler is not None:
            x = self.upsampler(x)
        return x


class MidBlock3D(nn.Module):
    """ResNets only, no spatial change."""

    def __init__(self, channels: int, num_resnets: int, causal: bool = True,
                 padding_mode: str = "replicate"):
        super().__init__()
        self.resnets = [
            ResnetBlock3D(channels, channels, causal=causal, spatial_padding_mode=padding_mode)
            for _ in range(num_resnets)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for r in self.resnets:
            x = r(x)
        return x


# =============================================================================
# Encoder (NHWC internal)
# =============================================================================

class Encoder(nn.Module):
    def __init__(self, padding_mode: str = "replicate"):
        super().__init__()
        # Input after patchify: 3*4*4=48 channels → 128
        self.conv_in = CausalConv3d(48, 128, kernel_size=3, causal=True,
                                     spatial_padding_mode=padding_mode)
        self.down_blocks = [
            DownBlock3D(128, 256, num_resnets=4, has_downsample=True, causal=True, padding_mode=padding_mode),
            DownBlock3D(256, 512, num_resnets=3, has_downsample=True, causal=True, padding_mode=padding_mode),
            DownBlock3D(512, 512, num_resnets=3, has_downsample=True, causal=True, padding_mode=padding_mode),
            DownBlock3D(512, 512, num_resnets=3, has_downsample=False, causal=True, padding_mode=padding_mode),
        ]
        self.mid_block = MidBlock3D(512, num_resnets=4, causal=True, padding_mode=padding_mode)
        self.norm_out = PixelNorm()
        self.act_out = nn.SiLU()
        # 128 latent + 1 log-var = 129 output channels
        self.conv_out = CausalConv3d(512, 129, kernel_size=3, causal=True,
                                      spatial_padding_mode=padding_mode)

    def __call__(self, x_ncdhw: mx.array) -> mx.array:
        """
        x_ncdhw: (N, C, T, H, W) PyTorch format
        Returns: (N, 128, T', H', W') NCDHW
        """
        # Convert to NHWC and patchify
        x = ncdhw_to_nhwc(x_ncdhw)               # (N, T, H, W, C)
        x = patchify_nhwc(x, patch_size=4)        # (N, T, H/4, W/4, C*16)

        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)

        # Keep only latent channels (drop log-var), convert back to NCDHW
        x = x[:, :, :, :, :128]                    # (N, T, H, W, 128)
        return nhwc_to_ncdhw(x)                    # (N, 128, T, H, W)


# =============================================================================
# Decoder (NHWC internal)
# =============================================================================

class Decoder(nn.Module):
    def __init__(self, padding_mode: str = "replicate"):
        super().__init__()
        self.conv_in = CausalConv3d(128, 512, kernel_size=3, causal=False,
                                     spatial_padding_mode=padding_mode)
        self.mid_block = MidBlock3D(512, num_resnets=4, causal=False, padding_mode=padding_mode)
        self.up_blocks = [
            UpBlock3D(512, 512, num_resnets=3, has_upsample=False, causal=False, padding_mode=padding_mode),
            UpBlock3D(512, 512, num_resnets=3, has_upsample=True,  causal=False, padding_mode=padding_mode),
            UpBlock3D(512, 256, num_resnets=3, has_upsample=True,  causal=False, padding_mode=padding_mode),
            UpBlock3D(256, 128, num_resnets=4, has_upsample=True,  causal=False, padding_mode=padding_mode),
        ]
        self.norm_out = PixelNorm()
        self.act_out = nn.SiLU()
        # 3*4*4=48 output channels → unpatchify → 3 RGB
        self.conv_out = CausalConv3d(128, 48, kernel_size=3, causal=False,
                                      spatial_padding_mode=padding_mode)

    def __call__(self, x_ncdhw: mx.array, target_shape: Tuple[int, ...]) -> mx.array:
        """
        x_ncdhw: (N, C, T', H', W') PyTorch format
        Returns: (N, C_out, T, H, W) NCDHW, cropped to target_shape
        """
        x = ncdhw_to_nhwc(x_ncdhw)                # (N, T, H, W, C)
        x = self.conv_in(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)

        x = unpatchify_nhwc(x, patch_size=4, out_channels=3)  # (N, T', H', W', 3)
        x = nhwc_to_ncdhw(x)                                   # (N, 3, T', H', W')

        # Crop to target
        _, _, T_tgt, H_tgt, W_tgt = target_shape
        return x[:, :, :T_tgt, :H_tgt, :W_tgt]


# =============================================================================
# Full VAE
# =============================================================================

class CausalVideoAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.mean_of_means = mx.zeros((128,))
        self.std_of_means = mx.ones((128,))

    def encode(self, x_ncdhw: mx.array) -> mx.array:
        """(1, 3, T, H, W) NCDHW → normalized latent (128, T_l, H_l, W_l)"""
        latent = self.encoder(x_ncdhw)[0]  # (128, T_l, H_l, W_l)
        mean = mx.reshape(self.mean_of_means, (-1, 1, 1, 1))
        std = mx.reshape(self.std_of_means, (-1, 1, 1, 1))
        return (latent - mean) / std

    def decode(self, z: mx.array, target_shape: Optional[Tuple[int, ...]] = None) -> mx.array:
        """(128, T_l, H_l, W_l) → (1, 3, T, H, W) NCDHW in [-1, 1]"""
        mean = mx.reshape(self.mean_of_means, (-1, 1, 1, 1))
        std = mx.reshape(self.std_of_means, (-1, 1, 1, 1))
        z = z * std + mean
        z = mx.expand_dims(z, 0)  # add batch dim

        if target_shape is None:
            _, _, T_l, H_l, W_l = z.shape
            # LTX VAE: 8x temporal (3 downsamplers), 32x spatial (patchify 4x + 3 downsamplers)
            target_shape = (1, 3, (T_l - 1) * 8 + 1, H_l * 32, W_l * 32)

        return self.decoder(z, target_shape)


# =============================================================================
# Weight Loader
# =============================================================================

def load_weights_from_safetensors(model: CausalVideoAutoencoder, path: str):
    """Load diffusers-format safetensors into NHWC-native MLX model."""
    import safetensors

    with safetensors.safe_open(path, framework="np") as f:
        keys = sorted(f.keys())

    loaded_count = 0
    with safetensors.safe_open(path, framework="np") as f:
        for key in keys:
            weight = mx.array(f.get_tensor(key))

            # Convert Conv3d weights: PT (out,in,T,H,W) → MLX (out,T,H,W,in)
            if weight.ndim == 5 and 'weight' in key:
                weight = pt_to_mlx_conv3d(weight)

            if _set_weight(model, key, weight):
                loaded_count += 1

    # Statistics
    with safetensors.safe_open(path, framework="np") as f:
        if 'latents_mean' in f.keys():
            model.mean_of_means = mx.array(f.get_tensor('latents_mean'))
        if 'latents_std' in f.keys():
            model.std_of_means = mx.array(f.get_tensor('latents_std'))

    print(f'Loaded {loaded_count}/{len(keys)} weight tensors from {path}')


def _set_weight(model, key: str, weight: mx.array) -> bool:
    """Set a single weight by diffusers key path. Returns True if set successfully."""
    parts = key.split('.')

    # Navigate to the target sub-module
    if parts[0] == 'encoder':
        obj = model.encoder
    elif parts[0] == 'decoder':
        obj = model.decoder
    else:
        return False  # statistics keys

    idx = 1
    while idx < len(parts):
        p = parts[idx]

        if p in ('conv_in', 'conv_out', 'mid_block', 'norm_out', 'upsampler', 'downsampler'):
            obj = getattr(obj, p)
            idx += 1
        elif p in ('down_blocks', 'up_blocks'):
            blk_idx = int(parts[idx + 1])
            lst = getattr(obj, p)
            obj = lst[blk_idx]
            idx += 2
        elif p == 'resnets':
            r_idx = int(parts[idx + 1])
            obj = obj.resnets[r_idx]
            idx += 2
        elif p == 'downsamplers':
            # diffusers downsamplers.0.conv → our downsampler.conv
            obj = obj.downsampler
            idx += 1  # skip '0'
        elif p == 'upsamplers':
            obj = obj.upsampler
            idx += 1  # skip '0'
        elif p == 'conv1':
            obj = obj.conv1
            idx += 1
        elif p == 'conv2':
            obj = obj.conv2
            idx += 1
        elif p == 'conv_shortcut':
            obj = obj.conv_shortcut
            idx += 1
        elif p == 'norm3':
            # LayerNorm in shortcut path: has weight and bias
            param_name = parts[idx + 1]
            if hasattr(obj, 'norm3') and hasattr(obj.norm3, param_name):
                setattr(obj.norm3, param_name, weight)
                return True
            return False
        elif p == 'conv':
            # Handle conv.weight, conv.bias, or nested conv.conv.weight
            next_p = parts[idx + 1] if idx + 1 < len(parts) else None

            if next_p == 'conv':
                # Double-nested: upsampler.conv.conv.weight → drill through CausalConv3d
                if hasattr(obj, 'conv'):
                    inner = obj.conv  # CausalConv3d
                    if hasattr(inner, 'conv'):
                        inner = inner.conv  # nn.Conv3d
                    param_name = parts[idx + 2]  # 'weight' or 'bias'
                    if hasattr(inner, param_name):
                        setattr(inner, param_name, weight)
                        return True
                return False

            # Single conv: param_name is 'weight' or 'bias'
            param_name = next_p
            if hasattr(obj, 'conv'):
                inner = obj.conv
                # Drill through CausalConv3d wrapper if needed
                if hasattr(inner, 'conv') and isinstance(inner.conv, nn.Conv3d):
                    inner = inner.conv
                if hasattr(inner, param_name):
                    setattr(inner, param_name, weight)
                    return True
            return False
        elif p in ('weight', 'bias'):
            if hasattr(obj, p):
                setattr(obj, p, weight)
                return True
            # Try obj.conv.weight
            if hasattr(obj, 'conv') and hasattr(obj.conv, p):
                setattr(obj.conv, p, weight)
                return True
            return False
        else:
            idx += 1

    return False


# =============================================================================
# Public API (NCDHW external, compatible with PyTorch LtxVAE)
# =============================================================================

class LtxVAE:
    """MLX VAE — same API as PyTorch flash_head.ltx_video.ltx_vae.LtxVAE."""

    def __init__(self, safetensors_path: str = None):
        self.model = CausalVideoAutoencoder()
        if safetensors_path is not None:
            load_weights_from_safetensors(self.model, safetensors_path)

    def encode(self, video: mx.array) -> mx.array:
        """video: (1, 3, T, H, W) NCDHW in [-1,1] → latent: (128, T_l, H_l, W_l)"""
        return self.model.encode(video)

    def decode(self, zs: mx.array) -> mx.array:
        """zs: (128, T_l, H_l, W_l) → video: (1, 3, T, H, W) NCDHW in [-1,1]"""
        return self.model.decode(zs)
