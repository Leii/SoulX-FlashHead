"""
MLX CausalVideoAutoencoder — matches diffusers AutoencoderKLLTXVideo exactly.

Architecture (from safetensors keys):
  Encoder: patchify(4×4 spatial only) → conv_in(48→128) →
    [res_x×4 → compress_all → res_x_y(128→256)] →
    [res_x×3 → compress_all → res_x_y(256→512)] →
    [res_x×3 → compress_all] →
    [res_x×3] → [res_x×4 (mid)] →
    conv_out(512→129) → keep[:128]
  Decoder: conv_in(128→512) →
    [res_x×4 (mid)] → [res_x×3] →
    [res_x×3 → upsample] →
    [res_x_y(512→256) → res_x×3 → upsample] →
    [res_x_y(256→128) → res_x×4 → upsample] →
    conv_out(128→48) → unpatchify → 3 RGB
"""

from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn

from flash_head_mlx.ltx_vae.ops import (
    CausalConv3d, PixelNorm, ResnetBlock3D,
    Downsample3D, DepthToSpaceUpsample,
    patchify_nhwc, unpatchify_nhwc,
    ncdhw_to_nhwc, nhwc_to_ncdhw, pt_to_mlx_conv3d,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        pm = "zeros"  # spatial_padding_mode

        # Patchify 4×4 spatial → 48 ch (in forward)
        self.conv_in = CausalConv3d(48, 128, 3, causal=True, spatial_padding_mode=pm)

        # Block 0: res_x×4 → compress_all → res_x_y(128→256)
        self.down_0_res = [ResnetBlock3D(128, 128, causal=True, spatial_padding_mode=pm) for _ in range(4)]
        self.down_0_ds = Downsample3D(128, spatial_padding_mode=pm)
        self.down_0_xy = ResnetBlock3D(128, 256, causal=True, spatial_padding_mode=pm)

        # Block 1: res_x×3 → compress_all → res_x_y(256→512)
        self.down_1_res = [ResnetBlock3D(256, 256, causal=True, spatial_padding_mode=pm) for _ in range(3)]
        self.down_1_ds = Downsample3D(256, spatial_padding_mode=pm)
        self.down_1_xy = ResnetBlock3D(256, 512, causal=True, spatial_padding_mode=pm)

        # Block 2: res_x×3 → compress_all (no channel change)
        self.down_2_res = [ResnetBlock3D(512, 512, causal=True, spatial_padding_mode=pm) for _ in range(3)]
        self.down_2_ds = Downsample3D(512, spatial_padding_mode=pm)

        # Block 3: res_x×3 (no downsample)
        self.down_3_res = [ResnetBlock3D(512, 512, causal=True, spatial_padding_mode=pm) for _ in range(3)]

        # Mid: res_x×4
        self.mid_res = [ResnetBlock3D(512, 512, causal=True, spatial_padding_mode=pm) for _ in range(4)]

        self.norm_out = PixelNorm()
        self.act_out = nn.SiLU()
        self.conv_out = CausalConv3d(512, 129, 3, causal=True, spatial_padding_mode=pm)

    def __call__(self, x_ncdhw: mx.array) -> mx.array:
        x = ncdhw_to_nhwc(x_ncdhw)
        x = patchify_nhwc(x, patch_size=4)
        x = self.conv_in(x)

        for r in self.down_0_res: x = r(x)
        x = self.down_0_ds(x)
        x = self.down_0_xy(x)

        for r in self.down_1_res: x = r(x)
        x = self.down_1_ds(x)
        x = self.down_1_xy(x)

        for r in self.down_2_res: x = r(x)
        x = self.down_2_ds(x)

        for r in self.down_3_res: x = r(x)
        for r in self.mid_res: x = r(x)

        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)
        x = x[:, :, :, :, :128]  # drop log-var
        return nhwc_to_ncdhw(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════════════════════════════

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        pm = "zeros"

        self.conv_in = CausalConv3d(128, 512, 3, causal=False, spatial_padding_mode=pm)

        # Mid: res_x×4
        self.mid_res = [ResnetBlock3D(512, 512, causal=False, spatial_padding_mode=pm) for _ in range(4)]

        # Up 0: res_x×3 (no upsampler)
        self.up_0_res = [ResnetBlock3D(512, 512, causal=False, spatial_padding_mode=pm) for _ in range(3)]

        # Up 1: res_x×3 → upsample
        self.up_1_res = [ResnetBlock3D(512, 512, causal=False, spatial_padding_mode=pm) for _ in range(3)]
        self.up_1_us = DepthToSpaceUpsample(512, causal=False, spatial_padding_mode=pm)

        # Up 2: res_x_y(512→256) → res_x×3 → upsample
        self.up_2_xy = ResnetBlock3D(512, 256, causal=False, spatial_padding_mode=pm)
        self.up_2_res = [ResnetBlock3D(256, 256, causal=False, spatial_padding_mode=pm) for _ in range(3)]
        self.up_2_us = DepthToSpaceUpsample(256, causal=False, spatial_padding_mode=pm)

        # Up 3: res_x_y(256→128) → res_x×4 → upsample
        self.up_3_xy = ResnetBlock3D(256, 128, causal=False, spatial_padding_mode=pm)
        self.up_3_res = [ResnetBlock3D(128, 128, causal=False, spatial_padding_mode=pm) for _ in range(4)]
        self.up_3_us = DepthToSpaceUpsample(128, causal=False, spatial_padding_mode=pm)

        self.norm_out = PixelNorm()
        self.act_out = nn.SiLU()
        self.conv_out = CausalConv3d(128, 48, 3, causal=False, spatial_padding_mode=pm)

    def __call__(self, x_ncdhw: mx.array, target_shape: Optional[Tuple[int, ...]] = None) -> mx.array:
        """PT decoder order: mid → res3 → US → res3 → xy → US → res3 → xy → US → res4"""
        x = ncdhw_to_nhwc(x_ncdhw)
        x = self.conv_in(x)

        # Mid + first res group (512→512)
        for r in self.mid_res: x = r(x)
        for r in self.up_0_res: x = r(x)

        # First upsample, then res group at 512
        x = self.up_1_us(x)
        for r in self.up_1_res: x = r(x)

        # res_x_y 512→256, second upsample, then res group at 256
        x = self.up_2_xy(x)
        x = self.up_2_us(x)
        for r in self.up_2_res: x = r(x)

        # res_x_y 256→128, third upsample, then res group at 128
        x = self.up_3_xy(x)
        x = self.up_3_us(x)
        for r in self.up_3_res: x = r(x)

        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.conv_out(x)

        x = unpatchify_nhwc(x, patch_size=4, out_channels=3)
        x = nhwc_to_ncdhw(x)

        # Crop to target if specified (otherwise return full output, matching PT behavior)
        if target_shape is not None:
            _, _, T_tgt, H_tgt, W_tgt = target_shape
            return x[:, :, :T_tgt, :H_tgt, :W_tgt]
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# Full VAE
# ═══════════════════════════════════════════════════════════════════════════════

class CausalVideoAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.mean_of_means = mx.zeros((128,))
        self.std_of_means = mx.ones((128,))

    def encode(self, x_ncdhw: mx.array) -> mx.array:
        latent = self.encoder(x_ncdhw)[0]
        mean = mx.reshape(self.mean_of_means, (-1, 1, 1, 1))
        std = mx.reshape(self.std_of_means, (-1, 1, 1, 1))
        return (latent - mean) / std

    def decode(self, z: mx.array, target_shape: Optional[Tuple[int, ...]] = None) -> mx.array:
        mean = mx.reshape(self.mean_of_means, (-1, 1, 1, 1))
        std = mx.reshape(self.std_of_means, (-1, 1, 1, 1))
        z = z * std + mean
        z = mx.expand_dims(z, 0)
        if target_shape is None:
            _, _, T_l, H_l, W_l = z.shape
            target_shape = (1, 3, (T_l - 1) * 8 + 1, H_l * 32, W_l * 32)
        return self.decoder(z, target_shape)


# ═══════════════════════════════════════════════════════════════════════════════
# Weight Loader — maps diffusers key paths to MLX model attributes
# ═══════════════════════════════════════════════════════════════════════════════

def load_weights_from_safetensors(model: CausalVideoAutoencoder, path: str):
    """Load diffusers-format safetensors into MLX VAE."""
    import safetensors

    with safetensors.safe_open(path, framework="np") as f:
        keys = sorted(f.keys())

    loaded = 0
    skipped = []

    with safetensors.safe_open(path, framework="np") as f:
        for key in keys:
            weight = mx.array(f.get_tensor(key))
            if weight.ndim == 5 and 'weight' in key:
                weight = pt_to_mlx_conv3d(weight)
            if _set_weight(model, key, weight):
                loaded += 1
            else:
                skipped.append(key)

    # Latent statistics
    with safetensors.safe_open(path, framework="np") as f:
        if 'latents_mean' in f.keys():
            model.mean_of_means = mx.array(f.get_tensor('latents_mean'))
        if 'latents_std' in f.keys():
            model.std_of_means = mx.array(f.get_tensor('latents_std'))

    print(f'Loaded {loaded}/{len(keys)} weight tensors from {path}')
    if skipped:
        print(f'  (skipped {len(skipped)}): {skipped}')


def _set_weight(model, key: str, weight: mx.array) -> bool:
    """
    Navigate diffusers key path → MLX model attribute, then set weight.

    Handles key patterns like:
      encoder.down_blocks.0.resnets.1.conv1.conv.weight
      encoder.down_blocks.0.downsamplers.0.conv.weight
      encoder.down_blocks.0.conv_out.conv_shortcut.conv.weight
      decoder.up_blocks.2.conv_in.norm3.bias
      decoder.up_blocks.1.upsamplers.0.conv.conv.weight
    """
    parts = key.split('.')

    # Determine encoder vs decoder
    is_enc = parts[0] == 'encoder'
    is_dec = parts[0] == 'decoder'
    if not (is_enc or is_dec):
        return False  # latents_mean, latents_std

    enc = model.encoder if is_enc else None
    dec = model.decoder if is_dec else None
    comp = enc if is_enc else dec

    # ── BLOCK INDEX MAPPING ──
    # Diffusers → our names for encoder down_blocks
    ENC_DOWN_MAP = {
        0: ('down_0_res', 4, 'down_0_ds', 'down_0_xy'),
        1: ('down_1_res', 3, 'down_1_ds', 'down_1_xy'),
        2: ('down_2_res', 3, 'down_2_ds', None),       # no conv_out
        3: ('down_3_res', 3, None, None),               # no downsample
    }
    # Diffusers → our names for decoder up_blocks
    # Format: (res_name, res_count, [us_name]) or (xy_name, res_name, res_count, us_name)
    DEC_UP_MAP = {
        0: {'res_name': 'up_0_res', 'res_count': 3, 'us_name': None},
        1: {'res_name': 'up_1_res', 'res_count': 3, 'us_name': 'up_1_us'},
        2: {'xy_name': 'up_2_xy', 'res_name': 'up_2_res', 'res_count': 3, 'us_name': 'up_2_us'},
        3: {'xy_name': 'up_3_xy', 'res_name': 'up_3_res', 'res_count': 4, 'us_name': 'up_3_us'},
    }

    # ── Parse key path ──
    idx = 1  # skip 'encoder'/'decoder'

    # conv_in / conv_out at top level
    if parts[idx] in ('conv_in', 'conv_out'):
        obj = getattr(comp, parts[idx])  # CausalConv3d
        # Next should be 'conv', then 'weight'/'bias'
        return _set_conv_weight(obj, parts, idx + 1, weight)

    # mid_block
    if parts[idx] == 'mid_block':
        # mid_block.resnets.N.conv1.conv.weight
        idx += 2  # skip 'mid_block.resnets'
        res_idx = int(parts[idx]); idx += 1
        obj = comp.mid_res[res_idx]
        return _set_resnet_weight(obj, parts, idx, weight)

    # down_blocks / up_blocks
    if parts[idx] in ('down_blocks', 'up_blocks'):
        blk_idx = int(parts[idx + 1]); idx += 2

        if is_enc:
            res_name, res_count, ds_name, xy_name = ENC_DOWN_MAP[blk_idx]
        else:
            # Decoder up_blocks have different structure per index
            mapping = DEC_UP_MAP[blk_idx]

        # What sub-section of the block?
        section = parts[idx]  # 'resnets', 'downsamplers', 'upsamplers', 'conv_out', 'conv_in'

        if section == 'resnets':
            idx += 1
            res_idx = int(parts[idx]); idx += 1
            if is_enc:
                obj = getattr(comp, res_name)[res_idx]
            else:
                obj = getattr(comp, mapping['res_name'])[res_idx]
            return _set_resnet_weight(obj, parts, idx, weight)

        elif section == 'downsamplers':
            idx += 2
            obj = getattr(comp, ds_name)
            return _set_conv_weight(obj.conv, parts, idx, weight)

        elif section == 'upsamplers':
            idx += 2
            obj = getattr(comp, mapping['us_name'])
            return _set_conv_weight(obj.conv, parts, idx, weight)

        elif section == 'conv_out':
            idx += 1
            obj = getattr(comp, xy_name)
            return _set_resnet_weight(obj, parts, idx, weight)

        elif section == 'conv_in':
            idx += 1
            obj = getattr(comp, mapping['xy_name'])
            return _set_resnet_weight(obj, parts, idx, weight)

    return False


def _set_resnet_weight(obj, parts, idx, weight):
    """Set weight on ResnetBlock3D sub-module. parts[idx] is conv1/conv2/conv_shortcut/norm3."""
    sub = parts[idx]
    if sub == 'conv1':
        return _set_conv_weight(obj.conv1, parts, idx + 1, weight)
    elif sub == 'conv2':
        return _set_conv_weight(obj.conv2, parts, idx + 1, weight)
    elif sub == 'conv_shortcut':
        # PlainConv3d — obj.conv_shortcut.conv is nn.Conv3d
        return _set_conv_weight(obj.conv_shortcut, parts, idx + 1, weight)
    elif sub == 'norm3':
        param = parts[idx + 1]  # 'weight' or 'bias'
        setattr(obj.norm3, param, weight)
        return True
    return False


def _set_conv_weight(obj, parts, idx, weight):
    """
    Set weight on an nn.Conv3d. Handles key paths like:
      conv.weight           (1 'conv' level)
      conv.conv.weight      (2 'conv' levels — upsampler)
    Navigates through CausalConv3d/PlainConv3d wrappers to inner nn.Conv3d.
    """
    # Skip 'conv' parts — there may be 1 or 2 levels
    while idx < len(parts) and parts[idx] == 'conv':
        # Drill through wrapper to inner conv
        if hasattr(obj, 'conv') and isinstance(obj.conv, nn.Conv3d):
            obj = obj.conv
        idx += 1

    # One more drill-through after skipping all 'conv' parts
    if hasattr(obj, 'conv') and isinstance(obj.conv, nn.Conv3d):
        obj = obj.conv

    p = parts[idx] if idx < len(parts) else None
    if p == 'weight':
        obj.weight = weight
        return True
    elif p == 'bias':
        obj.bias = weight
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

class LtxVAE:
    """MLX VAE — same API as PyTorch version."""

    def __init__(self, safetensors_path: str = None):
        self.model = CausalVideoAutoencoder()
        if safetensors_path is not None:
            load_weights_from_safetensors(self.model, safetensors_path)

    def encode(self, video: mx.array) -> mx.array:
        return self.model.encode(video)

    def decode(self, zs: mx.array) -> mx.array:
        return self.model.decode(zs)
