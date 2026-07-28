"""
MLX WanModelAudioProject — diffusion transformer with RoPE + AudioProj.

Mirrors flash_head/src/modules/flash_head_model.py using MLX native ops.
Complex-number RoPE is implemented with real-valued arithmetic (cos/sin decomposition)
since MLX does not support complex64.

Architecture (Lite model, 1.3B params, 30 layers):
  Input:  noise x (1, 128, 5, 16, 16) + ref_latent y (1, 128, 5, 16, 16)
  Concat → (1, 256, 5, 16, 16)
  PatchEmbed (1×1×1 Conv3d) → (1, 5*16*16, 1536) sequence
  30× DiTAudioBlock (SelfAttn+RoPE + CrossAttn+AudioCtx + FFN+Modulation)
  Head → (1, 5*16*16, 128)
  Unpatchify → (1, 128, 5, 16, 16)
"""

import math
from typing import List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# RoPE — real arithmetic (MLX has no complex64)
# ═══════════════════════════════════════════════════════════════════════════════

def sinusoidal_embedding_1d(dim: int, position: mx.array) -> mx.array:
    """1D sinusoidal timestep embedding (matches PyTorch's torch.outer)."""
    half_dim = dim // 2
    exponent = -mx.arange(0, half_dim, dtype=mx.float32) * (math.log(10000.0) / half_dim)
    # MLX has no torch.outer — use explicit dim expansion for outer product
    pos = position.astype(mx.float32)
    if pos.ndim == 0:
        pos = mx.expand_dims(pos, 0)
    sinusoid = pos[:, None] * mx.exp(exponent)     # (N, 1) * (half,)= (N, half)
    return mx.concatenate([mx.cos(sinusoid), mx.sin(sinusoid)], axis=-1)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0) -> Tuple[mx.array, mx.array]:
    """
    Precompute RoPE frequencies as (cos, sin) pair.
    Each has shape (end, dim//2).

    Equivalent to torch.polar(torch.ones_like(freqs), freqs).
    """
    half = dim // 2
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32)[:half] / dim))
    t = mx.arange(end, dtype=mx.float32)
    freqs = mx.outer(t, freqs)          # (end, half)
    return mx.cos(freqs), mx.sin(freqs)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0) -> Tuple[mx.array, mx.array]:
    """
    3D RoPE: frame, height, width each get a portion of the head dimension.

    head_dim breakdown (example head_dim=128):
      frame: 128 - 2*42 = 44  (largest, for temporal)
      height: 42               (for vertical)
      width:  42               (for horizontal)
    """
    f_dim = dim - 2 * (dim // 3)
    h_dim = dim // 3
    w_dim = dim // 3

    f_cos, f_sin = precompute_freqs_cis(f_dim, end, theta)
    h_cos, h_sin = precompute_freqs_cis(h_dim, end, theta)
    w_cos, w_sin = precompute_freqs_cis(w_dim, end, theta)

    cos = mx.concatenate([f_cos, h_cos, w_cos], axis=-1)
    sin = mx.concatenate([f_sin, h_sin, w_sin], axis=-1)
    return cos, sin


def rope_apply(x: mx.array, cos: mx.array, sin: mx.array,
               grid_sizes: Tuple[int, int, int]) -> mx.array:
    """
    Apply 3D RoPE via real arithmetic: (a+bi)·(cos+sin·i) = (a·cos-b·sin) + (a·sin+b·cos)i.

    Args:
        x:   (B, L, N, C)  where C = head_dim
        cos: (M, C//2)     precomputed cos frequencies
        sin: (M, C//2)     precomputed sin frequencies
        grid_sizes: (f, h, w) spatial layout

    Returns:
        (B, L, N*C)  flattened after RoPE application
    """
    B, s, n, c = x.shape
    half_c = c // 2

    # --- split frequency bands for frame / height / width ---
    f_dim = half_c - 2 * (half_c // 3)
    h_dim = half_c // 3
    w_dim = half_c // 3

    cos_f, cos_rest = mx.split(cos, [f_dim], axis=-1)
    cos_h, cos_w = mx.split(cos_rest, [h_dim], axis=-1)
    sin_f, sin_rest = mx.split(sin, [f_dim], axis=-1)
    sin_h, sin_w = mx.split(sin_rest, [h_dim], axis=-1)

    f, h, w = grid_sizes
    seq_len = f * h * w

    # --- tile each band across the 3D grid ---

    def _tile_3d(arr, spatial_dim, feat_dim, broadcast_shape):
        """Reshape 1D frequency array → 3D grid via broadcasting."""
        return mx.broadcast_to(
            mx.reshape(arr[:spatial_dim, :feat_dim], broadcast_shape),
            (f, h, w, feat_dim))

    c_f = _tile_3d(cos_f, f, f_dim, (f, 1, 1, f_dim))
    s_f = _tile_3d(sin_f, f, f_dim, (f, 1, 1, f_dim))
    c_h = _tile_3d(cos_h, h, h_dim, (1, h, 1, h_dim))
    s_h = _tile_3d(sin_h, h, h_dim, (1, h, 1, h_dim))
    c_w = _tile_3d(cos_w, w, w_dim, (1, 1, w, w_dim))
    s_w = _tile_3d(sin_w, w, w_dim, (1, 1, w, w_dim))

    # concatenate bands → (f*h*w, half_c) after reshape
    c_full = mx.reshape(mx.concatenate([c_f, c_h, c_w], axis=-1), (seq_len, half_c))
    s_full = mx.reshape(mx.concatenate([s_f, s_h, s_w], axis=-1), (seq_len, half_c))

    # --- pad to match padded sequence length (beyond seq_len, RoPE=1+0i) ---
    if seq_len < s:
        pad_len = s - seq_len
        c_pad = mx.ones((pad_len, half_c), dtype=c_full.dtype)
        s_pad = mx.zeros((pad_len, half_c), dtype=s_full.dtype)
        c_full = mx.concatenate([c_full, c_pad], axis=0)
        s_full = mx.concatenate([s_full, s_pad], axis=0)

    # broadcast to (1, s, 1, half_c)
    c_full = mx.reshape(c_full, (1, s, 1, half_c))
    s_full = mx.reshape(s_full, (1, s, 1, half_c))

    # --- complex multiplication in real form ---
    # PyTorch view_as_complex(reshape(..., -1, 2)) pairs interleaved:
    #   (e0,e1)→e0+ie1, (e2,e3)→e2+ie3, ...
    # So real=even indices, imag=odd indices
    x_real = x[:, :, :, 0::2]   # even indices: 0, 2, 4, ...
    x_imag = x[:, :, :, 1::2]   # odd indices:  1, 3, 5, ...

    out_real = x_real * c_full - x_imag * s_full
    out_imag = x_real * s_full + x_imag * c_full

    # PT's view_as_real().flatten(2) interleaves: [real0, imag0, real1, imag1, ...]
    out = mx.stack([out_real, out_imag], axis=-1)       # (B, s, N, half_c, 2)
    out = mx.reshape(out, (B, s, n, c))                 # (B, s, N, C) interleaved

    # Return (B, s, N, C) — caller flattens to (B, s, N*C)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.sqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms).astype(x.dtype) * self.weight


# ═══════════════════════════════════════════════════════════════════════════════
# Attention
# ═══════════════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(q: mx.array, k: mx.array, v: mx.array,
                                 num_heads: int) -> mx.array:
    """
    Multi-head attention.
    All inputs: (B, L, N*D) flattened format.
    Returns: (B, L, N*D).
    """
    B, L_q = q.shape[0], q.shape[1]
    head_dim = q.shape[-1] // num_heads

    q = mx.reshape(q, (B, L_q, num_heads, head_dim))
    k = mx.reshape(k, (k.shape[0], k.shape[1], num_heads, head_dim))
    v = mx.reshape(v, (v.shape[0], v.shape[1], num_heads, head_dim))

    x = mx.fast.scaled_dot_product_attention(
        mx.transpose(q, (0, 2, 1, 3)),   # (B, N, L_q, D)
        mx.transpose(k, (0, 2, 1, 3)),   # (B, N, L_kv, D)
        mx.transpose(v, (0, 2, 1, 3)),   # (B, N, L_kv, D)
        scale=1.0 / math.sqrt(head_dim),
    )
    x = mx.transpose(x, (0, 2, 1, 3))    # (B, L_q, N, D)
    return mx.reshape(x, (B, L_q, -1))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array,
                 grid_sizes: Tuple[int, int, int]) -> mx.array:
        B, s = x.shape[0], x.shape[1]
        n, d = self.num_heads, self.head_dim

        q = mx.reshape(self.norm_q(self.q(x)), (B, s, n, d))
        k = mx.reshape(self.norm_k(self.k(x)), (B, s, n, d))
        v = self.v(x)

        q = rope_apply(q, cos, sin, grid_sizes)
        k = rope_apply(k, cos, sin, grid_sizes)

        q = mx.reshape(q, (B, s, n * d))
        k = mx.reshape(k, (B, s, n * d))

        return self.o(scaled_dot_product_attention(q, k, v, num_heads=n))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6,
                 has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.has_image_input = has_image_input

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

    def __call__(self, x: mx.array, y: mx.array) -> mx.array:
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)

        out = scaled_dot_product_attention(q, k, v, num_heads=self.num_heads)

        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            out = out + scaled_dot_product_attention(q, k_img, v_img,
                                                     num_heads=self.num_heads)

        return self.o(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Transformer Block
# ═══════════════════════════════════════════════════════════════════════════════

class DiTAudioBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int,
                 ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approx='tanh'),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = mx.random.normal((1, 6, dim)) / math.sqrt(dim)

    def __call__(self, x: mx.array, context: mx.array, t_mod: mx.array,
                 cos: mx.array, sin: mx.array,
                 grid_sizes: Tuple[int, int, int]) -> mx.array:
        """
        Args:
            x:       (1, L, dim)      latent tokens
            context: (1, F, M, dim)   audio context tokens (F=latent frames)
            t_mod:   (1, 6, dim)      time modulation
            cos, sin: RoPE frequencies
            grid_sizes: (f, h, w)
        """
        # modulation: (1, 6, dim) → split into 6 × (1, 1, dim)
        e = mx.split(t_mod + self.modulation.astype(t_mod.dtype), 6, axis=1)

        # -- self-attention with RoPE --
        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], cos, sin, grid_sizes)
        x = x + y * e[2]

        # -- cross-attention (per latent-frame) --
        F = context.shape[1]
        x_ca = mx.reshape(self.norm3(x), (F, -1, self.dim))      # (F, L/F, dim)
        ctx_ca = mx.reshape(context, (F, -1, self.dim))           # (F, M, dim)
        x_ca = self.cross_attn(x_ca, ctx_ca)
        x = x + mx.reshape(x_ca, (1, -1, self.dim))

        # -- feed-forward --
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]

        return x


# ═══════════════════════════════════════════════════════════════════════════════
# MLP / Head / AudioProj
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """MLP with LayerNorm sandwich. Used for audio_emb and img_emb."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)


class Head(nn.Module):
    """Final projection head — maps transformer tokens back to pixel space."""

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int],
                 eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        patch_prod = patch_size[0] * patch_size[1] * patch_size[2]
        self.norm = nn.LayerNorm(dim, eps=eps, affine=False)
        self.head = nn.Linear(dim, out_dim * patch_prod)
        self.modulation = mx.random.normal((1, 2, dim)) / math.sqrt(dim)

    def __call__(self, x: mx.array, t: mx.array) -> mx.array:
        """
        Args:
            x: (B, L, dim)
            t: (B*F, dim) or (B, dim) — timestep embedding (before time_projection)
        Returns:
            (B, L, out_dim * patch_prod)
        """
        B, L, D = x.shape
        F = t.shape[0] // B

        # PT: modulation.unsqueeze(1) → (1, 1, 2, D) + t.unflatten.unsqueeze(2) → (B, F, 1, D)
        # chunk(2, dim=2) → shift (B,F,1,D), scale (B,F,1,D)
        mod = mx.reshape(self.modulation, (1, 1, 2, D))          # (1, 2, D) → (1, 1, 2, D)
        t_exp = mx.expand_dims(mx.reshape(t, (B, F, D)), 2)      # (B, F, D) → (B, F, 1, D)
        combined = mod.astype(t_exp.dtype) + t_exp               # (B, F, 2, D)
        shift, scale = mx.split(combined, 2, axis=2)             # each: (B, F, 1, D)

        x = mx.reshape(x, (B, F, L // F, D))
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return mx.reshape(x, (B, L, -1))


class AudioProjModel(nn.Module):
    """
    Projects raw audio features (Wav2Vec2 hidden states) into DiT context tokens.

    Input:
      audio_embeds (first frame):    (B, 1, 5, 12, 768)
      audio_embeds_vf (latter):      (B, N, 12, 12, 768)   N = remaining latent frames

    Output:  (B, F, context_tokens, output_dim)   e.g. (1, 5, 32, 1536)
    """

    def __init__(
        self,
        seq_len: int = 5,
        seq_len_vf: int = 12,
        blocks: int = 12,
        channels: int = 768,
        intermediate_dim: int = 512,
        output_dim: int = 1536,
        context_tokens: int = 32,
        norm_output_audio: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        input_dim = seq_len * blocks * channels          # 5 * 12 * 768 = 46080
        input_dim_vf = seq_len_vf * blocks * channels    # 12 * 12 * 768 = 110592

        self.proj1 = nn.Linear(input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()

    def __call__(self, audio_embeds: mx.array, audio_embeds_vf: mx.array) -> mx.array:
        F = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B = audio_embeds.shape[0]

        # first frame: (B, 1, 5, 12, 768) → (B, 5*12*768) → (B, 512)
        a = mx.reshape(audio_embeds, (B, -1))
        a = nn.relu(self.proj1(a))
        a = mx.reshape(a, (B, 1, self.intermediate_dim))

        # latter frames: (B, N, 12, 12, 768) → (B*N, 12*12*768) → (B*N, 512)
        N = audio_embeds_vf.shape[1]
        a_vf = mx.reshape(audio_embeds_vf, (B * N, -1))
        a_vf = nn.relu(self.proj1_vf(a_vf))
        a_vf = mx.reshape(a_vf, (B, N, self.intermediate_dim))

        # concat → (B, F, 512) → (B*F, 512)
        a_c = mx.concatenate([a, a_vf], axis=1)
        a_c = mx.reshape(a_c, (B * F, self.intermediate_dim))

        # project → context tokens
        a_c = nn.relu(self.proj2(a_c))
        tokens = self.proj3(a_c)  # (B*F, context_tokens * output_dim)
        tokens = mx.reshape(tokens, (B * F, self.context_tokens, self.output_dim))
        tokens = self.norm(tokens)
        tokens = mx.reshape(tokens, (B, F, self.context_tokens, self.output_dim))
        return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════════════════════

class WanModelAudioProject(nn.Module):
    """
    MLX port of the SoulX-FlashHead diffusion transformer.

    Lite config (1.3B):
      dim=1536, ffn_dim=8960, num_heads=12, num_layers=30
      in_dim=256, out_dim=128, patch_size=(1,1,1), vae_stride=(8,32,32)
    """

    def __init__(
        self,
        dim: int = 1536,
        in_dim: int = 256,
        ffn_dim: int = 8960,
        out_dim: int = 128,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        vae_stride: Tuple[int, int, int] = (8, 32, 32),
        patch_size: Tuple[int, int, int] = (1, 1, 1),
        num_heads: int = 12,
        num_layers: int = 30,
        has_image_input: bool = False,
        audio_window: int = 5,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.vae_stride = vae_stride
        self.vae_scale = vae_stride[0]   # temporal stride (8 for LTX)
        self.audio_window = audio_window
        self.num_layers = num_layers
        self.out_dim = out_dim

        # Patch embedding: Conv3d 1×1×1 (pointwise projection)
        self.patch_embedding = nn.Conv3d(in_dim, dim,
                                         kernel_size=patch_size,
                                         stride=patch_size, padding=0)

        # Text embedding (unused at inference but kept for weight compat)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approx='tanh'),
            nn.Linear(dim, dim),
        )

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )

        # Transformer blocks
        self.blocks = [
            DiTAudioBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ]

        # Output head
        self.head = Head(dim, out_dim, patch_size, eps)

        # RoPE frequencies (precomputed once, moved to device on first forward)
        head_dim = dim // num_heads
        cos, sin = precompute_freqs_cis_3d(head_dim)
        self.freqs_cos = cos
        self.freqs_sin = sin

        # Audio embedding MLP
        self.audio_emb = MLP(768, dim)

        # Image embedding MLP (Pro model)
        if has_image_input:
            self.img_emb = MLP(1280, dim)

        # Audio projection (Wav2Vec2 features → DiT context tokens)
        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + self.vae_scale - 1,
            intermediate_dim=512,
            output_dim=dim,          # matches DiT hidden dim
            context_tokens=32,
            norm_output_audio=True,
        )

    # ── patchify / unpatchify (NCDHW↔sequence, MLX NHWC Conv3d) ──

    def _ncdhw_to_nhwc(self, x: mx.array) -> mx.array:
        """(B, C, T, H, W) → (B, T, H, W, C)"""
        return mx.transpose(x, (0, 2, 3, 4, 1))

    def _nhwc_to_ncdhw(self, x: mx.array) -> mx.array:
        """(B, T, H, W, C) → (B, C, T, H, W)"""
        return mx.transpose(x, (0, 4, 1, 2, 3))

    def patchify(self, x: mx.array) -> Tuple[mx.array, Tuple[int, int, int]]:
        """
        x: (B, C, T, H, W) NCDHW
        Returns: (B, T*H*W, dim) sequence + grid_sizes (T, H, W)
        """
        B, C, T, H, W = x.shape
        grid_size = (T, H, W)

        # NCDHW → NHWC for MLX Conv3d
        x_nhwc = self._ncdhw_to_nhwc(x)            # (B, T, H, W, C)
        x_nhwc = self.patch_embedding(x_nhwc)       # (B, T, H, W, dim)
        x_seq = mx.reshape(x_nhwc, (B, T * H * W, self.dim))
        return x_seq, grid_size

    def unpatchify(self, x: mx.array, grid_size: Tuple[int, int, int]) -> mx.array:
        """
        x: (B, T*H*W, out_dim * patch_prod)
        Returns: (B, out_dim, T, H, W) NCDHW
        """
        f, h, w = grid_size
        B = x.shape[0]
        pT, pH, pW = self.patch_size
        # (B, f*h*w, out_dim * pT*pH*pW) → (B, out_dim, f*pT, h*pH, w*pW)
        x = mx.reshape(x, (B, f, h, w, pT, pH, pW, self.out_dim))
        x = mx.transpose(x, (0, 7, 1, 4, 2, 5, 3, 6))  # (B, out, f, pT, h, pH, w, pW)
        x = mx.reshape(x, (B, self.out_dim, f * pT, h * pH, w * pW))
        return x

    # ── forward ──

    def __call__(
        self,
        x: mx.array,              # (1, 128, T_l, H_l, W_l) noise latent
        timestep: mx.array,       # (1,) or scalar — diffusion timestep
        context: mx.array,        # (1, num_frames, 5, 12, 768)  audio features
        y: Optional[mx.array] = None,  # (1, 128, T_l, H_l, W_l) reference latent
    ) -> mx.array:
        """
        Single denoising step prediction.

        Returns: (1, out_dim, T_l, H_l, W_l) flow prediction, NCDHW.
        """
        B, C, T_l, H_l, W_l = x.shape

        # --- concat noise + reference latent ---
        if y is not None:
            x = mx.concatenate([x, y], axis=1)  # (1, 256, T_l, H_l, W_l)

        # --- patchify ---
        x_seq, grid_sizes = self.patchify(x)    # (1, L, dim)

        # --- time embedding ---
        t_emb = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.astype(x_seq.dtype))
        )  # (1, dim)
        t_mod = mx.reshape(self.time_projection(t_emb), (B, 6, self.dim))  # (1, 6, dim)

        # --- audio context processing ---
        # context: (1, num_frames, 5, 12, 768)
        # Split: first frame (full window) + latter frames (sliding window)
        first_frame_audio = context[:, :1, :, :, :]    # (1, 1, 5, 12, 768)
        latter = context[:, 1:, :, :, :]               # (1, num_frames-1, 5, 12, 768)

        n_latent = (context.shape[1] - 1) // self.vae_scale
        latter = mx.reshape(latter, (B, n_latent, self.vae_scale, 5, 12, 768))

        mid_idx = self.audio_window // 2  # 2

        first_g = latter[:, :, :1, :mid_idx + 1, :, :]           # (B, n_l, 1, 3, 12, 768)
        mid_g = latter[:, :, 1:-1, mid_idx:mid_idx + 1, :, :]    # (B, n_l, 6, 1, 12, 768)
        last_g = latter[:, :, -1:, mid_idx:, :, :]               # (B, n_l, 1, 3, 12, 768)

        first_g = mx.reshape(first_g, (B, n_latent, 3, 12, 768))
        mid_g = mx.reshape(mid_g, (B, n_latent, 6, 12, 768))
        last_g = mx.reshape(last_g, (B, n_latent, 3, 12, 768))

        latter_processed = mx.concatenate([first_g, mid_g, last_g], axis=2)
        # (B, n_latent, 12, 12, 768)

        audio_ctx = self.audio_proj(first_frame_audio, latter_processed)
        # (1, F, 32, dim)  where F = 1 + n_latent

        # --- transformer blocks ---
        for block in self.blocks:
            x_seq = block(x_seq, audio_ctx, t_mod,
                          self.freqs_cos, self.freqs_sin, grid_sizes)

        # --- head + unpatchify ---
        x_seq = self.head(x_seq, t_emb)
        x_out = self.unpatchify(x_seq, grid_sizes)
        return x_out


# ═══════════════════════════════════════════════════════════════════════════════
# Weight Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _navigate_and_set(model, parts, weight) -> bool:
    """
    Navigate the MLX model tree following diffusers key parts, then set the weight.

    Key examples:
      patch_embedding.weight
      blocks.0.self_attn.q.weight
      blocks.0.ffn.0.weight
      blocks.0.modulation              (no .weight suffix!)
      audio_emb.proj.0.weight
      head.modulation
    """
    obj = model
    i = 0
    n = len(parts)

    while i < n:
        p = parts[i]

        # ── sub-module / leaf attributes (checked first to shadow top-level routing) ──
        if p in ('self_attn', 'cross_attn'):
            obj = getattr(obj, p)
            i += 1
        elif p == 'ffn':
            i += 1
            if i < n and parts[i].isdigit():
                obj = obj.ffn.layers[int(parts[i])]
                i += 1
        elif p == 'proj':
            i += 1
            if i < n and parts[i].isdigit():
                obj = obj.proj.layers[int(parts[i])]
                i += 1
        elif p in ('q', 'k', 'v', 'o', 'head',
                    'norm_q', 'norm_k', 'norm_k_img',
                    'norm1', 'norm2', 'norm3',
                    'proj1', 'proj1_vf', 'proj2', 'proj3', 'norm'):
            obj = getattr(obj, p)
            i += 1

        # ── top-level dispatch ──
        elif p == 'patch_embedding':
            obj = model.patch_embedding
            i += 1
        elif p == 'text_embedding':
            i += 1
            if i < n and parts[i].isdigit():
                obj = model.text_embedding.layers[int(parts[i])]
                i += 1
        elif p == 'time_embedding':
            i += 1
            if i < n and parts[i].isdigit():
                obj = model.time_embedding.layers[int(parts[i])]
                i += 1
        elif p == 'time_projection':
            i += 1
            if i < n and parts[i].isdigit():
                obj = model.time_projection.layers[int(parts[i])]
                i += 1
        elif p == 'blocks':
            i += 1
            blk_idx = int(parts[i])
            obj = model.blocks[blk_idx]
            i += 1
        elif p == 'head':
            obj = model.head
            i += 1
        elif p == 'audio_emb':
            obj = model.audio_emb
            i += 1
        elif p == 'img_emb':
            obj = model.img_emb
            i += 1
        elif p == 'audio_proj':
            obj = model.audio_proj
            i += 1

        # ── weight / bias / parameter setters ──
        elif p == 'weight':
            if isinstance(obj, nn.Conv3d):
                weight = mx.transpose(weight, (0, 2, 3, 4, 1))
            obj.weight = weight
            return True
        elif p == 'bias':
            obj.bias = weight
            return True
        elif p == 'modulation':
            obj.modulation = weight
            return True
        else:
            i += 1

    return False


def load_diffusion_weights(model: WanModelAudioProject, safetensors_path: str):
    """Load diffusers-format safetensors into the MLX diffusion model."""
    import safetensors

    with safetensors.safe_open(safetensors_path, framework="np") as f:
        keys = sorted(f.keys())

    loaded = 0
    skipped = []

    with safetensors.safe_open(safetensors_path, framework="np") as f:
        for key in keys:
            parts = key.split('.')
            weight = mx.array(f.get_tensor(key))
            if _navigate_and_set(model, parts, weight):
                loaded += 1
            else:
                skipped.append(key)

    print(f'Loaded {loaded}/{len(keys)} weight tensors from {safetensors_path}')
    if skipped:
        print(f'Skipped {len(skipped)} keys: {skipped[:10]}...')
