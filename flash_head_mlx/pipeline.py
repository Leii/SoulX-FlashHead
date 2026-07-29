"""
MLX FlashHeadPipeline — end-to-end audio-driven lip-sync video generation.

All model computation (VAE + Diffusion) uses native MLX for fast inference.
Wav2Vec2 audio encoding uses PyTorch (HuggingFace).
"""

import os
import time
from typing import Dict, Optional, Tuple
from PIL import Image
import numpy as np

import mlx.core as mx

# MLX diffusion model
from flash_head_mlx.modules.flash_head_model import (
    WanModelAudioProject,
    load_diffusion_weights,
)

# MLX VAE
_mlx_vae = None


def _get_mlx_vae(ckpt_dir: str):
    """Load the MLX LTX VAE (singleton)."""
    global _mlx_vae
    if _mlx_vae is None:
        from flash_head_mlx.ltx_vae import LtxVAE
        vae_safetensors = os.path.join(ckpt_dir, "VAE_LTX", "diffusion_pytorch_model.safetensors")
        _mlx_vae = LtxVAE(safetensors_path=vae_safetensors)
    return _mlx_vae


def vae_encode(video_mlx: mx.array, ckpt_dir: str) -> mx.array:
    """Encode video to latent using MLX VAE. video: (1,3,T,H,W) NCDHW in [-1,1]."""
    vae = _get_mlx_vae(ckpt_dir)
    return vae.encode(video_mlx)  # (128, T_l, H_l, W_l)


def vae_decode(latent_mlx: mx.array, ckpt_dir: str) -> mx.array:
    """Decode latent to video using MLX VAE. latent: (128,T_l,H_l,W_l). Returns: (1,3,T,H,W)."""
    vae = _get_mlx_vae(ckpt_dir)
    return vae.decode(latent_mlx)  # (1, 3, T, H, W)


def timestep_transform(t: float, shift: float = 5.0, num_timesteps: int = 1000) -> float:
    """Flow-matching timestep shift."""
    t = t / num_timesteps
    new_t = shift * t / (1.0 + (shift - 1.0) * t)
    return new_t * num_timesteps


def resize_and_centercrop(image: Image.Image, target_size: Tuple[int, int]) -> mx.array:
    """Resize and center-crop a PIL image to target (H, W)."""
    tw, th = target_size[1], target_size[0]  # PIL: (W, H)

    # Resize shortest side to target
    w, h = image.size
    scale = max(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    # Center crop
    left = (new_w - tw) // 2
    top = (new_h - th) // 2
    image = image.crop((left, top, left + tw, top + th))

    # To numpy, normalize to [-1, 1]
    arr = np.array(image).astype(np.float32) / 127.5 - 1.0  # (H, W, 3)
    # To NCDHW: (1, 3, 1, H, W)
    arr = arr.transpose(2, 0, 1)[None, :, None, :, :]
    return mx.array(arr)


def match_and_blend_colors(
    video: mx.array,
    reference: mx.array,
    strength: float = 1.0,
) -> mx.array:
    """
    Color correction: match video colors to reference image.
    video, reference: (C, T, H, W) NCDHW, values in [-1, 1].

    Simplified version using histogram matching in YCbCr.
    """
    if strength <= 0.0:
        return video[0]  # remove batch dim: (3, T, H, W)

    # Convert to numpy for color correction
    video_np = np.array(video)
    ref_np = np.array(reference)

    C, T, H, W = video_np.shape
    # Match per-frame colors to reference
    # Simple approach: match mean/std in RGB space
    # Reference: (C, 1, H, W) or (C,)
    ref_mean = ref_np.mean(axis=(1, 2, 3), keepdims=True)  # (C, 1, 1, 1)
    ref_std = ref_np.std(axis=(1, 2, 3), keepdims=True) + 1e-8

    vid_mean = video_np.mean(axis=(2, 3), keepdims=True)  # (C, T, 1, 1)
    vid_std = video_np.std(axis=(2, 3), keepdims=True) + 1e-8

    corrected = (video_np - vid_mean) * (ref_std / vid_std) + ref_mean
    corrected = video_np + strength * (corrected - video_np)
    return mx.array(np.clip(corrected, -1.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class FlashHeadPipelineMLX:
    """
    MLX-native FlashHead pipeline.

    Usage:
        pipe = FlashHeadPipelineMLX(
            checkpoint_dir="models/SoulX-FlashHead-1_3B",
            model_type="lite",
            wav2vec_dir="models/wav2vec2-base-960h",
        )
        pipe.prepare_params(image_path, target_size=(512, 512), frame_num=33, ...)
        video = pipe.generate(audio_embedding)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model_type: str = "lite",
        wav2vec_dir: str = None,
        num_timesteps: int = 1000,
        use_timestep_transform: bool = True,
    ):
        self.model_type = model_type
        self.use_ltx = model_type == "lite"
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.ckpt_dir = checkpoint_dir

        if not self.use_ltx:
            raise NotImplementedError("Only 'lite' model type is supported in MLX")

        # ---- VAE (MLX native) ----
        _get_mlx_vae(checkpoint_dir)  # preload

        # ---- Diffusion Model (MLX) ----
        model_dir = os.path.join(checkpoint_dir, "Model_Lite")
        model_safetensors = os.path.join(model_dir, "diffusion_pytorch_model.safetensors")
        print(f"Loading diffusion model from {model_safetensors} ...")
        self.model = WanModelAudioProject()
        load_diffusion_weights(self.model, model_safetensors)

        # ---- Load Wav2Vec2 (PyTorch) ----
        self.audio_encoder = None
        self.wav2vec_feature_extractor = None
        if wav2vec_dir is not None and os.path.exists(wav2vec_dir):
            self._load_wav2vec2(wav2vec_dir)

        # Config
        self.vae_stride = (8, 32, 32)  # LTX VAE stride
        self.motion_frames_num = 1
        self.latent_motion_frames = None

    def _load_wav2vec2(self, wav2vec_dir: str):
        """Load Wav2Vec2 from HuggingFace (PyTorch)."""
        import torch
        from transformers import Wav2Vec2FeatureExtractor

        # Custom Wav2Vec2 import
        from flash_head.audio_analysis.wav2vec2 import Wav2Vec2Model

        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            wav2vec_dir, local_files_only=True
        ).to("cpu")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_dir, local_files_only=True
        )
        print(f"Loaded Wav2Vec2 from {wav2vec_dir}")

    # ── Parameter Preparation ──

    def prepare_params(
        self,
        cond_image_path_or_dir: str,
        target_size: Tuple[int, int] = (512, 512),
        frame_num: int = 33,
        motion_frames_num: int = 1,
        sampling_steps: int = 4,
        seed: int = 42,
        shift: float = 5.0,
        color_correction_strength: float = 0.0,
    ):
        """Preprocess reference image and set up generation parameters."""
        import glob as _glob

        self.target_h, self.target_w = target_size
        self.lat_h = self.target_h // self.vae_stride[1]  # 16
        self.lat_w = self.target_w // self.vae_stride[2]  # 16
        self.frame_num = frame_num
        self.motion_frames_num = motion_frames_num
        self.color_correction_strength = color_correction_strength

        # Load image(s)
        if os.path.isdir(cond_image_path_or_dir):
            image_paths = sorted(_glob.glob(os.path.join(cond_image_path_or_dir, "*.png")))
            if not image_paths:
                image_paths = sorted(_glob.glob(os.path.join(cond_image_path_or_dir, "*.jpg")))
        else:
            image_paths = [cond_image_path_or_dir]

        self.cond_image_tensors = {}
        self.ref_img_latents = {}

        for img_path in image_paths:
            person_name = os.path.splitext(os.path.basename(img_path))[0]
            image = Image.open(img_path).convert("RGB")
            cond_tensor = resize_and_centercrop(image, target_size)  # (1, 3, 1, H, W) NCDHW

            # Repeat to frame_num for VAE encoding
            video_frames = mx.repeat(cond_tensor, frame_num, axis=2)  # (1, 3, T, H, W)
            ref_latent = vae_encode(video_frames, self.ckpt_dir)  # (128, T_l, H_l, W_l)

            self.cond_image_tensors[person_name] = cond_tensor
            self.ref_img_latents[person_name] = ref_latent

        # Use first person as default
        if person_name:
            self.person_name = person_name
        else:
            self.person_name = list(self.cond_image_tensors.keys())[0]

        self.original_color_reference = self.cond_image_tensors[self.person_name]
        self.ref_img_latent = self.ref_img_latents[self.person_name]
        self.latent_motion_frames = mx.array(self.ref_img_latent[:, :1])  # First latent frame

        # Prepare timesteps
        if sampling_steps == 2:
            ts = [1000.0, 500.0]
        elif sampling_steps == 4:
            ts = [1000.0, 750.0, 500.0, 250.0]
        else:
            ts = list(np.linspace(self.num_timesteps, 1.0, sampling_steps, dtype=np.float32))
        ts.append(0.0)

        if self.use_timestep_transform:
            ts = [timestep_transform(t, shift, self.num_timesteps) for t in ts]
        self.timesteps = [mx.array([float(t)]) for t in ts]

        # Random number generator (MLX global seed)
        self.seed = seed
        mx.random.seed(seed)

        print(f"Prepared {len(image_paths)} reference image(s), {len(ts)-1} sampling steps")

    # ── Audio Processing ──

    def preprocess_audio(
        self,
        speech_array: np.ndarray,
        sr: int = 16000,
        fps: int = 25,
    ) -> mx.array:
        """
        Encode audio to Wav2Vec2 features, then prepare sliding-window context.

        Args:
            speech_array: (N,) float audio samples at 16kHz
            sr: sample rate
            fps: video FPS

        Returns:
            (1, num_frames, audio_window, num_blocks, hidden_dim) audio features
        """
        import torch
        from einops import rearrange

        if self.audio_encoder is None:
            raise RuntimeError("Wav2Vec2 not loaded. Pass wav2vec_dir to __init__.")

        video_length = len(speech_array) * fps / sr
        audio_end_idx = int(video_length)

        # Wav2Vec2 feature extraction
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().unsqueeze(0)

        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature,
                seq_len=int(video_length),
                output_hidden_states=True,
            )

        if len(embeddings) == 0:
            raise RuntimeError("Failed to extract audio embedding")

        # Stack hidden states: (layers, seq_len, hidden_dim)
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")  # (frames, layers, hidden_dim)

        # Sliding window: 5-frame window centered on each frame
        # indices = [-2, -1, 0, 1, 2] — 2 frames before, current, 2 after
        indices = (torch.arange(2 * 2 + 1) - 2) * 1
        center_indices = torch.arange(0, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_end_idx - 1)

        audio_embedding = audio_emb[center_indices][None, ...].contiguous()
        # (1, num_frames, 5, 12, 768)

        return mx.array(audio_embedding.numpy())

    # ── Generation ──

    def generate(self, audio_embedding: mx.array) -> mx.array:
        """
        Run the diffusion sampling loop to generate video frames.

        Args:
            audio_embedding: (1, num_frames, 5, 12, 768) audio features

        Returns:
            (C, T, H, W) video tensor, values in [-1, 1]
        """
        lat_temporal = (self.frame_num - 1) // self.vae_stride[0] + 1

        # Initialize noise
        noise = mx.random.normal(
            (self.model.out_dim, lat_temporal, self.lat_h, self.lat_w),
        )
        noise = noise.astype(mx.float32)

        ref_latent_expanded = mx.expand_dims(self.ref_img_latent, 0)  # (1, 128, T_l, H_l, W_l)

        num_steps = len(self.timesteps) - 1
        for i in range(num_steps):
            t_start = time.time()

            # Inject motion frames into noise
            if self.latent_motion_frames is not None:
                mf = self.latent_motion_frames.shape[1]
                noise = mx.concatenate(
                    [self.latent_motion_frames, noise[:, mf:]], axis=1
                )

            # Model forward
            flow_pred = self.model(
                x=mx.expand_dims(noise, 0),
                timestep=self.timesteps[i],
                context=audio_embedding,
                y=ref_latent_expanded,
            )
            flow_pred = flow_pred[0]  # remove batch dim

            # Flow-matching Euler step
            t_i = self.timesteps[i] / self.num_timesteps
            t_i_1 = self.timesteps[i + 1] / self.num_timesteps

            # x_0 = x_t - v * t  (flow prediction points to data)
            x_0 = noise - flow_pred * t_i.reshape((-1, 1, 1, 1))

            # Euler step: x_{t-dt} = (1 - t_{i+1}) * x_0 + t_{i+1} * noise
            new_noise = mx.random.normal(x_0.shape)
            noise = (1.0 - t_i_1.reshape((-1, 1, 1, 1))) * x_0 + \
                    t_i_1.reshape((-1, 1, 1, 1)) * new_noise

            mx.eval(noise)

            t_elapsed = time.time() - t_start
            print(f"[generate] step {i+1}/{num_steps}: {t_elapsed:.2f}s")

        # Inject final motion frames
        if self.latent_motion_frames is not None:
            mf = self.latent_motion_frames.shape[1]
            noise = mx.concatenate(
                [self.latent_motion_frames, noise[:, mf:]], axis=1
            )

        # Decode latent to video
        print("[generate] Decoding video...")
        t_decode = time.time()
        video = vae_decode(noise, self.ckpt_dir)  # (1, C, T, H, W) NCDHW
        mx.eval(video)
        print(f"[generate] Decode: {time.time() - t_decode:.2f}s")

        # Color correction
        if self.color_correction_strength > 0.0:
            t_cc = time.time()
            video_cc = match_and_blend_colors(
                video[0], self.original_color_reference[0],
                self.color_correction_strength,
            )
            video = mx.expand_dims(video_cc, 0)
            mx.eval(video)
            print(f"[generate] Color correction: {time.time() - t_cc:.2f}s")

        # Update motion frames for next segment
        # video: (1, 3, T, H, W) NCDHW → slice temporal dim (dim 2)
        cond_frame = video[0, :, -self.motion_frames_num:, :, :]  # (3, mf, H, W)
        # VAE encode expects (1, 3, T, H, W)
        cond_frame_5d = mx.expand_dims(cond_frame, 0)  # (1, 3, mf, H, W)
        self.latent_motion_frames = vae_encode(cond_frame_5d, self.ckpt_dir)
        mx.eval(self.latent_motion_frames)

        return video[0]  # remove batch dim: (3, T, H, W)

    # ── Streaming ──

    def generate_streaming(
        self,
        audio_embedding: mx.array,
        chunk_frames: int = 8,
    ) -> mx.array:
        """
        Generate video with sliding-window streaming (lower latency, suitable for
        real-time or long videos).

        Args:
            audio_embedding: audio features for the FULL video
            chunk_frames: number of video frames per generation chunk

        Yields:
            (C, chunk_frames, H, W) video chunks
        """
        total_frames = audio_embedding.shape[1]
        lat_per_chunk = (chunk_frames - 1) // self.vae_stride[0] + 1

        for start in range(0, total_frames, chunk_frames):
            end = min(start + chunk_frames, total_frames)

            # Slice audio for this chunk, including overlap for context
            ctx_start = max(0, start - 2)  # 2 frames context
            ctx_end = min(total_frames, end + 2)
            chunk_audio = audio_embedding[:, ctx_start:ctx_end, :, :, :]

            # Adjust timesteps for streaming (fewer steps per chunk)
            # Generate this chunk
            vid_chunk = self._generate_chunk(chunk_audio, ctx_start, start, end)

            yield vid_chunk

    def _generate_chunk(
        self, audio: mx.array, ctx_start: int, chunk_start: int, chunk_end: int
    ) -> mx.array:
        """Generate a single video chunk (simplified)."""
        frames = chunk_end - chunk_start
        lat_t = (frames - 1) // self.vae_stride[0] + 1

        noise = mx.random.normal(
            (self.model.out_dim, lat_t, self.lat_h, self.lat_w),
            dtype=mx.float32,
        )

        ref_latent = mx.expand_dims(self.ref_img_latent[:, :lat_t], 0)

        for i in range(len(self.timesteps) - 1):
            if self.latent_motion_frames is not None:
                mf = min(self.latent_motion_frames.shape[1], noise.shape[1])
                noise = mx.concatenate(
                    [self.latent_motion_frames[:, :mf], noise[:, mf:]], axis=1
                )

            flow_pred = self.model(
                x=mx.expand_dims(noise, 0),
                timestep=self.timesteps[i],
                context=audio,
                y=ref_latent,
            )[0]

            t_i = self.timesteps[i] / self.num_timesteps
            t_i_1 = self.timesteps[i + 1] / self.num_timesteps
            x_0 = noise - flow_pred * t_i.reshape((-1, 1, 1, 1))
            noise = (1.0 - t_i_1.reshape((-1, 1, 1, 1))) * x_0 + \
                    t_i_1.reshape((-1, 1, 1, 1)) * mx.random.normal(x_0.shape)

            mx.eval(noise)

        video = vae_decode(noise, self.ckpt_dir)
        return video[0]  # remove batch dim: (3, T, H, W)
