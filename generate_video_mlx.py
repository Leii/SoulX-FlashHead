#!/usr/bin/env python3
"""
MLX-native video generation script for SoulX-FlashHead.

Usage (same as generate_video.py):
    python generate_video_mlx.py \
        --ckpt_dir models/SoulX-FlashHead-1_3B \
        --wav2vec_dir models/wav2vec2-base-960h \
        --model_type lite \
        --cond_image assets/example.png \
        --audio_path assets/example.wav \
        --sample_steps 4

Key differences from the PyTorch version:
    - Uses MLX for all model computation (VAE + Diffusion)
    - Wav2Vec2 audio encoder still uses PyTorch (small, non-bottleneck)
    - No distributed/USP support (single-device MLX)
"""

import argparse
import os
import sys
import time
import numpy as np
import librosa
import imageio
import subprocess
from datetime import datetime
from loguru import logger

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_head_mlx import FlashHeadPipelineMLX


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate video from one image using FlashHead (MLX)"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default=None,
        help="Path to FlashHead model checkpoint directory.")
    parser.add_argument(
        "--wav2vec_dir", type=str, default=None,
        help="Path to the wav2vec checkpoint directory.")
    parser.add_argument(
        "--model_type", type=str, default="lite",
        choices=["pro", "lite"],
        help="Model type (pro or lite). Only 'lite' is supported in MLX.")
    parser.add_argument(
        "--save_file", type=str, default=None,
        help="Output video file path.")
    parser.add_argument(
        "--base_seed", type=int, default=42,
        help="Random seed.")
    parser.add_argument(
        "--cond_image", type=str, default=None,
        help="Condition image path.")
    parser.add_argument(
        "--cond_image_dir", type=str, default=None,
        help="Directory of condition images.")
    parser.add_argument(
        "--audio_path", type=str, default=None,
        help="Audio file path (WAV, MP3, etc).")
    parser.add_argument(
        "--sample_steps", type=int, default=4,
        help="Number of diffusion sampling steps (2, 4, 8, ...).")
    parser.add_argument(
        "--sample_shift", type=float, default=5.0,
        help="Timestep shift for flow matching.")
    parser.add_argument(
        "--color_correction_strength", type=float, default=1.0,
        help="Color correction strength (0.0 = off, 1.0 = full).")
    parser.add_argument(
        "--audio_encode_mode", type=str, default="once",
        choices=["stream", "once"],
        help="Audio encoding mode.")
    parser.add_argument(
        "--use_face_crop", action="store_true", default=False,
        help="Enable face detection and crop.")
    parser.add_argument(
        "--height", type=int, default=512, help="Output height.")
    parser.add_argument(
        "--width", type=int, default=512, help="Output width.")
    parser.add_argument(
        "--frame_num", type=int, default=33,
        help="Frames per generation chunk.")
    parser.add_argument(
        "--motion_frames_latent_num", type=int, default=2,
        help="Number of latent motion frames (controls overlap between chunks).")
    parser.add_argument(
        "--cached_audio_duration", type=int, default=8,
        help="Cached audio duration in seconds (streaming mode).")

    args = parser.parse_args()

    # Validate
    assert args.ckpt_dir is not None, "Please specify --ckpt_dir"
    assert args.wav2vec_dir is not None, "Please specify --wav2vec_dir"
    assert args.model_type == "lite", "Only 'lite' model is supported in MLX mode"
    assert args.cond_image or args.cond_image_dir, "Please specify --cond_image or --cond_image_dir"
    assert args.audio_path is not None, "Please specify --audio_path"

    return args


def save_video(frames_list, video_path, audio_path, fps):
    """Save frames to MP4 video with audio."""
    temp_video_path = video_path.replace('.mp4', '_tmp.mp4')
    with imageio.get_writer(
        temp_video_path, format='mp4', mode='I',
        fps=fps, codec='h264', ffmpeg_params=['-bf', '0']
    ) as writer:
        for frames in frames_list:
            if hasattr(frames, 'numpy'):
                frames = frames.numpy()
            frames = frames.astype(np.uint8)
            for i in range(frames.shape[0]):
                writer.append_data(frames[i])

    # Merge video + audio
    cmd = [
        'ffmpeg', '-i', temp_video_path, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-shortest', video_path, '-y'
    ]
    subprocess.run(cmd, capture_output=True)
    os.remove(temp_video_path)


def generate(args):
    # ---- Init pipeline ----
    cond_path = args.cond_image_dir if args.cond_image_dir else args.cond_image

    pipeline = FlashHeadPipelineMLX(
        checkpoint_dir=args.ckpt_dir,
        model_type=args.model_type,
        wav2vec_dir=args.wav2vec_dir,
    )

    # ---- Compute derived params ----
    target_size = (args.height, args.width)
    tgt_fps = 25
    sample_rate = 16000

    vae_temporal_stride = 8  # LTX VAE
    motion_frames_num = (args.motion_frames_latent_num - 1) * vae_temporal_stride + 1
    slice_len = args.frame_num - motion_frames_num

    # ---- Prepare reference image + latents ----
    pipeline.prepare_params(
        cond_image_path_or_dir=cond_path,
        target_size=target_size,
        frame_num=args.frame_num,
        motion_frames_num=motion_frames_num,
        sampling_steps=args.sample_steps,
        seed=args.base_seed,
        shift=args.sample_shift,
        color_correction_strength=args.color_correction_strength,
    )

    # ---- Load audio ----
    human_speech_array_all, _ = librosa.load(
        args.audio_path, sr=sample_rate, mono=True
    )
    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    human_speech_array_frame_num = args.frame_num * sample_rate // tgt_fps

    logger.info("Data preparation done. Starting video generation...")

    generated_list = []

    if args.audio_encode_mode == 'once':
        # Pad audio to avoid truncating the last chunk
        remainder = (
            len(human_speech_array_all) - human_speech_array_frame_num
        ) % human_speech_array_slice_len
        if remainder > 0:
            pad_length = human_speech_array_slice_len - remainder
            human_speech_array_all = np.concatenate([
                human_speech_array_all,
                np.zeros(pad_length, dtype=human_speech_array_all.dtype),
            ])

        # Encode full audio
        logger.info("Encoding full audio (once mode)...")
        audio_embedding_all = pipeline.preprocess_audio(
            human_speech_array_all, sr=sample_rate, fps=tgt_fps
        )
        total_audio_frames = audio_embedding_all.shape[1]
        num_chunks = (total_audio_frames - args.frame_num) // slice_len + 1

        logger.info(f"Total audio frames: {total_audio_frames}, chunks: {num_chunks}")

        for chunk_idx in range(num_chunks):
            t0 = time.time()

            start = chunk_idx * slice_len
            end = start + args.frame_num
            audio_chunk = audio_embedding_all[:, start:end, :, :, :]

            # Generate
            import mlx.core as mx
            audio_chunk_mlx = mx.array(np.array(audio_chunk))
            video = pipeline.generate(audio_chunk_mlx)

            # Convert to numpy: (3, T, H, W) → (T, H, W, 3)
            video_np = np.array(video)
            video_np = video_np.transpose(1, 2, 3, 0)  # (T, 512, 512, 3)
            video_np = ((video_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

            if chunk_idx != 0:
                video_np = video_np[motion_frames_num:]

            t_elapsed = time.time() - t0
            logger.info(
                f"Chunk {chunk_idx+1}/{num_chunks}: "
                f"{video_np.shape[0]} frames, {t_elapsed:.1f}s"
            )

            generated_list.append(video_np)

    elif args.audio_encode_mode == 'stream':
        cached_audio_length_sum = sample_rate * args.cached_audio_duration
        audio_end_idx = args.cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - args.frame_num

        from collections import deque
        audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

        # Pad audio
        remainder = len(human_speech_array_all) % human_speech_array_slice_len
        if remainder > 0:
            pad_length = human_speech_array_slice_len - remainder
            human_speech_array_all = np.concatenate([
                human_speech_array_all,
                np.zeros(pad_length, dtype=human_speech_array_all.dtype),
            ])

        human_speech_array_slices = human_speech_array_all.reshape(
            -1, human_speech_array_slice_len
        )
        num_chunks = len(human_speech_array_slices)

        for chunk_idx, speech_slice in enumerate(human_speech_array_slices):
            t0 = time.time()

            audio_dq.extend(speech_slice.tolist())
            audio_array = np.array(audio_dq)

            # Encode streaming chunk
            audio_embedding = pipeline.preprocess_audio(
                audio_array, sr=sample_rate, fps=tgt_fps
            )
            # Slice to relevant range
            audio_chunk = audio_embedding[:, audio_start_idx:audio_end_idx, :, :, :]

            import mlx.core as mx
            audio_chunk_mlx = mx.array(np.array(audio_chunk))
            video = pipeline.generate(audio_chunk_mlx)

            video_np = np.array(video)
            video_np = video_np.transpose(1, 2, 3, 0)
            video_np = ((video_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            video_np = video_np[motion_frames_num:]

            t_elapsed = time.time() - t0
            logger.info(
                f"Chunk {chunk_idx+1}/{num_chunks}: "
                f"{video_np.shape[0]} frames, {t_elapsed:.1f}s"
            )

            generated_list.append(video_np)

    # ---- Save video ----
    if args.save_file is None:
        output_dir = 'sample_results'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        args.save_file = os.path.join(output_dir, f"res_{timestamp}.mp4")

    save_video(generated_list, args.save_file, args.audio_path, fps=tgt_fps)
    logger.info(f"Saved video to {args.save_file}")
    logger.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
