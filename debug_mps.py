"""
诊断 MPS 上画面模糊的根因
用法: python debug_mps.py --ckpt_dir /path/to/checkpoint
"""
import torch
import argparse
import os
import sys

def test_vae_roundtrip(ckpt_dir, model_type):
    """测试 VAE 编码→解码回环是否正确（支持 Pro 和 Lite）"""
    print("=" * 60)
    print(f"测试 1: VAE encode → decode 回环 (model_type={model_type})")
    print("=" * 60)

    if model_type == "lite":
        vae_dir = os.path.join(ckpt_dir, "VAE_LTX")
        if not os.path.exists(vae_dir):
            print(f"Lite VAE 目录不存在: {vae_dir}")
            return False

        from flash_head.ltx_video.ltx_vae import LtxVAE
        print("加载 LtxVAE...")
        vae = LtxVAE(
            pretrained_model_type_or_path=vae_dir,
            dtype=torch.float32,
            device="mps",
        )
        # LtxVAE 输入: [1, 3, T, H, W], 输出: [1, 3, T, H, W]
        # encode: 1,3,33,512,512 -> latent: 128,5,16,16
        # decode: 128,5,16,16 -> 1,3,33,512,512
        frames = 33
        h, w = 512, 512

    else:
        vae_path = os.path.join(ckpt_dir, "VAE_Wan/Wan2.1_VAE.pth")
        if not os.path.exists(vae_path):
            print(f"WanVAE checkpoint 不存在: {vae_path}")
            return False

        from flash_head.wan.modules.vae import WanVAE
        print("加载 WanVAE...")
        vae = WanVAE(
            vae_path=vae_path,
            dtype=torch.float32,
            device="mps",
            parallel=False,
        )
        vae.model.eval()
        frames = 33
        h, w = 512, 512

    # 创建测试图像
    print("创建测试图像...")
    test_img = torch.randn(1, 3, frames, h, w, device="mps", dtype=torch.float32) * 0.5
    test_img = test_img.clamp(-1, 1)
    print(f"输入图像 shape: {test_img.shape}, range: [{test_img.min():.3f}, {test_img.max():.3f}]")

    # Encode
    print("Encoding...")
    torch.mps.synchronize()
    with torch.no_grad():
        latent = vae.encode(test_img)
    torch.mps.synchronize()
    print(f"Latent shape: {latent.shape}, mean: {latent.mean():.4f}, std: {latent.std():.4f}")

    # 检查 latent
    if torch.isnan(latent).any():
        print("❌ Encode 失败: latent 含有 NaN!")
        return False
    if torch.isinf(latent).any():
        print("❌ Encode 失败: latent 含有 Inf!")
        return False
    if latent.std() < 1e-6:
        print("❌ Encode 失败: latent std 接近 0")
        return False

    # Decode
    print("Decoding...")
    torch.mps.synchronize()
    with torch.no_grad():
        decoded = vae.decode(latent)
    torch.mps.synchronize()

    if model_type == "lite":
        decoded = decoded.squeeze(0)  # LtxVAE 返回 [1, 3, T, H, W]
    else:
        decoded = decoded.squeeze(0)  # WanVAE 也返回 [1, C, T, H, W]

    print(f"解码图像 shape: {decoded.shape}, range: [{decoded.min():.3f}, {decoded.max():.3f}]")

    if torch.isnan(decoded).any():
        print("❌ Decode 失败: 输出含有 NaN!")
        return False

    # 检查是否模糊
    frame_diff = decoded[:, 1:, :, :] - decoded[:, :-1, :, :]
    diff_std = frame_diff.std()
    print(f"帧间差异 std: {diff_std:.6f}")

    from torch.nn.functional import conv2d
    laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device="mps")
    laplacian = laplacian.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    mid_frame = decoded[:, decoded.shape[1]//2, :, :].unsqueeze(1)
    edge_energy = 0
    for c in range(3):
        edge = conv2d(mid_frame[:, c:c+1, :, :], laplacian[:1, :1, :, :], padding=1)
        edge_energy += edge.var().item()
    print(f"边缘能量 (Laplacian variance): {edge_energy:.6f}")

    if diff_std < 0.01:
        print("❌ VAE DECODE 有问题: 帧间差异极小")
        return False
    if edge_energy < 0.001:
        print("❌ VAE DECODE 有问题: 几乎无边缘/高频信息")
        return False

    print("✅ VAE encode→decode 回环正常")
    return True


def test_model_freqs(ckpt_dir):
    """检查模型 RoPE freqs 的 dtype"""
    print("\n" + "=" * 60)
    print("测试 2: 检查模型 RoPE 频率 precision")
    print("=" * 60)

    model_dir = os.path.join(ckpt_dir, "Model_Pro")
    if not os.path.exists(model_dir):
        model_dir = os.path.join(ckpt_dir, "teacher")
    if not os.path.exists(model_dir):
        print(f"Model checkpoint 不存在: {model_dir}")
        return

    from flash_head.src.modules.flash_head_model import WanModelAudioProject

    print("加载模型...")
    model = WanModelAudioProject.from_pretrained(model_dir)

    # 检查 freqs buffer
    if hasattr(model, 'freqs'):
        freqs = model.freqs
        print(f"freqs dtype: {freqs.dtype}")
        print(f"freqs device: {freqs.device}")
        print(f"freqs shape: {freqs.shape}")

        if freqs.dtype == torch.complex128:
            print("❌ freqs 是 complex128 (float64-based)，MPS 不支持!")
            print("   这会导致 RoPE 位置编码计算错误")
        elif freqs.dtype == torch.complex64:
            print("✅ freqs 是 complex64，MPS 兼容")
        else:
            print(f"⚠️ freqs 是意外的 dtype: {freqs.dtype}")
    else:
        print("⚠️ 模型没有 freqs buffer (可能在其他名字下)")


def test_precompute_freqs():
    """检查 precompute_freqs_cis 的输出 dtype"""
    print("\n" + "=" * 60)
    print("测试 3: precompute_freqs_cis 输出 dtype")
    print("=" * 60)

    from flash_head.src.modules.flash_head_model import precompute_freqs_cis_3d

    freqs = precompute_freqs_cis_3d(dim=128, end=1024)
    print(f"precompute_freqs_cis_3d 输出 dtype: {freqs.dtype}")

    if freqs.dtype == torch.complex128:
        print("❌ precompute_freqs_cis 使用了 .double() 创建 complex128!")
        print("   需要改为 .float() 避免 MPS 不兼容")
    elif freqs.dtype == torch.complex64:
        print("✅ complex64，MPS 兼容")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, help="FlashHead checkpoint 目录")
    parser.add_argument("--model_type", type=str, default="pro", choices=["pro", "lite", "pretrained"],
                        help="模型类型: pro, lite, pretrained")
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # 测试 1: VAE 回环
    vae_ok = test_vae_roundtrip(args.ckpt_dir, args.model_type)

    # 测试 2: 模型 freqs (Lite 模型结构不同，跳过)
    if args.model_type != "lite":
        test_model_freqs(args.ckpt_dir)
    else:
        print("\n跳过测试 2: Lite 模型结构不同，不检查 WanModelAudioProject freqs")

    # 测试 3: precompute 函数
    test_precompute_freqs()

    # 总结
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)

    if not vae_ok:
        print("→ VAE decode 有问题，问题在 VAE 层")
        print("  可能原因: MPS Conv3d 或 causal padding 与 CUDA 行为不一致")
    else:
        print("→ VAE 正常，问题在扩散模型")
        print("  可能原因:")
        print("  1. RoPE 频率精度问题 (complex128 → complex64)")
        print("  2. sinusoidal_embedding float64 → float32 精度损失")
        print("  3. 扩散模型其他 MPS 不兼容算子")

if __name__ == "__main__":
    main()
