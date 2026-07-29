# LTX VAE：从 PyTorch 到 MLX 的移植指南

## 背景

SoulX-FlashHead 使用的 diffusers `AutoencoderKLLTXVideo` / `CausalVideoAutoencoder` 是一个基于 3D 卷积的视频自编码器。原始实现依赖 PyTorch + diffusers，在 Apple Silicon 上只能在 CPU 运行（MPS Conv3d 支持不完整），decode 一段 33 帧视频约需 24 秒。

移植到 MLX 后，全程在 Apple GPU 运行，相同操作用时约 5.3 秒，约 **5 倍加速**，且精度损失可忽略（端到端 max diff ≈ 0.019，约 2.5/255 像素值）。

---

## 1. 核心差异：内存布局

这是移植中最根本的区别，影响所有算子的实现：

| 维度 | PyTorch | MLX |
|------|---------|-----|
| 数据排布 | NCDHW `(B, C, T, H, W)` | NHWC `(B, T, H, W, C)` |
| Conv3d 权重 | `(C_out, C_in, kT, kH, kW)` | `(C_out, kT, kH, kW, C_in)` |
| Conv 边界处理 | 内建 `padding` + `padding_mode` | 需手动填充后传 `padding=0` |

**转换函数：**

```python
# 数据：NCDHW ↔ NHWC
def ncdhw_to_nhwc(x):
    return mx.transpose(x, (0, 2, 3, 4, 1))  # (N,C,T,H,W) → (N,T,H,W,C)

# 权重：PT Conv3d → MLX Conv3d
def pt_to_mlx_conv3d(pt_weight):
    return mx.transpose(pt_weight, (0, 2, 3, 4, 1))  # (out,in,kT,kH,kW) → (out,kT,kH,kW,in)
```

**关键原则：每写完一个算子，立即用相同输入对比 PT/MLX 的 max diff，确认一致后再继续。**

---

## 2. 算子实现要点

### 2.1 CausalConv3d

PT 的 `CausalConv3d` 时间维度和空间维度的填充策略不同：

- **时间维度（始终 replicate）：**
  - `causal=True`：复制第一帧 `kT-1` 次，prepend（纯历史信息，无未来泄露）
  - `causal=False`：复制第一帧和最后一帧各 `kT//2` 次

- **空间维度：** 根据 `spatial_padding_mode` 决定 zeros 或 replicate

- **MLX 做法：** 手动做完所有填充 → 传 `padding=0` 的 nn.Conv3d

```python
class CausalConv3d(nn.Module):
    def _pad(self, x):
        # 时间：始终 replicate
        if self.causal:
            pad_t_l, pad_t_r = kT - 1, 0
        else:
            pad_t_l = pad_t_r = kT // 2
        if pad_t_l > 0:
            x = mx.concatenate([mx.repeat(x[:, :1], pad_t_l, axis=1), x], axis=1)
        if pad_t_r > 0:
            x = mx.concatenate([x, mx.repeat(x[:, -1:], pad_t_r, axis=1)], axis=1)

        # 空间：zeros 或 replicate
        if self.spatial_padding_mode == "replicate":
            # replicate 边缘像素
        else:
            x = mx.pad(x, [(0,0),(0,0),(pad_h,pad_h),(pad_w,pad_w),(0,0)])
        return x

    def __call__(self, x):
        return self.conv(self._pad(x))  # conv 用 padding=0
```

### 2.2 PlainConv3d（ResNet Shortcut）

ResNet 的 `conv_shortcut` 在 PT 中使用 `make_linear_nd`（即 1×1×1 Conv3d），**不做任何填充，忽略 causal 设置**。不能用 CausalConv3d（因为 CausalConv3d 会自动加时间填充），需要单独实现：

```python
class PlainConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        return self.conv(x)  # 不做填充
```

### 2.3 ResnetBlock3D

PT 的执行顺序：`PixelNorm → SiLU → CausalConv → PixelNorm → SiLU → CausalConv`，然后与 shortcut 相加。Shortcut 只在 `in_ch != out_ch` 时存在，且使用 `LayerNorm + PlainConv3d`：

```python
class ResnetBlock3D(nn.Module):
    def __call__(self, x):
        h = self.act(self.norm1(x))
        h = self.conv1(h)        # CausalConv3d
        h = self.act(self.norm2(h))
        h = self.conv2(h)        # CausalConv3d

        if self.use_shortcut:
            x = self.conv_shortcut(self.norm3(x))  # LayerNorm + PlainConv3d
        return x + h
```

### 2.4 DepthToSpaceUpsample（最复杂的算子）

这是调试中最容易出错的算子。PT 做三件事：

1. **CausalConv3d** — `(in_ch → in_ch×8)`，causal 根据调用者传入（decoder 传 `False`）
2. **Pixel Shuffle** — 把 8 倍通道展开为 2×2×2 的时空上采样
3. **去掉第一帧** — 补偿时间轴上采样的边界效果

**关键陷阱 —— 通道重排顺序：**

PT 的 rearrange 把 4096 个通道视为 `(C=512, pT=2, pH=2, pW=2)`，**C 变化最慢，pW 变化最快**：

```
PT: rearrange(x, "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)")
    → 通道 0 = (c=0, pT=0, pH=0, pW=0)
    → 通道 1 = (c=0, pT=0, pH=0, pW=1)
    → ...
    → 通道 8 = (c=1, pT=0, pH=0, pW=0)
```

MLX NHWC 中的等效 reshape ：**C 必须放在最慢变化的位置**：

```python
# ✅ 正确：C 在最前 → C 最慢，pW 在最末 → pW 最快
x = mx.reshape(x, (N, T, H, W, C, 2, 2, 2))   # ( ... C, pT, pH, pW)
x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))  # → (N, T, pT, H, pH, W, pW, C)
x = mx.reshape(x, (N, T*2, H*2, W*2, C))        # 展平 (T,pT)→T*2 等
x = x[:, 1:, :, :, :]                            # 去掉第一帧

# ❌ 错误：C 在最后 → C 最快，与 PT 的通道映射完全错乱
x = mx.reshape(x, (N, T, H, W, 2, 2, 2, C))    # ( ... pT, pH, pW, C) ← C 最快！
```

### 2.5 Patchify / Unpatchify

空间 4×4 的 patch embedding，时间维度不变。PT 的通道排序规则：

```
PT: new_c = c * 16 + w_patch * 4 + h_patch
    → C 最慢，H_patch 最快
```

对应的 NHWC 实现：

```python
def patchify_nhwc(x, patch_size=4):
    N, T, H, W, C = x.shape; p = patch_size
    x = mx.reshape(x, (N, T, H//p, p, W//p, p, C))
    x = mx.transpose(x, (0, 1, 2, 4, 6, 5, 3))  # → (N,T,H/p,W/p, C, pW, pH)
    x = mx.reshape(x, (N, T, H//p, W//p, p*p*C))
    return x
```

---

## 3. 架构映射：从 safetensors key 反推 block 结构

diffusers 的 safetensors 文件使用 `encoder.down_blocks.X.resnets.Y.xxx` 这样的 key 命名。通过解析 key 可以反推出精确的 block 结构：

### Encoder

```
Patchify(4×4, spatial only) → 48ch
conv_in(48→128)
  Block 0: res_x×4(128) → Downsample(2,2,2) → res_x_y(128→256)
  Block 1: res_x×3(256) → Downsample(2,2,2) → res_x_y(256→512)
  Block 2: res_x×3(512) → Downsample(2,2,2)
  Block 3: res_x×3(512)
  Mid:     res_x×4(512)
norm → SiLU → conv_out(512→129) → keep[:128]
```

### Decoder

**注意顺序：** PT 是先 upsample 再做 resnet，不是先 resnet 再 upsample。

```
conv_in(128→512)
  Mid:     res_x×4(512)
  Up 0:    res_x×3(512)
  Up 1:    Upsample → res_x×3(512)
  Up 2:    res_x_y(512→256) → Upsample → res_x×3(256)
  Up 3:    res_x_y(256→128) → Upsample → res_x×4(128)
norm → SiLU → conv_out(128→48) → Unpatchify → 3ch RGB
```

---

## 4. 权重加载：key-path 解析器

从 safetensors key 映射到 MLX 模块属性。例如：

```
encoder.down_blocks.0.resnets.1.conv1.conv.weight
   → model.encoder.down_0_res[1].conv1.conv.weight

decoder.up_blocks.1.upsamplers.0.conv.conv.weight
   → model.decoder.up_1_us.conv.conv.weight

decoder.up_blocks.2.conv_in.norm3.bias
   → model.decoder.up_2_xy.norm3.bias
```

实现方式：把 key 按 `.` 拆分，逐段导航到对应的 MLX 属性。遇到 `5D + "weight"` 的 tensor 自动做 NHWC 转换。

---

## 5. 隐变量统计（Latent Stats）

VAE 的 encode 输出需要做 per-channel 标准化：

```python
normalized = (latent - mean_of_means) / std_of_means
```

- `latents_mean` / `latents_std` 存储在 safetensors 中，shape `(128,)`
- PT 的 `mean_of_means = latents_mean`（直接使用，不是 mean of means）
- decode 时需要反标准化：`z * std + mean`

---

## 6. 调试方法：逐层对比法

移植 VAE 最有效的调试策略：

```
1. 用相同的随机输入 + 相同的权重
2. 在 PT 和 MLX 上分别 forward
3. 逐层计算 max(|MLX - PT|)
4. 找到第一个出现大偏差的层（max > 1e-3）
5. 对比该层的源码实现，找出差异
6. 修复 → 重新验证 → 继续
```

**实际遇到的 5 个 bug：**

| # | 症状 | 根因 | 所在层 |
|---|------|------|--------|
| 1 | 全屏闪烁方块 | Patchify 通道排序 `(C,pW,pH)` ∵ `(C,pH,pW)` | conv_in 之前 |
| 2 | Decoder 全错 | resnet 和 upsample 顺序颠倒 | up_1_res |
| 3 | ResNet 输出偏差 | Shortcut 用了 CausalConv3d 而非 PlainConv | conv_shortcut |
| 4 | Upsample 后全错 | DepthToSpace reshape 把 C 放在最快维度 | up_1_us |
| 5 | Upsample 仍大偏差 | CausalConv3d 硬编码 `causal=True`，decoder 应传 `False` | up_1_us |

---

## 7. Pipeline 集成

将 `vae_encode` / `vae_decode` 从 PT CPU 切换到 MLX GPU：

```python
# 旧：需要 MLX → numpy → PT tensor → CPU 计算 → numpy → MLX
def vae_encode(video_mlx, ckpt_dir):
    vae = _get_torch_vae(ckpt_dir)         # 加载 PT VAE
    video_pt = torch.from_numpy(np.array(video_mlx))  # MLX → numpy → PT
    with torch.no_grad():
        latent_pt = vae.encode(video_pt)    # CPU 计算 (~10s)
    return mx.array(latent_pt.numpy())      # PT → numpy → MLX

# 新：全程 MLX，无拷贝
def vae_encode(video_mlx, ckpt_dir):
    vae = _get_mlx_vae(ckpt_dir)           # 加载 MLX VAE（单例）
    return vae.encode(video_mlx)            # GPU 计算 (~4.7s)
```

---

## 8. 最终效果

| 指标 | PT VAE (CPU) | MLX VAE (Apple GPU) |
|------|-------------|---------------------|
| Encode | ~10s | ~4.7s |
| Decode | ~24s | ~5.3s |
| 端到端 max diff | — | 0.019（约 2.5/255） |

整个 pipeline 现在是纯 MLX 原生运行（VAE + Diffusion），Wav2Vec2 音频编码仍保留 PyTorch（轻量、非瓶颈）。

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `flash_head_mlx/ltx_vae/ops.py` | 基础算子：CausalConv3d、ResnetBlock3D、DepthToSpaceUpsample、patchify 等 |
| `flash_head_mlx/ltx_vae/vae.py` | Encoder/Decoder 架构 + 权重加载器 + LtxVAE 公共 API |
| `flash_head_mlx/pipeline.py` | FlashHeadPipelineMLX 集成，调用 MLX VAE |
| `generate_video_mlx.py` | MLX 原生视频生成入口脚本 |
