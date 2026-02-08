# inference.py 逐段解析

本文档逐段解释 `inference.py` 文件的每个部分。

---

## 第一部分：导入依赖 (L1-20)

```python
import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
```

### 解释

| 模块 | 作用 |
|------|------|
| `argparse` | 解析命令行参数 |
| `torch` | PyTorch 深度学习框架 |
| `OmegaConf` | 加载 YAML 配置文件 |
| `tqdm` | 显示进度条 |
| `transforms` | 图像预处理（用于 I2V） |
| `write_video` | 将 tensor 保存为 MP4 视频 |
| `einops.rearrange` | 张量维度重排 |
| `torch.distributed` | 多 GPU 分布式推理支持 |
| `CausalInferencePipeline` | **核心**：少步因果推理管线 |
| `CausalDiffusionInferencePipeline` | 多步扩散推理管线 |
| `TextDataset` | 加载文本提示数据集 |
| `TextImagePairDataset` | 加载图文对数据集（I2V） |
| `DynamicSwapInstaller` | 低显存时动态换入换出模型 |

---

## 第二部分：命令行参数定义 (L22-36)

```python
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="Number of overlap frames between sliding windows")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--save_with_index", action="store_true",
                    help="Whether to save the video using the index or prompt as the filename")
args = parser.parse_args()
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config_path` | str | 必填 | 配置文件路径（如 `configs/self_forcing_dmd.yaml`） |
| `--checkpoint_path` | str | 可选 | 模型权重路径 |
| `--data_path` | str | 必填 | 输入提示文件路径 |
| `--extended_prompt_path` | str | 可选 | 扩展提示文件路径 |
| `--output_folder` | str | 必填 | 输出视频目录 |
| `--num_output_frames` | int | 21 | 生成的潜在空间帧数 |
| `--i2v` | flag | False | 是否使用图生视频模式 |
| `--use_ema` | flag | False | 是否使用 EMA 权重 |
| `--seed` | int | 0 | 随机种子 |
| `--num_samples` | int | 1 | 每个提示生成几个视频 |
| `--save_with_index` | flag | False | 文件名用索引还是提示文本 |

---

## 第三部分：分布式初始化 (L38-50)

```python
# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)
```

### 解释
- **分布式模式**：通过 `torchrun` 启动时会设置 `LOCAL_RANK` 环境变量
- **单卡模式**：直接使用 `cuda` 设备
- **随机种子**：分布式时每个 GPU 使用不同种子（`seed + local_rank`），确保生成不同视频

---

## 第四部分：显存检测与梯度禁用 (L52-55)

```python
print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)
```

### 解释
- **显存检测**：若可用显存 < 40GB，启用低显存模式
- **禁用梯度**：推理时不需要计算梯度，节省显存

---

## 第五部分：加载配置文件 (L57-59)

```python
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
```

### 解释
1. 加载用户指定的配置文件（如 `self_forcing_dmd.yaml`）
2. 加载默认配置
3. 合并配置（用户配置覆盖默认配置）

---

## 第六部分：初始化推理管线 (L61-67)

```python
# Initialize pipeline
if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    pipeline = CausalDiffusionInferencePipeline(config, device=device)
```

### 解释
- **有 `denoising_step_list`**：使用少步推理（Self-Forcing 蒸馏模型，如 4 步）
- **无 `denoising_step_list`**：使用标准多步扩散推理（如 50 步）

管线内部会初始化三个模型：
1. `generator`：扩散生成器（CausalWanModel）
2. `text_encoder`：文本编码器（UMT5-XXL）
3. `vae`：变分自编码器

---

## 第七部分：加载模型权重 (L69-79)

```python
if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)
```

### 解释
1. **加载权重**：从 checkpoint 加载 generator 权重（可选 EMA 版本）
2. **精度转换**：转为 bfloat16 减少显存占用
3. **显存管理**：
   - 低显存模式：text_encoder 动态换入换出
   - 正常模式：所有模型常驻 GPU

---

## 第八部分：创建数据集 (L82-100)

```python
# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)
```

### 解释
- **I2V 模式**：加载图文对，图像 resize 到 480×832
- **T2V 模式**：只加载文本提示
- **分布式采样**：多 GPU 时自动分配数据

---

## 第九部分：创建输出目录 (L102-107)

```python
# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()
```

### 解释
- 只在主进程创建目录，避免竞争条件
- `barrier()` 同步所有进程，确保目录创建完成后再继续

---

## 第十部分：辅助函数 (L110-120)

```python
def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]
    output = torch.stack(output, dim=0)
    return output
```

### 解释
这是一个 VAE 编码辅助函数（当前代码中未被使用，可能是遗留代码）。

---

## 第十一部分：主推理循环 (L123-193) — 逐行详解

### L123: 主循环开始

```python
for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
```

| 元素 | 说明 |
|------|------|
| `i` | 循环计数器（从 0 开始） |
| `batch_data` | 从 DataLoader 获取的一批数据（dict 格式） |
| `tqdm` | 进度条显示 |
| `disable=(local_rank != 0)` | 只在主进程（rank=0）显示进度条，避免多 GPU 时重复输出 |

---

### L124: 获取样本索引

```python
idx = batch_data['idx'].item()
```

- `batch_data['idx']` 是一个只有 1 个元素的 tensor
- `.item()` 将其转换为 Python int
- `idx` 用于后续判断是否是有效样本（而非填充的 dummy 数据）

---

### L126-131: 解包批次数据

```python
# For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
# Unpack the batch data for convenience
if isinstance(batch_data, dict):
    batch = batch_data
elif isinstance(batch_data, list):
    batch = batch_data[0]  # First (and only) item in the batch
```

由于 `batch_size=1`，数据已经是单个样本，这里只是兼容不同的数据格式。

---

### L133-134: 初始化存储变量

```python
all_video = []
num_generated_frames = 0  # Number of generated (latent) frames
```

- `all_video` 存储生成的视频片段（用于长视频拼接，当前代码未使用多段）
- `num_generated_frames` 统计生成的潜在帧数

---

### L136-150: 图生视频 (I2V) 分支

```python
if args.i2v:
    # For image-to-video, batch contains image and caption
    prompt = batch['prompts'][0]  # Get caption from batch
    prompts = [prompt] * args.num_samples
```

- 提取文本描述
- 复制 `num_samples` 份（用于生成多个样本）

```python
    # Process the image
    image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
```

**维度变换详解**：
```
batch['image']          : [1, 3, 480, 832]        # DataLoader 输出
  .squeeze(0)           : [3, 480, 832]           # 移除 batch 维度
  .unsqueeze(0)         : [1, 3, 480, 832]        # 重新添加 batch 维度
  .unsqueeze(2)         : [1, 3, 1, 480, 832]     # 添加时间维度 (单帧)
  .to(...)              : [1, 3, 1, 480, 832]     # 转移到 GPU + bfloat16
```

```python
    # Encode the input image as the first latent
    initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
    initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)
```

**VAE 编码**:
```
image                   : [1, 3, 1, 480, 832]     # 像素空间
  → VAE.encode          : [1, 1, 16, 60, 104]     # 潜在空间 (压缩 8x8, 通道 3→16)
  .repeat(num_samples)  : [B, 1, 16, 60, 104]     # 复制 B 份
```

```python
    sampled_noise = torch.randn(
        [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
    )
```

**噪声形状**: `[B, 20, 16, 60, 104]`
- I2V 模式下，第一帧是输入图像，只需生成后续 20 帧
- 21 - 1 = 20 帧噪声

---

### L151-163: 文生视频 (T2V) 分支

```python
else:
    # For text-to-video, batch is just the text prompt
    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * args.num_samples
    else:
        prompts = [prompt] * args.num_samples
    initial_latent = None
```

- 优先使用扩展提示（extended_prompt，通常由 GPT-4 等扩展生成，更详细）
- 如果没有扩展提示，使用原始提示
- T2V 无初始帧，`initial_latent = None`

```python
    sampled_noise = torch.randn(
        [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )
```

**噪声形状**: `[B, 21, 16, 60, 104]`
- T2V 需要生成全部 21 帧
- 各维度含义：
  - `B`: batch size（`num_samples`）
  - `21`: 潜在帧数
  - `16`: 潜在通道数
  - `60`: 潜在高度（480 ÷ 8）
  - `104`: 潜在宽度（832 ÷ 8）

---

### L165-172: 核心推理调用

```python
# Generate 81 frames
video, latents = pipeline.inference(
    noise=sampled_noise,
    text_prompts=prompts,
    return_latents=True,
    initial_latent=initial_latent,
    low_memory=low_memory,
)
```

**参数说明**：
| 参数 | 类型 | 形状/值 | 说明 |
|------|------|---------|------|
| `noise` | Tensor | `[B, 21, 16, 60, 104]` | 初始高斯噪声 |
| `text_prompts` | List[str] | 长度 B | 文本提示列表 |
| `return_latents` | bool | True | 是否返回潜在空间结果 |
| `initial_latent` | Tensor/None | `[B, 1, 16, 60, 104]` or None | I2V 的第一帧 |
| `low_memory` | bool | True/False | 是否启用低显存模式 |

**返回值**：
| 返回值 | 形状 | 说明 |
|--------|------|------|
| `video` | `[B, 81, 3, 480, 832]` | 像素空间视频（归一化到 [0,1]） |
| `latents` | `[B, 21, 16, 60, 104]` | 潜在空间表示 |

> **帧数关系**: 21 帧潜在空间 × 4（VAE 时间上采样）= 84 帧，但边界处理后输出 81 帧

---

### L173-175: 后处理

```python
current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
all_video.append(current_video)
num_generated_frames += latents.shape[1]
```

**维度变换**：
```
video                   : [B, 81, 3, 480, 832]    # PyTorch 格式: C 在前
  → rearrange           : [B, 81, 480, 832, 3]    # OpenCV/视频格式: C 在后
  .cpu()                : 转移到 CPU 内存
```

---

### L177-178: 最终视频准备

```python
# Final output video
video = 255.0 * torch.cat(all_video, dim=1)
```

- `torch.cat(all_video, dim=1)`: 沿时间维度拼接（当前只有一段）
- `* 255.0`: 从 [0, 1] 缩放到 [0, 255]（视频存储需要 uint8）

---

### L180-181: 清理缓存

```python
# Clear VAE cache
pipeline.vae.model.clear_cache()
```

VAE 解码器使用因果卷积，会缓存之前帧的特征。每个样本处理完后清理，避免内存泄漏。

---

### L183-192: 保存视频

```python
# Save the video if the current prompt is not a dummy prompt
if idx < num_prompts:
```

分布式训练时可能有填充的 dummy 数据，只保存有效样本。

```python
    model = "regular" if not args.use_ema else "ema"
    for seed_idx in range(args.num_samples):
        # All processes save their videos
        if args.save_with_index:
            output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
        else:
            output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
        write_video(output_path, video[seed_idx], fps=16)
```

**文件命名**：
| 模式 | 示例文件名 |
|------|-----------|
| `--save_with_index` | `0-0_ema.mp4`, `0-1_ema.mp4` |
| 默认（提示文本） | `A cat playing with...-0.mp4` |

**`write_video` 参数**：
- `output_path`: 输出路径
- `video[seed_idx]`: 形状 `[81, 480, 832, 3]`，值范围 [0, 255]
- `fps=16`: 帧率 16 FPS，81 帧 ≈ **5 秒视频**

---

## 张量形状完整追踪

```
输入提示: "A cat playing with a ball"
     ↓
文本编码器 (UMT5-XXL)
     ↓
prompt_embeds: [1, 512, 4096]
     ↓
随机噪声: [1, 21, 16, 60, 104]
     ↓
扩散生成器 (CausalWanModel) × 4 步去噪
     ↓
潜在空间: [1, 21, 16, 60, 104]
     ↓
VAE 解码器
     ↓
像素空间: [1, 81, 3, 480, 832]
     ↓
rearrange + × 255
     ↓
视频: [81, 480, 832, 3] (uint8)
     ↓
write_video
     ↓
output.mp4 (480×832, 81帧, 16fps, ~5秒)
```

---

## 第十二部分：视频保存 (L177-192)

```python
# Final output video
video = 255.0 * torch.cat(all_video, dim=1)

# Clear VAE cache
pipeline.vae.model.clear_cache()

# Save the video if the current prompt is not a dummy prompt
if idx < num_prompts:
    model = "regular" if not args.use_ema else "ema"
    for seed_idx in range(args.num_samples):
        if args.save_with_index:
            output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
        else:
            output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
        write_video(output_path, video[seed_idx], fps=16)
```

### 解释
1. **像素值缩放**：从 `[0, 1]` 缩放到 `[0, 255]`
2. **清理缓存**：释放 VAE 解码器的缓存
3. **保存视频**：
   - 帧率：16 FPS
   - 文件名：索引模式 `0-0_ema.mp4` 或提示模式 `prompt[:100]-0.mp4`

---

## 📊 数据流总结

```
输入文本提示
    ↓
TextDataset 加载
    ↓
生成随机噪声 [B, 21, 16, 60, 104]
    ↓
pipeline.inference()
    ├── text_encoder: 文本 → 嵌入 [B, 512, 4096]
    ├── generator: 噪声 → 潜在空间 [B, 21, 16, 60, 104]
    └── vae.decode: 潜在空间 → 像素 [B, 81, 3, 480, 832]
    ↓
rearrange: [B, T, C, H, W] → [B, T, H, W, C]
    ↓
× 255.0
    ↓
write_video: 保存为 MP4
```

---

## 🚀 运行示例

```bash
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --output_folder videos/output \
    --use_ema \
    --num_samples 2
```
