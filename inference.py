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

# -----------------------------------------------------------------------------
# 参数解析配置
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="配置文件路径")  # Path to the config file
parser.add_argument("--checkpoint_path", type=str, help="模型检查点文件夹路径")  # Path to the checkpoint folder
parser.add_argument("--data_path", type=str, help="数据集路径")  # Path to the dataset
parser.add_argument("--extended_prompt_path", type=str, help="扩展提示词路径")  # Path to the extended prompt
parser.add_argument("--output_folder", type=str, help="输出文件夹")  # Output folder
parser.add_argument("--num_output_frames", type=int, default=21,
                    help="滑动窗口之间的重叠帧数")  # Number of overlap frames between sliding windows
parser.add_argument("--i2v", action="store_true", help="是否执行图生视频 (默认为文生视频)")  # Whether to perform I2V (or T2V by default)
parser.add_argument("--use_ema", action="store_true", help="是否使用 EMA 参数")  # Whether to use EMA parameters
parser.add_argument("--seed", type=int, default=0, help="随机种子")  # Random seed
parser.add_argument("--num_samples", type=int, default=1, help="每个提示词生成的样本数量")  # Number of samples to generate per prompt
parser.add_argument("--save_with_index", action="store_true",
                    help="是否使用索引而不是提示词作为文件名保存视频")  # Whether to save the video using the index or prompt as the filename
args = parser.parse_args()

# -----------------------------------------------------------------------------
# 初始化分布式推理环境
# -----------------------------------------------------------------------------
if "LOCAL_RANK" in os.environ:
    # 如果环境变量中有 LOCAL_RANK，说明是多 GPU 分布式运行
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size() # 获取总进程数
    set_seed(args.seed + local_rank)   # 为每个进程设置不同的随机种子
else:
    # 单 GPU 运行
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB') # 打印显存剩余空间
low_memory = get_cuda_free_memory_gb(gpu) < 40 # 如果显存小于 40GB，标记为低显存模式

torch.set_grad_enabled(False) # 推理模式，禁用梯度计算

# -----------------------------------------------------------------------------
# 加载配置
# -----------------------------------------------------------------------------
config = OmegaConf.load(args.config_path) # 加载用户提供的配置
default_config = OmegaConf.load("configs/default_config.yaml") # 加载默认配置
config = OmegaConf.merge(default_config, config) # 合并配置，用户配置覆盖默认配置

# -----------------------------------------------------------------------------
# 初始化推理管道 (Pipeline)
# -----------------------------------------------------------------------------
if hasattr(config, 'denoising_step_list'):
    # 如果配置中有 denoising_step_list，使用少步推理管道 (Few-step inference)
    pipeline = CausalInferencePipeline(config, device=device)
else:
    # 否则使用多步扩散推理管道 (Multi-step diffusion inference)
    pipeline = CausalDiffusionInferencePipeline(config, device=device)

# -----------------------------------------------------------------------------
# 加载模型检查点
# -----------------------------------------------------------------------------
if args.checkpoint_path:
    # 加载指定的 checkpoints
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    # 根据是否使用 EMA (指数移动平均) 加载对应的 generator 权重
    pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])

# 将管道转换为 bfloat16 精度以节省显存和加速
pipeline = pipeline.to(dtype=torch.bfloat16)

# -----------------------------------------------------------------------------
# 模型设备分配 (显存优化)
# -----------------------------------------------------------------------------
if low_memory:
    # 低显存模式下，使用动态交换安装器来管理 text_encoder
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    # 显存充足时，直接将 text_encoder 放到 GPU
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu) # 生成器始终在 GPU
pipeline.vae.to(device=gpu)       # VAE 始终在 GPU


# -----------------------------------------------------------------------------
# 创建数据集
# -----------------------------------------------------------------------------
if args.i2v:
    # 图生视频模式
    assert not dist.is_initialized(), "I2V does not support distributed inference yet" # 目前 I2V 不支持分布式
    transform = transforms.Compose([
        transforms.Resize((480, 832)), # 调整图像大小
        transforms.ToTensor(),         # 转换为 Tensor
        transforms.Normalize([0.5], [0.5]) # 归一化到 [-1, 1]
    ])
    # 使用图文对数据集
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    # 文生视频模式，使用文本数据集
    dataset = TextDataset(prompt_path=args.data_path, extended_prompt_path=args.extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}") # 打印提示词总数

# -----------------------------------------------------------------------------
# 创建 DataLoader
# -----------------------------------------------------------------------------
if dist.is_initialized():
    # 分布式模式下使用 DistributedSampler
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    # 单机模式下使用 SequentialSampler
    sampler = SequentialSampler(dataset)
# 创建 DataLoader，batch_size 固定为 1
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# -----------------------------------------------------------------------------
# 创建输出目录
# -----------------------------------------------------------------------------
# 仅在主进程中创建输出目录，避免竞争条件
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier() # 等待所有进程同步


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    """
    辅助编码函数 (看起来在这里未被直接调用，可能是遗留代码或备用)
    """
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


# -----------------------------------------------------------------------------
# 主推理循环
# -----------------------------------------------------------------------------
for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # DataLoader batch_size=1，batch_data 可能已经被包装了一层
    # 这里为了方便解包 batch 数据
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # 记录生成的帧数 (latents)

    if args.i2v:
        # 图生视频: batch 包含图像和文本描述
        prompt = batch['prompts'][0]  # 获取 batch 中的文本
        prompts = [prompt] * args.num_samples

        # 处理图像: 调整维度并转为 bfloat16
        # Batch size 为 1，squeeze(0) 去掉 batch 维，得到 [C, H, W]
        # unsqueeze(0) 加回 batch 维 [1, C, H, W]
        # unsqueeze(2) 增加时间维 [1, C, 1, H, W]
        # image shape: [1, 3, 1, 480, 832] (Assumption: C=3 RGB, T=1, H=480, W=832)
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # 将输入图像通过 VAE 编码为 Latent
        # VAE Encoding: [1, 3, 1, 480, 832] -> [1, 16, 1, 60, 104]
        # initial_latent shape: [1, 16, 1, 60, 104] (C=16 latent channels)
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        
        # 调整维度以适配 Pipeline: Pipeline 期望 [B, T, C, H, W]
        # rearrange: [1, 16, 1, 60, 104] -> [1, 1, 16, 60, 104]
        initial_latent = rearrange(initial_latent, 'b c t h w -> b t c h w')
        
        # 复制 image latent 以匹配 num_samples (Batch Expansion)
        # initial_latent shape: [num_samples, 1, 16, 60, 104]
        initial_latent = initial_latent.repeat(args.num_samples, 1, 1, 1, 1)

        # 采样初始噪声: 
        # 因为第一帧已知 (Initial Latent)，我们需要生成剩余的 T-1 帧
        # sampled_noise shape: [num_samples, num_output_frames - 1, 16, 60, 104]
        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # 文生视频: batch 仅包含文本提示
        prompt = batch['prompts'][0]
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] * args.num_samples
        else:
            prompts = [prompt] * args.num_samples
        initial_latent = None

        # 采样初始噪声: 
        # 文生视频需要生成所有 T 帧
        # sampled_noise shape: [num_samples, num_output_frames, 16, 60, 104]
        # default num_output_frames = 21
        sampled_noise = torch.randn(
            [args.num_samples, args.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    # -------------------------------------------------------------------------
    # 执行推理: 生成 81 帧 (或其他配置的帧数)
    # -------------------------------------------------------------------------
    # video (Decoded Pixel): [num_samples, num_output_frames, 3, 480, 832]
    # latents (Latent Space): [num_samples, num_output_frames, 16, 60, 104]
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        low_memory=low_memory,
    )
    
    # 调整视频维度以适配 write_video
    # [B, T, C, H, W] -> [B, T, H, W, C]
    # [num_samples, 21, 3, 480, 832] -> [num_samples, 21, 480, 832, 3]
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # -------------------------------------------------------------------------
    # 后处理与保存
    # -------------------------------------------------------------------------
    # 将像素值映射回 0-255 范围
    # video shape: [num_samples, 21, 480, 832, 3] (assuming len(all_video)=1 and concat dim=1?? Wait, cat dim 1 means T dim)
    # Note: all_video is a list of videos. If dataloader batch > 1, this loop runs once.
    # Actually, dataloader produces 1 batch item per iter. all_video only has 1 item here.
    # So cat(dim=1) is essentially a no-op if len=1, or concatenating along time if we were accumulating blocks?
    # Here it seems we only process one sequence per batch.
    video = 255.0 * torch.cat(all_video, dim=1)

    # 清除 VAE 缓存
    pipeline.vae.model.clear_cache()

    # 保存视频
    # 如果 idx < num_prompts，说明是有效数据 (防止 DistributedSampler padding 的重复数据)
    if idx < num_prompts:
        model = "regular" if not args.use_ema else "ema"
        for seed_idx in range(args.num_samples):
            # 所有进程都保存其生成的视频
            if args.save_with_index:
                # 使用索引作为文件名
                output_path = os.path.join(args.output_folder, f'{idx}-{seed_idx}_{model}.mp4')
            else:
                # 使用提示词前100字符作为文件名
                output_path = os.path.join(args.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)
