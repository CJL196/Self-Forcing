import types
from typing import List, Optional
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel


# =================================================================================
# utils/wan_wrapper.py 详细代码解读
# =================================================================================
# 这个文件是 Wan2.1 视频生成模型的核心封装脚本。
# 它主要定义了三个关键的包装类，用于统一管理和调用模型的不同组件：
#
# 1. WanTextEncoder: 负责处理文本提示词 (Prompt)，使用 T5 模型将文本转换为嵌入向量 (Embeddings)。
# 2. WanVAEWrapper: 负责视频数据的编码和解码，将像素空间的视频转换为潜在空间 (Latent Space) 的表示，及其逆过程。
# 3. WanDiffusionWrapper: 这是核心类，封装了 Diffusion Policy (实际上是 Flow Matching) 模型。
#    它处理噪声预测、去噪过程、时间步 (Timestep) 管理以及因果推理 (Causal Inference) 相关的 KV Cache 逻辑。
#
# 该文件整合了 wan.modules 下的多个底层模块，并提供了便于推理脚本调用的高层接口。
# =================================================================================

class WanTextEncoder(torch.nn.Module):
    """
    负责文本编码的包装类。
    使用谷歌的 UMT5-XXL 模型将文本转换为语义向量。
    """
    def __init__(self) -> None:
        super().__init__()


        # 初始化时加载 umt5_xxl 模型
        # encoder_only=True: 我们只需要编码器部分，不需要生成文本的解码器
        # return_tokenizer=False: 我们手动管理 tokenizer
        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False) # 强制设为评估模式且不需要梯度 (只推理，不训练)

        # 加载预训练权重
        # 路径硬编码为 "wan_models/Wan2.1-T2V-1.3B/..."
        self.text_encoder.load_state_dict(
            torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        # 加载对应的 Huggingface Tokenizer
        # seq_len=512: 文本最大长度限制为 512
        self.tokenizer = HuggingfaceTokenizer(
            name="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/", seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        """
        前向传播函数。
        Args:
            text_prompts: 字符串列表，例如 ["一只猫", "一只狗"]
        Returns:
            dict: 包含 'prompt_embeds' 的字典
        """
        # ids: [Batch_Size, 512], mask: [Batch_Size, 512]
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        
        # 将数据移到 GPU (假设模型在 GPU 上)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        
        # 计算每个 prompt 的实际有效长度 (用于后续可能的 mask 处理)
        seq_lens = mask.gt(0).sum(dim=1).long()
        
        # 2. 编码 (Encoding)
        # 调用 T5 模型获取 context embeddings
        # context: [Batch_Size, 512, 4096]
        context = self.text_encoder(ids, mask)

        # 3. 清理 Padding
        # 手动将 mask 为 0 (Padding) 部分的 embeddings 置零，
        # 防止这些无意义的噪声影响后续生成
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    """
    负责视频 VAE (Variational Autoencoder) 的包装类。
    用于将高维的视频像素压缩到低维的 Latent 空间，或反之。
    """
    def __init__(self):
        super().__init__()
        # 定义 Normalize (归一化) 和 De-normalize (反归一化) 用的均值 mean 和标准差 std。
        # 这些数值是训练时在大规模数据集上统计得到的。
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        """
        [Pixel -> Latent] 编码函数
        输入: pixel [batch_size, num_channels, num_frames, height, width]
        输出: latent [batch_size, num_frames, num_channels, height, width] (注意维度变化!)
        """
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        # 准备归一化参数 scale
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        # 对 batch 中的每一个视频进行编码
        # self.model.encode 内部会同时执行 "减去均值除以方差" 的操作
        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        
        # 维度置换 (Permute):
        # 原始 VAE 输出: [batch_size, num_channels, num_frames, height, width]
        # Diffusion 需要: [batch_size, num_frames, num_channels, height, width]
        # 主要区别在于把 num_frames (T) 移到了 channels (C) 前面，符合 Video Tokenizer 的常见格式
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """
        [Latent -> Pixel] 解码函数
        输入: latent [batch_size, num_frames, num_channels, height, width]
        输出: pixel [batch_size, num_channels, num_frames, height, width] (变回去了)
        """
        # 1. 维度还原 (Un-permute)
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        
        # 如果使用缓存解码 (use_cache)，通常是对长视频进行分块解码，要求 batch_size 为 1
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        # 反归一化参数
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        # 选择解码方式：普通解码 vs 缓存解码 (更省显存)
        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        # 执行解码
        output = []
        for u in zs:
            # decode -> float -> clamp -> squeeze
            # clamp_(-1, 1): 强制像素值在 [-1, 1] 范围内，对应 pytorch 图片的标准范围
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        
        # 再次置换 (这一步看起来有点多余，因为 VAE 输出本来就是 B, C, F, H, W。
        # 但可能是为了保证和 encode 的输入格式严格一致，或者 output 这里的格式取决于 decode 实现)
        # 假设 decode 输出是 [B, C, F, H, W]，之前的 stack 也是 B 维度。
        # 这里的 permute 0, 2, 1, 3, 4 会把它变成 [B, F, C, H, W] ???
        # 等等，上面 encode 的最后一步是 permute(0, 2, 1, 3, 4) 转成了 F在前。
        # 这里 decode 的第一步是 permute(0, 2, 1, 3, 4) 转回 C在前。
        # decode 函数吐出来的通常是 [C, F, H, W] (squeeze后)。
        # stack 后是 [B, C, F, H, W]。
        # 原代码最后一行 output.permute(0, 2, 1, 3, 4) 实际上是把它又变成了 [B, F, C, H, W] ???
        # 修正：根据 inference.py 的用法，最后输出 video 应该是 [B, F, C, H, W]。
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    """
    【核心类】Diffusion 模型包装器。
    
    重点解释：
    这个类封装了主要的生成模型。它使用 Flow Matching (一种更通用的 Diffusion) 算法。
    """
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0
    ):
        super().__init__()

        # =================================================================================
        # 1. 模型加载 (Model Loading)
        # =================================================================================
        # 支持加载标准 WanModel 或支持因果注意力 (Causal Attention) 的 CausalWanModel。
        # CausalWanModel 是 Self-Forcing 自回归长视频生成的关键，它支持 KV Cache。
        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")
        self.model.eval()

        # =================================================================================
        # 2. 设置时间步 (Timestep) 属性
        # =================================================================================
        # 如果是 causal 模式，每个帧可能有不同的 timestep (例如前一段是 clean=0，后一段是 noisy=T)。
        # 如果是非 causal 模式，所有帧通常共享同一个 timestep。
        self.uniform_timestep = not is_causal

        # =================================================================================
        # 3. 初始化调度器 (Scheduler)
        # =================================================================================
        # 使用 FlowMatchScheduler。Flow Matching 是 Diffusion Models 的一种。
        # 简单理解：Standard Diffusion 模拟“如何去除噪声”，Flow Matching 模拟“数据如何随时间流向噪声”。
        # shift=timestep_shift: 时间步偏移参数，用于调整噪声调度的分布，对生成质量影响很大。
        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104] (硬编码的序列长度，对应特定 Latent 尺寸)
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        """
        [实验性功能] 添加额外的分类/判别分支。
        这个方法动态地向模型插入新的 Layer。
        """
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        # 1. 分类头 (MLP): 预测某种类别
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104] (或者其它维度)
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        
        # 2. Register Tokens (寄存器令牌): 
        # 类似于 ViT 中的做法，用于存储全局信息而不干扰图像 Patch
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        # 3. GAN Attention Blocks:
        # 可能是用于让 Register tokens 与图像特征进行交互的专用 Attention 层
        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        【数学核心】Flow Matching 预测转换: Flow -> X0
        
        Flow Matching 的基本公式：xt = (1 - (1-sigma_min)t) * x0 + t * x1
        在去噪过程中，模型预测的是 v (velocity 也就是 flow_pred)。
        我们需要根据当前的噪声图 xt 和预测的速度 v，反推原本的干净图像 x0。
        
        公式: x0 = xt - sigma_t * flow_pred
        (这里假设了 sigma_t 是随时间变化的噪声强度系数)
        
        Args:
            flow_pred: 模型预测的“流”或“速度” [B, C, H, W]
            xt: 当前的噪声图 [B, C, H, W]
            timestep: 当前时间步 [B]
        """
        # 1. 精度提升: 为了数值稳定性，将关键变量转为 float64 (double)
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        # 2. 查找当前 timestep 对应的 sigma (噪声强度)
        # 计算差的绝对值，找到最近的索引
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        
        # 3. 获取对应的 sigma_t 并调整形状以支持广播
        # reshape(-1, 1, 1, 1) 是为了匹配图像/Latent 的 [B, C, H, W]
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        
        # 4. 核心计算
        x0_pred = xt - sigma_t * flow_pred
        
        # 5. 转回原始精度
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        【数学核心】Flow Matching 逆向预测: X0 -> Flow
        
        这是上一个函数的逆过程。已知目标原图 x0 和当前噪声图 xt，计算模型应该预测出什么样的 vector field。
        这通常用于 Teacher Forcing 或计算 Loss。
        
        公式: flow_pred = (xt - x0) / sigma_t
        """
        # 1. 精度转换 (同上)
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        
        # 2. 找到对应的 sigma_t (同上)
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        
        # 3. 核心计算
        flow_pred = (xt - x0_pred) / sigma_t
        
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None
    ) -> torch.Tensor:
        """
        模型推理主入口。
        处理复杂的参数和不同的运行模式 (普通推理, 因果推理 KV Cache, Teacher Forcing, 分类引导等)。
        """
        # 1. 提取文本条件嵌入
        prompt_embeds = conditional_dict["prompt_embeds"]

        # 2. 处理时间步
        # [B, F] -> [B]
        # 如果是 uniform (非 causal)，取第一个即可
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction (Flow Prediction)
        
        # =================================================================================
        # 3. 分支逻辑 (Branching Logic)
        # =================================================================================
        
        # --- 分支 A: 使用 KV Cache ---
        # 场景：自回归生成，长视频推理。
        if kv_cache is not None:
             # permute(0, 2, 1, 3, 4) 是为了将 [B, F, C, H, W] 转为 DiT 需要的格式
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,    # 传入缓存
                crossattn_cache=crossattn_cache,
                current_start=current_start, # 告诉模型当前处理的是视频的第几帧 (patch位置)
                cache_start=cache_start
            ).permute(0, 2, 1, 3, 4) # 转换回 [B, F, C, H, W]
        
        else:
            # --- 分支 B: 提供 clean_x ---
            # 场景：Teacher Forcing 训练，或者某种引导生成。
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4), # 传入目标原图
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                 # --- 分支 C: 分类模式 ---
                 # 场景：使用了 adding_cls_branch 增加的分类头
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                # --- 分支 D: 普通去噪推理 ---
                # 场景：最常见的基础 T2V 生成
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len
                    ).permute(0, 2, 1, 3, 4)

        # 4. 结果转换 Flow -> X0
        # 模型输出的是 flow_pred (速度场)，我们把它转回 pred_x0 (预测图像)，
        # 这对于可视化当前生成效果很有帮助。
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1), # Flatten batch 和 frames 维度，因为是一起算的
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2]) # 恢复维度 [B, F]

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
