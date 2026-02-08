"""
================================================================================
model/base.py - 模型基类
================================================================================

【文件作用】
定义了所有模型的基类 BaseModel 和 SelfForcingModel。
提供了所有模型共用的初始化逻辑（构建 Generator, Teacher, Critic, VAE, TextEncoder）。
实现了 Self-Forcing 的核心驱动逻辑：_run_generator 和 _consistency_backward_simulation。

【关键功能】
1. _initialize_models: 统一初始化所有子模块（生成器、教师、Critic等），并设置哪些需要梯度。
2. _run_generator: 训练时的 forward pass 入口。它会调用推理 pipeline 来生成假视频。
3. _consistency_backward_simulation: 调用 pipeline/self_forcing_training.py 执行实际的自回归生成。
================================================================================
"""

from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from pipeline import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BaseModel(nn.Module):
    """
    基础模型类，处理各个子模块的初始化。
    """
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        
        # 加载去噪步数列表 (e.g., [1000, 750, 500, 250])
        # 这决定了模型在推理时走多少步
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            # 时间步反转逻辑 (如果需要)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        """
        初始化所有关键子模块。
        """
        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")

        # 1. Generator (生成器): 学生模型，需要训练 (requires_grad=True)
        # is_causal=True 表示这是一个支持自回归生成的因果模型
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        # 2. Real Score (教师模型): 预训练的 Wan2.1，作为Ground Truth分布的来源，不训练 (requires_grad=False)
        self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
        self.real_score.model.requires_grad_(False)

        # 3. Fake Score (Critic): 判别网络，学习区分生成样本和真实样本，需要训练
        self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, is_causal=False)
        self.fake_score.model.requires_grad_(True)

        # 4. Text Encoder (T5): 提取文本特征，冻结
        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        # 5. VAE: 视频编解码器，冻结
        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        # 获取调度器 (Scheduler)
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        随机生成时间步张量。
        支持对每个 Block 分别采样时间步，或者所有帧使用相同时间步。
        """
        if uniform_timestep:
            # 所有帧共用一个时间步
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            # ... (自回归分块时间步采样的复杂逻辑，这里主要用于更高级的训练配置)
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    """
    ================================================================================
    SelfForcingModel - Self-Forcing 技术的核心实现类
    ================================================================================
    
    【什么是 Self-Forcing?】
    传统的视频生成训练需要真实视频数据作为监督信号。但 Self-Forcing 采用了一种
    "无数据训练" (Data-Free Training) 的策略：
    
    1. 训练时，模型从纯噪声开始，自己生成一段视频（假视频）。
    2. 然后用这段假视频去计算损失（与教师模型的输出对比）。
    3. 梯度反向传播回生成器，更新生成器的参数。
    
    这就像让模型"吃自己的狗粮"——用自己的输出来训练自己。
    
    【为什么这样做有效?】
    传统扩散模型训练存在"训练-推理不一致"问题：
    - 训练时：输入是真实视频加噪后的样本（质量很高）
    - 推理时：输入是模型自己上一步的输出（可能有误差）
    
    Self-Forcing 在训练时就使用模型自己的输出作为输入，消除了这种不一致。
    
    【本类的核心方法】
    1. _run_generator: 生成假视频的入口，被 DMD.generator_loss() 和 DMD.critic_loss() 调用
    2. _consistency_backward_simulation: 实际执行自回归生成的方法
    3. _initialize_inference_pipeline: 初始化用于训练时模拟推理的 Pipeline
    ================================================================================
    """
    
    def __init__(self, args, device):
        """
        初始化 SelfForcingModel。
        
        Args:
            args: 配置对象，包含所有超参数
            device: 运行设备 (cuda:0, cuda:1, ...)
        """
        # 调用父类 BaseModel 的初始化方法
        # 这会初始化 generator, real_score, fake_score, text_encoder, vae 等组件
        super().__init__(args, device)
        
        # ════════════════════════════════════════════════════════════════════════
        # 初始化去噪损失函数
        # ════════════════════════════════════════════════════════════════════════
        # get_denoising_loss 根据配置返回对应的损失函数类（如 FlowPredLoss）
        # 这个损失函数用于训练 Critic (fake_score) 网络
        # args.denoising_loss_type 通常是 "flow"（Flow Matching Loss）
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ════════════════════════════════════════════════════════════════════════════
        运行生成器，产生用于训练的"假"视频
        ════════════════════════════════════════════════════════════════════════════
        
        【调用时机】
        这个方法在以下两个地方被调用：
        1. DMD.generator_loss(): 计算生成器损失时，需要先生成假视频
        2. DMD.critic_loss(): 计算 Critic 损失时，也需要先生成假视频
        
        【输入参数详解】
        Args:
            image_or_video_shape: 要生成的视频形状，格式为 [B, F, C, H, W]
                - B: Batch Size (通常为 1)
                - F: 帧数 (通常为 21)
                - C: 通道数 (16，因为是在 Latent 空间)
                - H, W: 高度和宽度 (60, 104 是因为 VAE 将 480x832 压缩了 8 倍)
            
            conditional_dict: 条件信息字典，包含：
                - "context": 文本编码器输出的 Context Embeddings
                - "context_mask": Attention Mask
                - 其他条件信息...
            
            initial_latent: 可选的初始潜变量（用于 Image-to-Video 任务）
                - 如果提供，则第一帧使用这个 latent 而不是噪声
        
        【返回值详解】
        Returns:
            pred_image_or_video_last_21: 生成的视频 Latent [B, 21, 16, 60, 104]
            gradient_mask: 梯度掩码，指示哪些帧需要计算梯度
            denoised_timestep_from: 去噪的起始时间步
            denoised_timestep_to: 去噪的结束时间步
        ════════════════════════════════════════════════════════════════════════════
        """
        
        # ════════════════════════════════════════════════════════════════════════
        # Step 1: 验证配置 - 确保启用了反向模拟
        # ════════════════════════════════════════════════════════════════════════
        # backward_simulation 是 Self-Forcing 的核心特性，必须开启
        # 如果不开启，就变成了传统的需要真实数据的训练
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        
        # ════════════════════════════════════════════════════════════════════════
        # Step 2: 处理初始 Latent (用于 Image-to-Video 场景)
        # ════════════════════════════════════════════════════════════════════════
        # 如果用户提供了一张图片作为第一帧，我们需要把它传给 Pipeline
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        
        # ════════════════════════════════════════════════════════════════════════
        # Step 3: 计算噪声张量的形状
        # ════════════════════════════════════════════════════════════════════════
        # 噪声张量的形状决定了我们要生成多少帧视频
        if self.args.i2v:
            # 如果是 Image-to-Video 任务：
            # 第一帧是给定的图片 latent，所以噪声只需要生成剩余的帧
            # 例如：要生成 21 帧，第一帧已有，噪声形状是 [B, 20, C, H, W]
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            # 如果是纯 Text-to-Video 任务：
            # 所有帧都从噪声生成，噪声形状是 [B, 21, C, H, W]
            noise_shape = image_or_video_shape.copy()

        # ════════════════════════════════════════════════════════════════════════
        # Step 4: 计算本次训练要生成的帧数（这是一个随机数！）
        # ════════════════════════════════════════════════════════════════════════
        # 训练时，我们不总是生成固定长度的视频。而是随机采样一个长度。
        # 这样做的目的是让模型适应不同长度的生成任务，增强泛化能力。
        
        # independent_first_frame=True 表示第一帧单独处理（用于 [1, 4, 4, 4, ...] 的结构）
        # independent_first_frame=False 表示所有帧统一处理（用于 [3, 3, 3, ...] 的结构）
        min_num_frames = 20 if self.args.independent_first_frame else 21  # 最少生成的帧数
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames  # 最多生成的帧数
        
        # 计算 Block 数量的范围
        # num_frame_per_block 是每个Block包含的帧数（通常为3）
        # 例如：21帧 = 7个Block × 3帧/Block
        max_num_blocks = max_num_frames // self.num_frame_per_block  # 例如: 21 // 3 = 7
        min_num_blocks = min_num_frames // self.num_frame_per_block  # 例如: 21 // 3 = 7
        
        # ════════════════════════════════════════════════════════════════════════
        # Step 5: 随机采样 Block 数量，并同步给所有 GPU
        # ════════════════════════════════════════════════════════════════════════
        # 在分布式训练中，所有 GPU 必须生成相同长度的视频
        # 否则 FSDP 的梯度同步会出问题
        
        # 在 GPU 0 上随机生成一个数
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        
        # 广播给所有其他 GPU，确保大家用的是同一个数
        dist.broadcast(num_generated_blocks, src=0)
        
        # 转换为 Python int
        num_generated_blocks = num_generated_blocks.item()
        
        # 计算实际要生成的帧数
        num_generated_frames = num_generated_blocks * self.num_frame_per_block  # 例如: 7 * 3 = 21
        
        # 如果第一帧是独立的且没有提供初始 latent，需要额外加一帧
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        
        # 更新噪声形状的帧数维度
        noise_shape[1] = num_generated_frames  # 例如: [1, 21, 16, 60, 104]

        # ════════════════════════════════════════════════════════════════════════
        # Step 6: 调用 Self-Forcing 的核心方法，执行自回归生成
        # ════════════════════════════════════════════════════════════════════════
        # 这是最关键的一步！
        # 
        # torch.randn() 生成纯高斯噪声，形状为 [B, F, C, H, W]
        # _consistency_backward_simulation() 会调用 SelfForcingTrainingPipeline
        # 它会模拟整个推理过程：从噪声开始，逐帧自回归生成完整视频
        # 
        # 返回值:
        # - pred_image_or_video: 生成的视频 latent [B, F, C, H, W]
        # - denoised_timestep_from: 最后使用的去噪起始时间步（用于 DMD Loss 计算）
        # - denoised_timestep_to: 最后使用的去噪结束时间步
        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape, device=self.device, dtype=self.dtype),
            **conditional_dict,  # 传递所有条件信息（文本嵌入等）
        )
        
        # ════════════════════════════════════════════════════════════════════════
        # Step 7: 处理超长视频的截断
        # ════════════════════════════════════════════════════════════════════════
        # DMD Loss 计算时，我们只使用最后 21 帧
        # 如果生成的视频超过 21 帧，需要进行特殊处理
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                # 取出除了最后 20 帧之外的所有帧（即前面多出来的部分）
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                
                # 将这些 latent 解码成像素空间的视频帧
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                
                # 取最后一帧像素，这将作为"边界帧"
                frame = pixels[:, -1:, ...].to(self.dtype)
                
                # 调整维度顺序以适配 VAE 编码器
                # 从 [B, T, C, H, W] 变成 [B, C, T, H, W]
                frame = rearrange(frame, "b t c h w -> b c t h w")
                
                # 重新编码这一帧为 latent（这样可以断开梯度，避免显存爆炸）
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
                
            # 拼接：边界帧的 latent + 最后 20 帧的 latent = 21 帧
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        else:
            # 如果生成的帧数不超过 21，直接使用
            pred_image_or_video_last_21 = pred_image_or_video

        # ════════════════════════════════════════════════════════════════════════
        # Step 8: 创建梯度掩码 (Gradient Mask)
        # ════════════════════════════════════════════════════════════════════════
        # 为什么需要梯度掩码？
        # 
        # 在自回归生成中，第一个 Block 的生成比较特殊：
        # - 它没有任何上下文（KV Cache 是空的）
        # - 它的质量可能较差
        # 
        # 为了避免用这些低质量的帧来计算 Loss（可能导致训练不稳定），
        # 我们使用 gradient_mask 来屏蔽第一个 Block 的梯度。
        # 
        # gradient_mask[i] = True 表示第 i 帧参与 Loss 计算
        # gradient_mask[i] = False 表示第 i 帧不参与 Loss 计算
        
        if num_generated_frames != min_num_frames:
            # 如果生成的帧数比最小帧数多，说明有多余的"历史帧"
            # 这些历史帧的梯度需要被屏蔽
            
            # 初始化为全 True（所有帧都参与计算）
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            
            if self.args.independent_first_frame:
                # 如果第一帧是独立的，只屏蔽第一帧
                gradient_mask[:, :1] = False
            else:
                # 否则屏蔽整个第一个 Block（通常是前 3 帧）
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            # 如果生成的帧数等于最小帧数，不需要屏蔽
            gradient_mask = None

        # ════════════════════════════════════════════════════════════════════════
        # Step 9: 类型转换并返回
        # ════════════════════════════════════════════════════════════════════════
        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        ════════════════════════════════════════════════════════════════════════════
        一致性反向模拟 (Consistency Backward Simulation)
        ════════════════════════════════════════════════════════════════════════════
        
        【名字的含义】
        "反向模拟" 指的是：在训练时，我们模拟（Simulate）推理（Inference）过程。
        "一致性" 指的是：训练和推理使用完全相同的代码路径，保持一致。
        
        【这个方法做了什么？】
        1. 接收一个纯噪声张量 [B, F, C, H, W]
        2. 调用 SelfForcingTrainingPipeline.inference_with_trajectory()
        3. Pipeline 会模拟完整的自回归推理过程：
           - 逐 Block 生成视频帧
           - 使用 KV Cache 加速 Attention 计算
           - 执行多步去噪（如 1000 → 750 → 500 → 250）
        4. 返回生成的视频 latent
        
        【为什么要懒加载 Pipeline？】
        SelfForcingTrainingPipeline 需要引用 generator（生成器模型）。
        但在 __init__ 时，generator 可能还没有被 FSDP 包装。
        所以我们延迟到第一次调用时才初始化 Pipeline，确保拿到的是包装后的模型。
        
        Args:
            noise: 纯高斯噪声 [B, F, C, H, W]
            conditional_dict: 条件信息（文本嵌入、初始 latent 等）
            
        Returns:
            output: 生成的视频 latent [B, F, C, H, W]
            denoised_timestep_from: 去噪起始时间步
            denoised_timestep_to: 去噪结束时间步
        ════════════════════════════════════════════════════════════════════════════
        """
        
        # ════════════════════════════════════════════════════════════════════════
        # 懒加载 (Lazy Initialization) Pipeline
        # ════════════════════════════════════════════════════════════════════════
        # self.inference_pipeline 初始值是 None
        # 第一次调用时才真正创建 Pipeline 对象
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        # ════════════════════════════════════════════════════════════════════════
        # 调用 Pipeline 执行实际的推理模拟
        # ════════════════════════════════════════════════════════════════════════
        # 这个方法定义在 pipeline/self_forcing_training.py 中
        # 它会执行：
        # 1. 初始化 KV Cache
        # 2. 逐 Block 循环生成视频
        # 3. 在每个 Block 内执行多步去噪
        # 4. 随机选择一个步骤保留梯度（用于反向传播）
        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        ════════════════════════════════════════════════════════════════════════════
        初始化用于训练时模拟推理的 Pipeline
        ════════════════════════════════════════════════════════════════════════════
        
        【为什么要单独封装一个 Pipeline？】
        
        训练时的"推理模拟"和真正的推理有一些区别：
        1. 训练时需要保留某些步骤的梯度，但不是所有步骤（显存不够）
        2. 训练时需要返回额外的信息（如 denoised_timestep）用于 Loss 计算
        3. 训练时可能需要一些特殊处理（如 context_noise）
        
        SelfForcingTrainingPipeline 专门处理这些训练特有的逻辑。
        
        【传入的参数解释】
        - denoising_step_list: 去噪时间步列表 [1000, 750, 500, 250]
            每个 Block 会经过这 4 个时间步的去噪
        - scheduler: 噪声调度器，负责加噪/去噪的数学计算
        - generator: 生成器模型（已被 FSDP 包装），执行实际的去噪预测
        - num_frame_per_block: 每个 Block 包含的帧数（通常为 3）
        - independent_first_frame: 是否独立处理第一帧
        - same_step_across_blocks: 是否在所有 Block 使用相同的随机梯度步
        - last_step_only: 是否只在最后一步保留梯度
        - num_max_frames: 最大帧数（用于预分配 KV Cache）
        - context_noise: 上下文噪声强度（用于缓存时添加微量噪声，防止过拟合）
        ════════════════════════════════════════════════════════════════════════════
        """
        self.inference_pipeline = SelfForcingTrainingPipeline(
            # 去噪时间步列表，例如 [1000, 750, 500, 250]
            # 模型会在这 4 个时间步逐步去噪，完成一个 Block 的生成
            denoising_step_list=self.denoising_step_list,
            
            # 噪声调度器，包含 add_noise, step 等方法
            # 用于计算 x_t = sqrt(alpha) * x_0 + sqrt(1-alpha) * noise
            scheduler=self.scheduler,
            
            # 生成器模型。注意：这里传入的是 FSDP 包装后的模型
            # 这意味着模型参数是分布在多个 GPU 上的
            generator=self.generator,
            
            # 每个 Block 的帧数。Self-Forcing 采用分块生成策略：
            # 每次生成 num_frame_per_block 帧，然后更新 KV Cache，再生成下一批
            num_frame_per_block=self.num_frame_per_block,
            
            # 是否独立处理第一帧。对于某些模型架构：
            # - True: [1, 3, 3, 3, ...] 结构，第一帧单独生成
            # - False: [3, 3, 3, ...] 结构，所有帧统一处理
            independent_first_frame=self.args.independent_first_frame,
            
            # 是否在所有 Block 使用相同的随机梯度步
            # - True: 所有 Block 在同一个去噪步（如第 2 步）保留梯度
            # - False: 每个 Block 独立随机选择梯度步
            same_step_across_blocks=self.args.same_step_across_blocks,
            
            # 是否只在最后一步保留梯度
            # - True: 只在去噪列表的最后一步（如 250）保留梯度
            # - False: 随机选择一步保留梯度
            last_step_only=self.args.last_step_only,
            
            # 最大生成帧数，用于预分配 KV Cache 的大小
            # KV Cache 大小 = num_max_frames * seq_len_per_frame
            num_max_frames=self.num_training_frames,
            
            # 上下文噪声强度。在更新 KV Cache 时，给生成的帧添加微量噪声
            # 这有助于防止模型过度依赖历史 Cache，增强鲁棒性
            context_noise=self.args.context_noise
        )
