"""
================================================================================
model/dmd.py - DMD 模型定义
================================================================================

【文件作用】
实现了 DMD (Distribution Matching Distillation) 算法的核心逻辑。
该模块包含：
1. DMD 类: 继承自 SelfForcingModel，封装了生成器和评分网络。
2. 损失函数计算:
   - generator_loss: 计算生成器的蒸馏损失（使其匹配教师模型分布）。
   - critic_loss: 计算 Critic 的去噪损失（使其学会评估生成样本的质量）。
   - _compute_kl_grad: 计算 KL 散度的梯度，这是 DMD 算法的数学核心。

【核心思想】
通过最小化生成分布 (p_fake) 和真实数据分布 (p_real) 之间的 KL 散度来训练生成器。
但是我们没有直接访问 p_real 的概率密度，而是利用 Score Matching 的技巧：
∇ D_KL(p_fake || p_real) ≈ E [ Score_fake(x) - Score_real(x) ]
其中：
- Score_fake(x) 由 Critic 网络近似。
- Score_real(x) 由预训练的“教师”扩散模型近似。
================================================================================
"""

from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch

from model.base import SelfForcingModel


class DMD(SelfForcingModel):
    """
    DMD 模型类
    包含生成器、真实评分网络（教师）和伪造评分网络（Critic）。
    """
    def __init__(self, args, device):
        """
        初始化 DMD (Distribution Matching Distillation) 模块。
        """
        super().__init__(args, device)
        
        # 从配置中读取超参数
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1) # e.g., 3
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21) # 训练时生成视频的总帧数

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        # 是否独立处理第一帧（通常第一帧是纯图像，后续是视频帧）
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
            
        # 启用梯度检查点以节省显存 (Gradient Checkpointing)
        # 用计算换显存，允许训练更大的 Batch Size 或模型
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # 推理 Pipeline (稍后懒加载)
        # 用于在训练过程中执行 Self-Forcing 的前向推理
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # 初始化 DMD 相关超参数
        self.num_train_timestep = args.num_train_timestep # 1000
        self.min_step = int(0.02 * self.num_train_timestep) # 20 (避免极小的时间步)
        self.max_step = int(0.98 * self.num_train_timestep) # 980
        
        # CFG (Classifier-Free Guidance) 的 Scale
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale # e.g., 3.0
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0 # Critic 通常不使用 CFG
            
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        # 确保 alphas_cumprod 在正确的设备上
        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 KL 散度的梯度 (DMD 论文核心公式 7)。
        这是将生成器分布推向真实分布的动力。
        
        Args:
            noisy_image_or_video: 加噪后的样本 x_t
            estimated_clean_image_or_video: 生成的原始样本 x_0
            timestep: 当前时间步 t
            conditional_dict: 条件嵌入 (Prompts)
            unconditional_dict: 无条件嵌入 (Negative Prompts)
            normalization: 是否归一化梯度 (论文公式 8)
        Returns:
            kl_grad: 计算出的梯度方向
        """
        # Step 1: 计算 Fake Score (Critic 的预测)
        # 这里的 fake_score 是一个可训练的网络，用于估计生成样本的 score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        # 如果 Critic 也使用了 CFG (通常不使用)
        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: 计算 Real Score (教师模型的预测)
        # real_score 是一个冻结的预训练模型 (Wan2.1-14B)，代表真实数据分布
        # 我们使用 CFG 来增强真实分布的质量
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        # 应用 CFG
        # pred_real_image = Cond + w * (Cond - Uncond)
        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: 计算 DMD 梯度 (DMD paper eq. 7)。
        # 方向 = Score_fake - Score_real
        # 这告诉我们：生成的样本应该往哪个方向移动，才能更像真实样本
        # 注意: 这里的预测是 x0 预测，所以方向与其 Score 梯度方向是相关的
        grad = (pred_fake_image - pred_real_image)

        # Step 4: 梯度归一化 (DMD paper eq. 8)。
        # 这一步对于训练稳定性至关重要
        if normalization:
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
            
        grad = torch.nan_to_num(grad) # 处理 NaN

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: Optional[int] = 0, # 这里类型标注应该是 int 或 None
        denoised_timestep_to: Optional[int] = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 DMD 损失。
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: 随机采样一个时间步 t
            # 我们希望在各个噪声水平上，生成和真实分布都是匹配的
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # 时间步偏移调整 (Timestep Shifting)
            # 用于将采样集中在对生成质量影响最大的中间信噪比区域
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            # Step 1.2: 加噪 (Add Noise)
            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: 计算 KL 梯度
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        # 这里的 Loss 本质上是一个 MSE 损失
        # 目标是：x_new = x_old - learning_rate * grad
        # 这等价于最小化 || x - (x - grad) ||^2
        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
            
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        生成器的损失计算函数。
        """
        # Step 1: 运行生成器，得到生成的视频 (Fake Video)
        # 这里会执行 Self-Forcing 的反向模拟 (Consistency Backward Simulation)
        # 即：从噪声开始，一步步生成完整的视频
        # 关键点：我们不需要真实视频，而是用模型自己的生成结果来训练自己！
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        # Step 2: 计算 DMD 损失
        # 看这个生成的视频分布是否匹配教师模型的分布
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Critic (Fake Score) 的损失计算函数。
        Critic 的作用是准确判断生成样本的 Score。
        """

        # Step 1: 同样，先用生成器生成假样本
        # 这一步不需要梯度，因为那是 Generator 的事
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
            )

        # Step 2: 随机采样时间步 t
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        # Step 3: 给生成的假样本加噪
        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        # Step 4: 让 Critic 预测
        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 5: 计算去噪损失 (Denoising Score Matching)
        # Critic 试图学会去噪这些“假”样本
        # 实际上就是标准的 Diffusion Loss，但作用在生成样本上
        if self.args.denoising_loss_type == "flow":
            # 如果是 Flow Matching 损失
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        # 计算损失 (MSE)
        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
