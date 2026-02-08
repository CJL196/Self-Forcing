"""
================================================================================
pipeline/self_forcing_training.py - Self-Forcing 训练 Pipeline
================================================================================

【文件作用】
实现了 Self-Forcing 的核心自回归生成逻辑。
在训练过程中，这个 Pipeline 会被调用来模拟视频生成过程，但是带有特殊的 Gradient Checkpointing 和 KV Cache 逻辑。
它允许我们在没有真实视频数据的情况下，通过让模型"吃自己的狗粮"（即用自己的生成结果作为下一步的输入）来训练模型。

【核心特性】
1. KV Cache 管理: 显式初始化和更新 KV Cache (self.kv_cache1)，避免对已经生成的帧重复计算 Attention。
2. 自回归生成循环:
   - 外层循环 (Temporal): 逐个 Block (e.g., 每 3 帧) 生成视频。
   - 内层循环 (Spatial/Denoising): 在每个 Block 内，进行少步数 (e.g., 4步) 去噪。
3. 随机退出机制 (Self-Forcing 核心):
   - 为了节省训练显存和计算，我们不在每一步都应用反向传播。
   - 而是随机选择一个去噪步数 (exit_flag)，只在该步保留梯度用于训练，其他步骤仅做 torch.no_grad() 推理。
================================================================================
"""

from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    """
    自强制训练 Pipeline。
    负责执行带有 KV Cache 和随机梯度检查点的自回归视频生成。
    """
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 **kwargs):
        """
        Args:
            denoising_step_list: 去噪步数列表，e.g., [1000, 750, 500, 250]
            scheduler: 噪声调度器
            generator: 生成器模型 (已被 FSDP 包装)
            num_frame_per_block: 每次生成的帧数块大小 (默认为 3)
            kv_cache_size: KV Cache 容量，预分配以避免动态内存分配开销
        """
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        
        # 移除 0 时间步，因为 inference 到 0 就结束了，不需要再作为输入
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        # Wan 模型特定参数
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560 # 每帧的 Token 数量 (patch化后)
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length # 预分配显存

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        """
        生成并同步退出标志 (exit_flags)。
        这决定了在每个 Block 的第几步去噪时计算梯度。
        必须在所有 GPU 间同步，以保证 FSDP 正常工作。
        """
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # 随机选择每个 Block 在哪一步保留梯度
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # 广播给所有进程
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        """
        执行带有轨迹记录的推理过程。
        
        Args:
            noise: 初始纯高斯噪声 [B, F, C, H, W]
        Returns:
            output: 生成的视频潜变量 (包括历史帧的轨迹)
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        # 计算总共有多少个 Block 需要生成
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
            
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # ═══════════════════════════════════════════════════════
        # Step 1: 初始化 KV Cache
        # ═══════════════════════════════════════════════════════
        # 初始化为全零张量，预分配最大显存
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        # Step 2: 处理初始 Latent (如果是 I2V 任务或有条件输入)
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            output[:, :1] = initial_latent
            # 运行一次生成器，仅为了填充 KV Cache
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0, # t=0
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # ═══════════════════════════════════════════════════════
        # Step 3: 自回归时序生成循环 (Temporal Loop)
        # ═══════════════════════════════════════════════════════
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames # 第一帧可能单独生成
            
        # 同步随机退出标志
        num_denoising_steps = len(self.denoising_step_list) # e.g., 4
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        
        # 确定从哪一帧开始计算梯度 (通常是只对最后生成的几个 Block 计算 Loss)
        # 这里硬编码为最后 21 帧
        start_gradient_frame_index = num_output_frames - 21

        # 遍历每个 Block (e.g., 0, 1, 2, ...)
        for block_index, current_num_frames in enumerate(all_num_frames):
            # 获取当前 Block 对应的时间段噪声
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # ──────────────────────────────────────────────────
            # Step 3.1: 空间去噪循环 (Spatial Denoising Loop)
            # ──────────────────────────────────────────────────
            # 在当前 Block 上进行多步去噪 (e.g., 1000 -> 750 -> 500 -> 250)
            for index, current_timestep in enumerate(self.denoising_step_list):
                # 判断当前步是否是选定的梯度计算步
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])

                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    # 🚀 Case A: 只推理，不保留梯度 (torch.no_grad)
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1, # 使用 KV Cache 加速
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        # 准备下一步的输入: 加噪声到下一级的 t
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # ════════════════════════════════════════════════════════════════════════════════════════
                    # 🎯 Case B: 这是被随机选中的"梯度步" (exit_flag == True)
                    # ════════════════════════════════════════════════════════════════════════════════════════
                    # 
                    # 【背景知识：什么是 exit_flag？】
                    # 在上面的代码中，我们为每个 Block 随机选择了一个 exit_flag（退出标志）。
                    # 例如，去噪步骤列表是 [1000, 750, 500, 250]，共 4 步。
                    # exit_flag 可能是 0, 1, 2, 或 3 中的任意一个。
                    # 
                    # 当 index == exit_flag 时，即当前去噪步是被选中的"梯度步"：
                    # - 我们会在这一步生成结果后立即退出循环（break）
                    # - 不再继续后续的去噪步骤
                    # 
                    # 【为什么要随机选择一步退出？】
                    # 1. 显存节省：如果对所有 4 步都保留梯度，显存会爆炸
                    # 2. Self-Forcing 核心思想：训练时模拟推理的"不完美"状态
                    #    - 推理时，每一步的输出都可能有误差
                    #    - 训练时，我们故意使用中间步骤的输出（而非最终完美输出）
                    #    - 这样模型学会了在有误差的情况下也能继续生成
                    # 
                    # 【两种子情况的区别】
                    # 我们把视频分成两部分：
                    # - "历史区域"：current_start_frame < start_gradient_frame_index
                    #   这些是较早生成的帧，不参与 Loss 计算
                    # - "训练区域"：current_start_frame >= start_gradient_frame_index
                    #   这些是最后 21 帧，会参与 DMD Loss 计算
                    # 
                    # start_gradient_frame_index = num_output_frames - 21
                    # 例如：如果总共生成 30 帧，那么 start_gradient_frame_index = 30 - 21 = 9
                    # 意味着第 0-8 帧是"历史区域"，第 9-29 帧是"训练区域"
                    # ════════════════════════════════════════════════════════════════════════════════════════
                    
                    if current_start_frame < start_gradient_frame_index:
                        # ──────────────────────────────────────────────────────────────────────────────────
                        # 子情况 B1：当前 Block 属于"历史区域"
                        # ──────────────────────────────────────────────────────────────────────────────────
                        # 即使这是被选中的"梯度步"，但因为这些帧不参与 Loss 计算，
                        # 所以我们仍然使用 torch.no_grad()，不保留梯度。
                        # 
                        # 这样做的原因：
                        # 1. 节省显存：早期帧的梯度对 Loss 没有贡献，保留它们只是浪费显存
                        # 2. 训练效率：梯度图越短，反向传播越快
                        # 
                        # 举个例子：
                        # - 假设生成 30 帧，每 Block 3 帧，共 10 个 Block
                        # - start_gradient_frame_index = 9，即只有最后 7 个 Block (第 3-9 块) 的帧参与训练
                        # - 前 3 个 Block (第 0-2 块) 的帧虽然也会被生成，但不会有梯度
                        # ──────────────────────────────────────────────────────────────────────────────────
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        # ──────────────────────────────────────────────────────────────────────────────────
                        # 子情况 B2：当前 Block 属于"训练区域" —— 这里是真正的训练时刻！
                        # ──────────────────────────────────────────────────────────────────────────────────
                        # 这里没有 torch.no_grad()，意味着 PyTorch 会记录完整的计算图。
                        # 后续的 loss.backward() 会把梯度一路传回到这里的 self.generator 参数。
                        # 
                        # 【这一步的输出 denoised_pred 会被用来干什么？】
                        # 1. 首先，它会被记录到 output 张量中（见下面的 Step 3.2）
                        # 2. output 最终会返回给 DMD.generator_loss() 或 DMD.critic_loss()
                        # 3. Loss 函数会用这个 output 来计算与教师模型的分布差距
                        # 4. loss.backward() 会通过这里的 denoised_pred 反向传播梯度
                        # 5. 梯度最终到达 self.generator 的权重，触发参数更新
                        # 
                        # 【为什么调用 self.generator 时没有 .backward()？】
                        # 因为这里只是前向传播（forward pass）。
                        # backward() 是在外层的 DMD.generator_loss() 返回后由 Trainer 调用的。
                        # PyTorch 的自动求导机制会记住整个计算图，延迟到 backward() 时统一计算梯度。
                        # ──────────────────────────────────────────────────────────────────────────────────
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    
                    # ════════════════════════════════════════════════════════════════════════════════════════
                    # 🛑 关键操作：立即退出当前 Block 的去噪循环！
                    # ════════════════════════════════════════════════════════════════════════════════════════
                    # 
                    # 【为什么要 break？】
                    # 这是 Self-Forcing 的核心设计！让我详细解释：
                    # 
                    # 假设去噪步骤是 [1000, 750, 500, 250]，exit_flag = 1（即在第 2 步退出）：
                    # 
                    # 正常推理流程（4 步完整去噪）：
                    #   噪声 x_1000 → 模型 → x_750 → 模型 → x_500 → 模型 → x_250 → 模型 → 清晰图像 x_0
                    # 
                    # Self-Forcing 训练流程（在第 2 步退出）：
                    #   噪声 x_1000 → 模型 → x_750 → 【退出！直接用 x_750 作为这个 Block 的输出】
                    # 
                    # 问题：x_750 还很模糊（还有噪声），不是完美的图像，为什么要用它？
                    # 
                    # 答案：这正是 Self-Forcing 的精髓！
                    # 
                    # 1. 【训练推理一致性】
                    #    - 推理时，当前帧的输出会作为下一帧的 KV Cache 输入
                    #    - 如果当前帧有误差，这个误差会传播到下一帧
                    #    - 传统训练使用 Ground Truth，模型没学过如何处理误差
                    #    - Self-Forcing 故意使用不完美的中间结果，让模型学会"容错"
                    # 
                    # 2. 【随机性带来鲁棒性】
                    #    - exit_flag 是随机的，有时在 step 0 退出（噪声很大），有时在 step 3 退出（基本清晰）
                    #    - 模型被迫学会处理各种质量级别的输入
                    #    - 这大大增强了模型在推理时的稳定性
                    # 
                    # 3. 【计算效率】
                    #    - 不需要像真正推理那样跑完所有 4 步
                    #    - 平均只需要跑 ~2 步就退出，节省了一半的计算
                    # ════════════════════════════════════════════════════════════════════════════════════════
                    break

            # ──────────────────────────────────────────────────
            # Step 3.2: 记录输出
            # ──────────────────────────────────────────────────
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # ──────────────────────────────────────────────────
            # Step 3.3: 更新 KV Cache (关键!)
            # ──────────────────────────────────────────────────
            # 用生成的 Clearn Frame (或者加了微量 Context Noise 的 Frame) 
            # 以 t=0 再次运行生成器，目的是更新 kv_cache1，供下一个 Block 使用。
            context_timestep = torch.ones_like(timestep) * self.context_noise
            
            # 添加微量噪声 (如果配置了 context_noise) 防止过拟合
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            
            # 必须用 no_grad，因为我们不希望梯度通过 KV Cache 传播到上一个 Block
            # 我们只训练当前 Block 的生成能力，利用之前的 Context
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1, # ← 更新 Cache
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # 更新当前帧指针
            current_start_frame += current_num_frames

        # Step 3.5: 返回去噪时间步信息 (用于 Loss 计算)
        # 告诉 Loss 函数当前采用了哪个时间步的预测结果
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        初始化 Per-GPU KV cache。
        结构: List[Dict]，列表长度为 Transformer 层数。
        Dict 包含 'k', 'v' 以及索引指针。
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device), # 全局 token 索引
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)   # 局部 token 索引
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        初始化 Cross-Attention Cache (用于缓存文本 Context 的 Attention)
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
