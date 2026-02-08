from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # =================================================================================
        # 1. 核心模型初始化 (Initialize all models)
        # =================================================================================
        # WanDiffusionWrapper: 扩散生成器核心。
        # 这里传入 is_causal=True 是至关重要的一步！
        # 这告诉底层模型："嘿，我们现在要做视频生成了，请你开启 KV Caching 模式，
        # 不要像训练时那样一次性看全图，而是要像 GPT 那样一段一段地生成。"
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        
        # WanTextEncoder: 文本编码器 (通常是 T5)。
        # 负责把用户的 "A cat walking on the grass" 变成机器能懂的向量 (Embeddings)。
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        
        # WanVAEWrapper: 变分自编码器。
        # 扩散模型是在“潜空间 (Latent Space)”里工作的 (压缩后的模糊特征)。
        # VAE 负责最后一步：把潜空间的特征“解压”回人眼能看的像素视频。
        self.vae = WanVAEWrapper() if vae is None else vae

        # =================================================================================
        # 2. 初始化因果推理超参数 (Initialize all causal hyperparmeters)
        # =================================================================================
        self.scheduler = self.generator.get_scheduler()
        
        # denoising_step_list: 去噪步数计划表。
        # 在 Self-Forcing 算法中，因为使用了 DMD 蒸馏技术，步数通常非常少 (例如只有 4 步)。
        # 比如: [1000, 750, 500, 250]。这意味着每生成一小段视频，只需要模型跑 4 次。
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        
        # warp_denoising_step: 一个高级的时间步映射技巧，用于微调采样过程。
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Wan2.1 模型的标准配置
        self.num_transformer_blocks = 30  # 模型深度
        self.frame_seq_length = 1560      # 每帧对应的 Token 数量 (经过 Patchify 后)

        # KV Cache 容器，稍后在 inference 中会分配巨大的显存给它
        self.kv_cache1 = None
        self.args = args
        
        # num_frame_per_block: 【关键参数】
        # 控制每次生成多少帧。在 Self-Forcing 中，通常设为 3。
        # 这意味着：生成 0-3 帧 -> 固化 -> 生成 3-6 帧 (看 0-3) -> 固化 -> 生成 6-9 帧...
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block (每次生成帧数)")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        核心推理函数：执行 Self-Forcing 的因果推理过程。

        ⚠️ 宏观逻辑 (Block-Based Autoregressive Generation):
        1. 分块 (Split): 我们不会一次性生成所有帧，而是把视频切成小块(Block), 每个 Block 比如 3 帧。
        2. 接龙 (Chain): 先生成第 1 块。生成好后，把它的特征"冻结"并存入 KV Cache。
        3. 依赖 (dependency): 生成第 2 块时，模型会读取 Cache 里的第 1 块特征，确保连贯性。
        4. 循环 (Loop): 如此往复，直到生成完整视频。

        参数解析:
        - noise: [Batch, Total_Frames, C, H, W]。
            这是高斯噪声起点。注意：它的形状(Total_Frames)直接决定了最终生成的视频长度。
        - text_prompts: 用户输入的文本提示词。
        - initial_latent: [Batch, Input_Frames, C, H, W]。
            I2V (图生视频) 时，这里是首帧的 Latent。
            Video Extension (视频扩充) 时，这里是前段视频的 Latent。
            T2V (文生视频) 时，这里是 None。
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        
        # =================================================================================
        # 2.2 变量初始化与分块计算
        # 计算我们需要循环多少次 (num_blocks) 才能填满这 num_frames 帧
        # =================================================================================
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # 这是最常见的情况。例如：
            # 如果我们要生成 21 帧，每块 3 帧 (num_frame_per_block=3)。
            # 那么 num_blocks = 21 / 3 = 7。我们需要跑 7 次大循环。
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # 这是一个极少用的测试分支，第一帧独立生成，不用管它
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        
        # 2.3 文本编码
        # 调用 T5 Encoder，把 Prompt 文本变成 embeddings。
        # 这些 embeddings 会被后续所有帧的生成过程复用。
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # 2.4 准备输出画布
        # 创建一个全零张量 (Canvas)。
        # 此时它里面什么都没有，我们接下来的循环会把生成好的 latents 一块一块“填”进去。
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # =================================================================================
        # 🔥 Step 1: KV Cache 的初始化 (Infrastructure Setup)
        # =================================================================================
        # 这是因果推理的地基。
        if self.kv_cache1 is None:
            # === Case A: 第一次运行 ===
            # 我们需要向 GPU 申请一大块显存。
            # kv_cache1 是一个列表，长度等于 Transformer 层数 (30)。
            # 每一层包含一个字典：{'k': ..., 'v': ..., 'global_end_index': ...}
            # 注意：_initialize_kv_cache 会按照【最大可能的序列长度】一次性分配内存，
            # 而不是动态 append。这能极大地减少显存碎片。
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            # 初始化 Cross-Attention Cache (用于缓存文本特征的 attention 结果)
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # === Case B: 显存复用 (Memory Reuse) ===
            # 如果之前的推理已经分配过 cache，我们直接复用物理显存。
            # 只是把 index 指针归零。这比 free 再 malloc 快得多，且安全。
            
            # 重置 Cross Attention 状态
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            
            # 重置 KV Cache 指针
            for block_index in range(len(self.kv_cache1)):
                # global_end_index = 0 意味着我们逻辑上清空了 cache，
                # 但物理显存还在那里，等待被新数据覆盖。
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # =================================================================================
        # 🔥 Step 2: 预填充上下文 (Prefill Context)
        # =================================================================================
        # 场景：I2V (图生视频) 或 视频续写。
        # 问题：KV Cache 现在是空的。如果直接开始生成，模型不知道前面的历史信息。
        # 解决：我们需要把已知的历史帧 (initial_latent) 先“过一遍”模型，存入 Cache。

        current_start_frame = 0
        if initial_latent is not None:
            # 设置 timestep 为 0。
            # 在 Diffusion 中，t=0 意味着“没有噪声”，即清晰图像。
            # 我们告诉模型：“嘿，这是完美的历史数据，请记住它。” (Teacher Forcing)
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            
            if self.independent_first_frame:
                 # (处理第一帧独立的特殊逻辑)
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # 正常逻辑：计算有多少个历史块需要预填
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            # 遍历每一个历史块
            for _ in range(num_input_blocks):
                # 切片：取出当前这几帧历史数据
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                
                # 填入 output 画布 (虽然是输入，但也放在 output 里保持完整)
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                
                # === 关键动作 (Key Action) ===
                # 运行 Generator。注意！这里我们【不接收返回值】！
                # 我们完全不在乎它的输出是什么。
                # 我们只在乎它的【副作用】：更新 self.kv_cache1。
                # 它会计算 current_ref_latents 的 K, V 并追加写入缓存。
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0, # t=0, 强制 Teacher Forcing
                    kv_cache=self.kv_cache1, # 传入 Cache 对象，内部会自动写入
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                
                # 移动指针，处理下一个块
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # =================================================================================
        # 🔥🔥 Step 3: 核心时序去噪循环 (Temporal Denoising Loop)
        # =================================================================================
        # 这是整个推理过程的心脏，实现了 Self-Forcing 机制。
        # 准备一个列表，比如 [3, 3, 3, 3, 3, 3, 3]，表示每个 Block 包含的帧数
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
            
        # === 外层循环：遍历每一个 Block (按时间顺序) ===
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            # 1. 准备噪声 inputs
            # 从原始的大 noise tensor 中切出当前这几帧的噪声
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # === Step 3.1: 内层循环：空间去噪 (Spatial Denoising) ===
            # 在当前这个时间块内，从纯噪声逐步还原出图像。
            # denoising_step_list 可能是 [1000, 750, 500, 250]
            for index, current_timestep in enumerate(self.denoising_step_list):
                print(f"current_timestep: {current_timestep}")
                
                # 构造 timestep 张量
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    # Case 1: 还没到最后一步
                    # 运行生成器预测去噪结果 (denoised_pred)
                    # 【重要】这里传入了 kv_cache1，因为我们要读取之前的历史信息！
                    # 但是，因为 timestep > 0 (还没彻底干净)，主要目的是利用 Cache，
                    # 此时模型【不会】把当前这几帧写入 Cache。
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    
                    # 调度器更新 (Step): 加点噪声准备下一次迭代
                    # 类似于 x_{t-1} = x_0 + noise
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # Case 2: 最后一步
                    # denoised_pred 就是我们终于生成好的 clean latents
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: 记录结果
            # 到这里，当前 Block (3帧) 已经完全生成完毕了！保存它。
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # =================================================================================
            # 🔥 Step 3.3: Self-Forcing 更新 (Critial Step)
            # =================================================================================
            # 此时，KV Cache 里还没有这 3 帧的信息（之前只是在 Read，没 Write）。
            # 为了让下一个 Block 能参考这 3 帧，我们必须做一次额外的转发来“记录”它们。
            
            # 1. 构造一个极小的 timestep (通常是 args.context_noise 或 0)
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            
            # 2. 再跑一次模型！这是一个额外的 overhead，但必不可少。
            # 这次输入的是刚刚生成的完美结果 (denoised_pred)。
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                # 【机制详解】
                # 当 Generator 发现 timestep ≈ 0 时，它会启动 update logic。
                # 它会计算当前这 3 帧的 Key 和 Value，并将其 boost (追加) 到 kv_cache1 的末尾。
                # 这样，下一次大循环 (Next Block) 就能看到这段历史了。
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # =================================================================================
        # Step 4: 解码 (Video Decoding)
        # =================================================================================
        # 此时 output 包含了所有生成的 Latent Frames。我们需要用 VAE 把它们变回人眼可看的像素。
        # use_cache=False: 这里不使用 VAE Cache，直接解码。
        video = self.vae.decode_to_pixel(output, use_cache=False)
        
        # 归一化: [-1, 1] -> [0, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        初始化 Wan 模型的 Per-GPU KV 缓存。
        
        策略：静态预分配 (Static Pre-allocation)。
        我们不使用 python list 动态 append，因为那会导致大量的显存碎片 (Fragmentation)。
        相反，我们根据最大可能的 Token 数量，一次性申请一个巨大的 Tensor 矩阵。
        
        kv_cache_size = 32760 对应了约 21 帧 * 1560 tokens 的容量。
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            # kv_cache_size = 滑动窗口帧数 * 每帧 Token 数 (1560)
            # 这种模式下 Cache 像循环队列，只存储最近 N 帧的信息
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # 32760 = 21 帧 * 1560 Tokens/帧
            # 21 帧是 Wan2.1 标准的生成长度，这里一次性预分配显存防止碎片
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            # 张量形状解释 [batch_size, kv_cache_size, 12, 128]:
            # 1. batch_size: 样本数量
            # 2. kv_cache_size: 最大序列长度 (总 Token 容量)
            # 3. 12: 注意力头数 (Wan2.1-1.3B 规格)
            # 4. 128: 每个头的维度 (12 * 128 = 1536，即模型的主隐藏层维度)
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        初始化 Cross-Attention (文本-视频注意力) 缓存。
        
        用途：缓存文本 Prompt 的 Attention 结果。
        原因：因为文本 Prompt 在整个视频生成过程中是不变的 (Temporal Invariant)。
             如果没有这个 Cache，每一帧 (每个 Block) 都要重新计算一次 Text-to-Image Attention，
             这会浪费大量的重复算力。有了它，只需要计算一次。
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            # 张量形状解释 [batch_size, 512, 12, 128]:
            # 1. batch_size: 样本数量
            # 2. 512: 文本最大 Token 长度 (对应 T5 Encoder 的 seq_len)
            # 3. 12: 注意力头数
            # 4. 128: 每个头的维度 (12 * 128 = 1536)
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
