# CausalInferencePipeline 源码详解

[pipeline/causal_inference.py](file:///home/node1/Desktop/n1/ai/videogen/Self-Forcing/pipeline/causal_inference.py)

该文件实现了 Self-Forcing 的**因果推理管线**，是项目中最核心的部分。它负责协调文本编码、潜在空间生成（通过自回归方式）和 VAE 解码。

---

## 1. 导入与类定义 (L1-46)

### 导入依赖
```python
from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
```
- `WanDiffusionWrapper`: 扩散生成器
- `WanTextEncoder`: 文本编码器
- `WanVAEWrapper`: 变分自编码器
- `DynamicSwapInstaller`: 动态显存管理（用于低显存模式）

### 初始化函数 `__init__`

```python
class CausalInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, generator=None, text_encoder=None, vae=None):
        super().__init__()
        # 1. 初始化模型
        self.generator = WanDiffusionWrapper(..., is_causal=True)
        self.text_encoder = WanTextEncoder()
        self.vae = WanVAEWrapper()

        # 2. 初始化超参数
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        # 常见 step_list: [1000, 750, 500, 250] (4步推理)

        # 3. 对应 Wan2.1 的配置
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560  # 每帧 1560 tokens
        self.kv_cache1 = None         # KV 缓存容器
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)  # 默认 3 帧一块
```

**关键点**：
- `is_causal=True`: 启用因果模式，支持 KV 缓存。
- `num_frame_per_block`: 控制每次生成多少帧（Self-Forcing 默认为 3）。
- `denoising_step_list`: 只有 4 步，因为模型经过了 DMD 蒸馏。

---

## 2. 核心推理函数 `inference` 超级深度解析 (L47-276)

⚠️ **注意**：本章节将进行保姆级的详细拆解，确保每一个变量、每一个循环、每一个条件的意图都解释得清清楚楚。

### 2.0 宏观逻辑

在深入代码之前，必须先理解 **Block-Based Autoregressive Generation（基于块的自回归生成）** 的思想：
1.  **分块**：我们不会一次性生成所有视频帧（因为显存不够，且难以维持长时序一致性）。我们将视频切分为一个个 **Block**，每个 Block 通常包含 3 帧。
2.  **接龙**：我们先生成第 1 个 Block（0-3帧）。生成好后，把它的特征固定下来（存入 KV Cache）。
3.  **依赖**：生成第 2 个 Block（3-6帧）时，模型会“回头看”第 1 个 Block 的特征，从而保证连贯性。
4.  **循环**：如此往复，直到生成所有帧。

### 2.1 函数签名与输入参数

```python
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
```

*   **`noise`**: `[Batch, Total_Frames, Channels, Height, Width]`
    *   这是扩散模型的起始点（纯高斯噪声）。
    *   **重要**：它的形状直接决定了我们要生成多长的视频（由 `Total_Frames` 决定）。
*   **`text_prompts`**: 用户输入的文本提示。
*   **`initial_latent`**: `[Batch, Input_Frames, C, H, W]`
    *   **I2V (图生视频) 模式**：这里传入首帧图像的 Latent。
    *   **Video Extension (视频扩充) 模式**：这里传入前一段视频的 Latent。
    *   **T2V (文生视频) 模式**：这里是 `None`。
*   **`return_latents`**: 调试用，如果为 `True`，除了返回最终像素视频，还返回 Latent 张量。

### 2.2 变量初始化与分块计算 (L72-83)

我们需要计算“总共要循环多少次”，即有多少个 Block。

```python
        # 获取总帧数
        batch_size, num_frames, num_channels, height, width = noise.shape
        
        # 计算 Block 数量
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            #这是最常见的情况
            # 假设 num_frames=21, per_block=3, 则 num_blocks = 7
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # 这是一个极少用的测试分支，第一帧独立生成，不用管它
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
```

### 2.3 文本编码 (L84-86)

```python
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
```
*   调用 T5 Encoder，把文本变成 Embeddings。
*   这些 Embeddings 会一直复用，指导每一个 Block 的生成。

### 2.4 输出容器准备 (L92-96)

```python
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
```
*   创建一个全零的张量作为画布。
*   最终生成的每一帧都会被“填”进这个 `output` 里。

---

### 🔥 2.5 步骤 1: KV Cache 的初始化 (L111-133)

这是因果推理（Causal Inference）的基础设施建设。

```python
        if self.kv_cache1 is None:
            # === Case A: 第一次运行 ===
            # 分配显存。kv_cache1 是一个列表，长度等于 Transformer 层数 (30)。
            # 每一层包含一个字典：{'k': ..., 'v': ..., 'global_end_index': ...}
            self._initialize_kv_cache(...)
            self._initialize_crossattn_cache(...)
        else:
            # === Case B: 显存复用 ===
            # 如果之前的推理已经分配过 cache，我们直接复用物理显存，
            # 只是把“指针”(global_end_index) 归零。
            # 这是为了极致的性能优化，避免反复 malloc/free 导致显存碎片。
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache1)):
                 # 把“写入位置”指针归零，相当于清空了内容，但没释放内存
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor([0], ...)
```

---

### 🔥 2.6 步骤 2: 预填充 (Prefill Context) (L134-170)

**场景**：假设我们要基于一张图生成视频 (I2V)，或者基于前 3 秒生成后 3 秒。
**问题**：KV Cache 现在是空的（或已归零）。模型如果直接开始生成后续帧，它不知道前面的历史信息。
**解决**：我们需要把已知的历史帧（`initial_latent`）先“过一遍”模型，把它们的特征存进 KV Cache。

```python
        current_start_frame = 0
        if initial_latent is not None:
             # 设置 timestep 为 0。
             # 在 Diffusion 中，t=0 意味着“没有噪声”，即清晰图像。
             # 我们告诉模型：“嘿，这是完美的历史数据，请记住它。”
            timestep = torch.ones([batch_size, 1], ...) * 0

            # 遍历所有输入的历史块
            for _ in range(num_input_blocks):
                # 切片取出当前要处理的那几帧历史数据
                current_ref_latents = initial_latent[:, current_start:current_end]
                
                # 填入 output 画布
                output[:, current_start:current_end] = current_ref_latents
                
                # === 关键动作 ===
                # 运行 Generator。注意这里没有接收返回值！
                # 我们不在乎它的输出，只在乎副作用：更新 self.kv_cache1
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0, # t=0, 强制 Teacher Forcing
                    kv_cache=self.kv_cache1, # 传入 Cache 对象，内部会自动写入
                    crossattn_cache=self.crossattn_cache,
                    # ...
                )
                # 移动指针
                current_start_frame += self.num_frame_per_block
```

---

### 🔥🔥 2.7 步骤 3: 核心时序去噪循环 (L176-245)

这是整个推理过程的心脏，实现了 **Self-Forcing** 机制。

它有两层循环：
1.  **Block 循环**：按时间顺序，一段一段生成视频。
2.  **Denoising 循环**：在每一段内部，从噪声逐步还原为图像。

```python
        # 准备一个列表，比如 [3, 3, 3, 3, 3, 3, 3]
        all_num_frames = [self.num_frame_per_block] * num_blocks

        # === 外层循环：遍历每一个 Block ===
        for current_num_frames in all_num_frames:
            
            # 1. 准备噪声
            # 取出当前这 3 帧对应的纯噪声
            noisy_input = noise[:, current_start_frame : current_start_frame + 3]

            # === 内层循环：遍历去噪步数 (例如 4 步) ===
            # denoising_step_list 可能是 [1000, 750, 500, 250]
            for index, current_timestep in enumerate(self.denoising_step_list):
                
                # 2. 构造 timestep 张量
                timestep = torch.ones(..., dtype=torch.int64) * current_timestep

                # 3. 预测噪声/原图 (Model Prediction)
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input, # 当前充满噪声的 3 帧
                    timestep=timestep,
                    kv_cache=self.kv_cache1, # 【这是关键】
                    # 这里传入 kv_cache1，模型会读取之前的历史信息！
                    # 但是！因为正在去噪中，结果还不确定，所以模型【不会】
                    # 把当前这 3 帧写入 Cache，只会读取前面的。
                    # ...
                )

                # 4. 调度器更新 (Step)
                if index < len(self.denoising_step_list) - 1:
                    # 如果还没到最后一步，就加点噪，准备下一次迭代
                    # 类似于 x_{t-1} = x_0 + noise
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred, ..., next_timestep
                    )
                else:
                    # 最后一步，denoised_pred 就是我们终于生成好的 clean latents
                    pass

            # -------------------------------------------------------
            #  到这里，当前 Block (3帧) 已经完全生成完毕了！
            #  但是，KV Cache 里还没有这 3 帧的信息。
            #  为了让下一个 Block 能参考这 3 帧，我们必须把它们存进去。
            # -------------------------------------------------------

            # 5. 记录结果
            output[:, current_start : current_end] = denoised_pred

            # 6. Self-Forcing 更新 (Step 3.3)
            # 再次构造一个 t=0 (或极小值) 的 timestep
            context_timestep = torch.ones_like(timestep) * self.args.context_noise 

            # 再跑一次模型！这是一次额外的 overhead。
            # 这次输入的是刚刚生成的 perfect result (denoised_pred)。
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1, 
                # 【注意】这一次调用，Generator 内部逻辑会检测到输入是 clean 的
                # (或者根据内部标志位)，它会将这 3 帧的 Key/Value 计算出来，
                # 并追加写入到 kv_cache1 的末尾！！！
                # ...
            )

            # 7. 移动指针，处理下一个 Block
            current_start_frame += current_num_frames
```

### 2.8 步骤 4: 解码 (Video Decoding) (L254-256)

此时 `output` 包含了所有生成的 Latent Frames。我们需要用 VAE 把它们变回人眼可看的像素。

```python
        # VAE 解码: Latent -> Pixel
        # output shape: [Batch, Frames, Channels, Height_Latent, Width_Latent]
        video = self.vae.decode_to_pixel(output, use_cache=False)
        
        # 归一化: [-1, 1] -> [0, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
```

---

## 3. 辅助函数解析 (L278-313)

### `_initialize_kv_cache`
*   **分配**：预先分配好能容纳整个视频所有帧的巨大 Tensor。
    *   例如：`Size = 32760` (对应 approx 21 帧 * 1560 tokens)。
*   **好处**：相比于 python list `append` 或者 torch `cat`，这种**静态预分配**大大减少了显存碎片，这对显存极其紧张的视频生成任务至关重要。

### `_initialize_crossattn_cache`
*   **用途**：缓存文本 Prompt 的 Attention 结果。
*   **原因**：文本 Prompt 从头到尾是不变的。如果没有这个 Cache，每一帧都要重新计算一遍 Text-to-Image Attention，浪费算力。有了它，只需要计算一次。

---

## 总结

这就是 Self-Forcing 的精髓：
1.  **Split**: 把长难任务拆成短任务 (Block)。
2.  **Generate**: 每一个短任务独立去噪生成。
3.  **Force**: 生成完后，强制把结果当作“真值”写入记忆 (KV Cache)。
4.  **Next**: 下一个短任务读取记忆，基于历史继续生成。

这种机制完美解决了长视频生成中的显存爆炸问题（不需要一次性把所有帧放入显存进行 Attention），同时也保证了视频在时间上的连贯性。
