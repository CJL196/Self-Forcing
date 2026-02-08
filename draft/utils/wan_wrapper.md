# utils/wan_wrapper.py 详细代码解读

## 1. 文件概览

`utils/wan_wrapper.py` 文件是 Wan2.1 视频生成模型的核心封装脚本。它主要定义了三个关键的包装类，用于统一管理和调用模型的不同组件：

1.  **`WanTextEncoder`**: 负责处理文本提示词（Prompt），使用 T5 模型将文本转换为嵌入向量（Embeddings）。
2.  **`WanVAEWrapper`**: 负责视频数据的编码和解码，将像素空间的视频转换为潜在空间（Latent Space）的表示，及其逆过程。
3.  **`WanDiffusionWrapper`**: 这是核心类，封装了 Diffusion Policy（实际上是 Flow Matching）模型。它处理噪声预测、去噪过程、时间步（Timestep）管理以及因果推理（Causal Inference）相关的 KV Cache 逻辑。

该文件整合了 `wan.modules` 下的多个底层模块，并提供了便于推理脚本调用的高层接口。

---

## 2. 导入部分

```python
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
```

**解释**:
这部分导入了必要的 PyTorch 库和类型提示。
-   引入了自定义的调度器 `FlowMatchScheduler`，这是基于流匹配（Flow Matching）的生成过程的核心。
-   引入了 Wan 模型的各个子模块：Tokenizer、WanModel（DiT结构）、VAE、T5 文本编码器以及支持因果推理的 `CausalWanModel`。

---

## 3. 类 `WanTextEncoder` 详解

这个类封装了文本编码逻辑，使用的是谷歌的 UMT5-XXL 模型。

### `__init__` (初始化)

```python
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/", seq_len=512, clean='whitespace')
```

**解释**:
-   初始化时加载了 `umt5_xxl` 模型，强制设为评估模式 (`eval()`) 且不需要梯度 (`requires_grad_(False)`)，因为它只用于提取特征，不参与训练。
-   权重文件路径硬编码为 `wan_models/Wan2.1-T2V-1.3B/...`。
-   加载对应的 Huggingface Tokenizer，最大序列长度设为 512。

### `forward` (前向传播)

```python
    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        # ... 数据移至 GPU ...
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }
```

**解释**:
-   接收文本列表，进行分词（Tokenization）。
-   调用 `text_encoder` 获取文本的上下文嵌入（Context Embeddings）。
-   手动将 Padding 部分的嵌入置零，消除无效 Token 的影响。
-   返回包含 `prompt_embeds` 的字典，供 Diffusion 模型使用。

---

## 4. 类 `WanVAEWrapper` 详解

这个类封装了视频 VAE（Variational Autoencoder），用于将高维的视频像素压缩到低维的 Latent 空间。

### `__init__` (初始化)

```python
    def __init__(self):
        super().__init__()
        # ... (定义 mean 和 std 常量) ...
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)
```

**解释**:
-   定义了用于 Normalize（归一化）和 De-normalize（反归一化）的均值 `mean` 和标准差 `std`。这些是训练时统计得到的参数。
-   加载预训练的 VAE 模型，Latent 维度 (`z_dim`) 为 16。

### `encode_to_latent` (编码)

```python
    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        # ... (准备 scale 参数) ...
        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output
```

**解释**:
-   输入是像素级视频。
-   VAE 的 `encode` 方法同时执行编码和标准化操作。
-   **注意**: 最后有一个维度置换 `permute(0, 2, 1, 3, 4)`，将 `num_frames` 移到了通道维度之前。这就是为什么 Diffusionmodel处理的 Latent 形状通常是 `(B, F, C, H, W)`。

### `decode_to_pixel` (解码)

```python
    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        # ... (准备 scale 参数和选择解码函数) ...
        
        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        
        output = output.permute(0, 2, 1, 3, 4)
        return output
```

**解释**:
-   首先将维度换回 `(B, C, F, H, W)` 以符合 VAE 输入要求。
-   支持 `use_cache`，这对于长视频生成的显存优化很重要（特别是时序上的分块解码）。
-   解码后将像素值截断 (`clamp`) 到 `[-1, 1]` 区间。
-   最后再次换位，保持输出格式一致。

---

## 5. 类 `WanDiffusionWrapper` 详解 (重点)

这是本文件的核心，封装了主要的生成模型。由于非常重要，下面将对关键方法**逐行详细解释**。

### `__init__`

```python
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0
    ):
        super().__init__()

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                f"wan_models/{model_name}/", local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(f"wan_models/{model_name}/")
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()
```

**解释**:
-   **模型加载**: 支持加载标准 `WanModel` 或支持因果注意力的 `CausalWanModel`。`CausalWanModel` 通常用于自回归式的长视频生成（Self-Forcing 策略的核心）。
-   **Scheduler**: 使用 `FlowMatchScheduler`。Flow Matching 是 Diffusion Models 的一种更通用的形式，通过回归向量场来生成数据。
-   **timestep_shift**: 时间步偏移参数，用于调整噪声调度的分布，对生成质量有很大影响。
-   **seq_len**: 硬编码的一个序列长度参数，可能对应特定的 Latent 尺寸 `(21 * 60 * 104 / 4)` 左右。

### `adding_cls_branch` (逐行详解)

这个方法似乎是为了在基础模型上添加额外的分类分支，可能用于 Classifier-Free Guidance (CFG) 的变体，或者某种判别/增强任务。

```python
    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        # 定义一个分类预测分支 (MLP)
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104] (注释里的维度可能指 Flatten 后的某种状态，或者特定层的输出)
            # update: 输入维度是 atten_dim * 3 + time_embed_dim
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(), # 激活函数
            nn.Linear(atten_dim, num_class) # 输出分类 logits
        )
        self._cls_pred_branch.requires_grad_(True) # 该分支需要训练

        # 定义 Register Tokens (寄存器令牌)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True) # 需要训练

        # 定义 GAN Attention Blocks
        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock() # 可能是用于处理 Register tokens 的特定注意力块
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True) # 需要训练
        # self.has_cls_branch = True
```

**详细解读**:
此函数动态地向模型添加了三个新的可训练组件：
1.  `_cls_pred_branch`: 一个简单的 MLP 分类器。
2.  `_register_tokens`: 类似于 ViT 中的 Register Tokens，用于存储全局信息而不干扰图像 patch。
3.  `_gan_ca_blocks`: 一组注意力块，可能用于让这些 Register Tokens 与图像特征交互。
这表明该代码库不仅仅用于推理，还包含了一些实验性的训练或微调逻辑（比如引入 GAN 损失或分类引导）。

### `_convert_flow_pred_to_x0` (逐行详解)

这是一个通过 Flow Matching 预测值 `v` (flow_pred) 和当前噪声图 `x_t` 来反推原图 `x_0` 的数学变换函数。

```python
    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        ... 文档字符串 ...
        """
        # 1. 精度提升: 为了数值稳定性，将核心变量转为 float64 (double)
        original_dtype = flow_pred.dtype
        # map 函数将 flow_pred, xt, sigmas, timesteps 全部转为 double 并放到正确设备上
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        # 2. 查找当前 timestep 对应的 sigma (噪声强度)
        # timestep 是 [B] 大小的 Tensor，scheduler.timesteps 是预设的时间步列表
        # 计算差的绝对值，找到最近的索引
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        
        # 3. 获取对应的 sigma_t并调整形状以支持广播
        # reshape(-1, 1, 1, 1) 是为了匹配图像/Latent 的 (B, C, H, W) 维度
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)

        # 4. 核心公式: x0 = x_t - sigma_t * flow_pred
        # 这里的 flow_pred 通常预测的是 vector field v = dx/dt = x_1 - x_0 (在简单路径下)
        # 或者更准确地说，Flow Matching 中，v_t = x_1 - x_0 (如果 path 是直线的)
        # 这里的公式似乎假设了特定的 Flow Matching 参数化形式。
        x0_pred = xt - sigma_t * flow_pred
        
        # 5. 转回原始精度并返回
        return x0_pred.to(original_dtype)
```

**详细解读**:
此函数实现了 Flow Matching 的一步逆变换。在 Flow Matching 的 Conditional Flow Matching (CFM) 设置中，通常有一条直线路径 $x_t = (1 - (1-\sigma_{min})t)x_0 + t x_1$ (这里的 $t$ 定义可能略有不同，通常是 $t \in [0, 1]$)。
这里使用的公式 `x0 = xt - sigma_t * flow_pred` 暗示了 `flow_pred` $\approx (x_t - x_0) / \sigma_t$。这意味着模型预测的是指向 $x_t$ 的“速度”，或者说去噪的方向。通过减去这个速度乘以时间/噪声尺度，我们尝试还原出干净的图像 $x_0$。

### `_convert_x0_to_flow_pred` (逐行详解)

这是上一个函数的逆操作：已知原图 `x_0` 和噪声图 `x_t`，计算模型应该预测的目标 `flow_pred`。这通常用于计算 Loss 或 Teacher Forcing。

```python
    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        ...
        pred = (x_t - x_0) / sigma_t
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

        # 3. 核心公式: flow_pred = (xt - x0_pred) / sigma_t
        # 这是上一个公式的代数变换。
        flow_pred = (xt - x0_pred) / sigma_t

        return flow_pred.to(original_dtype)
```

### `forward` (逐行重点详解)

这是模型推理的主入口，处理各种复杂情况（KV Cache, Teacher Forcing, 分类分支等）。

```python
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
        # 1. 提取条件嵌入 (Prompt Embeddings)
        prompt_embeds = conditional_dict["prompt_embeds"]

        # 2. 处理时间步 (timestep)
        # WanDiffusionWrapper 初始化时设定了 self.uniform_timestep。
        # 如果是 causal 模式 (is_causal=True)，uniform_timestep 为 False，每个帧可能有不同的 timestep。
        # 如果不是 causal，则认为是一个整体去噪，取第一个时间步即可。
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        
        # 3. 分支逻辑：决定如何调用底层模型 self.model
        
        # 分支 A: 使用 KV Cache (通常用于自回归推理或长视频生成加速)
        if kv_cache is not None:
            # model call
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4), # 调整维度为 [B, C, F, H, W]
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,    # 传入缓存
                crossattn_cache=crossattn_cache,
                current_start=current_start, # 当前处理的起始位置
                cache_start=cache_start
            ).permute(0, 2, 1, 3, 4) # 调回维度 [B, F, C, H, W]
        
        else:
            # 分支 B: 提供 clean_x (通常用于 Teacher Forcing 或训练阶段)
            if clean_x is not None:
                # teacher forcing 模式
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4), # 传入目标原图，可能用于引导生成
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            
            # 分支 C: 正常推理或分类模式
            else:
                if classify_mode:
                    # 分类模式：不仅返回 flow_pred，还返回 logits (分类结果)
                    # 这里的 classify_mode 对应之前 adding_cls_branch 增加的分支
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens, # 传入寄存器令牌
                        cls_pred_branch=self._cls_pred_branch, # 传入分类头
                        gan_ca_blocks=self._gan_ca_blocks,     # 传入 GAN 注意力块
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    # 分支 D: 最基础的去噪推理
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len
                    ).permute(0, 2, 1, 3, 4)

        # 4. 转换预测结果：Flow -> X0
        # 模型输出的是 vector field (flow_pred)，需要转换为去噪后的图像 (pred_x0)
        # 这对于可视化 current guess 或计算某些 loss 很有用。
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1), # Flatten batch 和 frames 维度，因为它们是一起处理的
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2]) # 恢复原来的 [B, F] 结构

        # 5. 返回结果
        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0
```

### 其他方法

-   **`get_scheduler`**: 这是一个辅助方法，用于将 `SchedulerInterface` 的一些静态方法动态绑定到当前的 `self.scheduler` 实例上。这可能是为了让 `scheduler` 对象在使用时更像一个功能完备的工具类，拥有 `convert_x0_to_noise` 等便捷方法。
-   **`post_init`**: 初始化后的钩子，目前只调用了 `get_scheduler`。

---

## 总结

`utils/wan_wrapper.py` 是一个精心设计的适配器层。它没有实现 Attention 或 Convolution 等底层运算，而是专注于：
1.  **数据流管理**：在 Pixel Space (VAE) 和 Latent Space (Diffusion) 之间转换。
2.  **数学转换**：处理 Flow Matching 的 $x_0 \leftrightarrow v$ 转换。
3.  **复杂推理逻辑**：支持 Causal Attention 的 KV Cache 传递、Teacher Forcing 训练模式以及带辅助分类头的特殊推理模式。

这对于理解 `inference.py` 或 `causal_inference.py` 中的 `model(...)` 调用至关重要。
