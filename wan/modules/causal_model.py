from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist

# Wan 1.3B 模型具有特殊的通道/头配置，需要启用 max-autotune 才能配合 flexattention 正常工作
# 参考 https://github.com/pytorch/pytorch/issues/133254
# 对于其他模型，可以改回默认设置
# flex_attention 是 PyTorch 的一个高性能注意力机制实现，通过 torch.compile 进行编译优化
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """
    应用因果旋转位置编码 (Causal RoPE)。
    
    Args:
        x (Tensor): 输入张量
        grid_sizes (Tensor): 网格尺寸 (frames, height, width) 的列表
        freqs (Tensor): 预计算的频率张量
        start_frame (int): 起始帧索引，用于处理时间维度的位置编码偏移
        
    Returns:
        Tensor: 应用 RoPE 后的张量
    """
    n, c = x.size(2), x.size(3) // 2

    # 分割频率，分别对应时间(temporal)、高度(height)、宽度(width)维度的编码
    # 这里的分割比例基于 RoPE 的设计，通常将维度分配给这三个轴
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # 遍历批次中的每个样本
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # 预计算乘数
        # 将输入 x 转换为复数形式，以便进行旋转操作
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        
        # 构建当前样本的频率张量
        # freqs[0] 对应时间维度，根据 start_frame 进行偏移切片
        # freqs[1] 对应高度维度
        # freqs[2] 对应宽度维度
        # 通过 view 和 expand 广播到 (f, h, w) 形状，然后拼接
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # 应用旋转位置编码：复数乘法
        # 之后转回实数形式并展平
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
         # 将处理后的序列部分与剩余部分（如果有）拼接
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # 添加到结果列表
        output.append(x_i)
    return torch.stack(output).type_as(x)


class CausalWanSelfAttention(nn.Module):
    """
    因果 Wan 自注意力模块 (Causal Wan Self-Attention)。
    支持全局注意力和局部窗口注意力，以及 KV 缓存机制。
    """

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size # 局部注意力窗口大小，-1 表示全局注意力
        self.sink_size = sink_size # 注意力汇聚点 (Attention Sink) 大小
        self.qk_norm = qk_norm # 是否对 Query 和 Key 进行归一化
        self.eps = eps
        # 计算最大注意力大小，用于缓存管理
        # 1560 是每个潜在帧(latent frame)的 token 数量 (例如 patch size 相关)
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # 定义层
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        # 如果启用 QK 归一化，则使用 RMSNorm，否则使用 Identity
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): 输入特征，形状 [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): 每个序列的长度，形状 [B]
            grid_sizes(Tensor): 网格尺寸，形状 [B, 3]，第二维包含 (F, H, W)
            freqs(Tensor): RoPE 频率，形状 [1024, C / num_heads / 2]
            block_mask (BlockMask): flex_attention 使用的块掩码
            kv_cache (dict): KV 缓存字典，用于推理加速
            current_start (int): 当前处理序列的起始位置
            cache_start (int): 缓存的起始位置
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # Query, Key, Value 投影函数
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # ==========================================
            # 1. 训练模式 (Training Phase)
            # ==========================================
            # 如果没有传入 kv_cache，说明我们处于训练阶段，需要一次性处理整个序列。
            
            # 判断是否是 Teacher Forcing (TF) 训练模式
            # 在某些训练策略中（如 Self-Forcing），输入序列的长度是 seq_lens[0] 的两倍。
            # 这通常意味着序列被拼接成了 [Clean Version, Noisy Version] 或 [Context, Target]。
            # 我们需要识别这种情况并进行特殊处理。
            is_tf = (s == seq_lens[0].item() * 2)
            
            if is_tf:
                # --- Case A: Teacher Forcing 模式 ---
                # 将 Q 和 K 切分为两部分，分别对应序列的前半段和后半段
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                
                # 分别对两部分应用旋转位置编码 (RoPE)
                # 关键点：这里两部分使用的是相同的 grid_sizes 和 freqs。
                # 这意味着它们共享相同的"绝对位置"语义（例如都从 t=0 开始），
                # 而不是让后半段的时间戳接在前半段之后。这对于成对训练至关重要。
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                # 将编码后的两部分重新拼接回原始形状
                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                # --- Padding for FlexAttention ---
                # 为了充分利用 PyTorch flex_attention (基于 FlashAttention) 的底层内核优化性能，
                # 序列长度最好是 128 的倍数。这里计算需要填充 (pad) 多少个 token。
                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                
                # 对 Query 进行零填充
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                # 对 Key 进行零填充
                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                # 对 Value 进行零填充
                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                # --- 执行注意力计算 ---
                # 使用 flex_attention 计算自注意力。
                # 注意：flex_attention 期望输入形状为 [B, H, L, D]，而我们目前是 [B, L, H, D]，因此需要 transpose(2, 1)。
                # block_mask 用于处理因果关系和可能的特殊掩码逻辑。
                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1) # 计算完成后，切除末尾的填充部分，并转置回原始形状 [B, L, H, D]

            else:
                # --- Case B: 普通训练模式 ---
                # 直接对整个序列应用旋转位置编码 (RoPE)
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                #同样进行填充以适应 flex_attention 优化 (逻辑同上)
                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            # ==========================================
            # 2. 推理模式 (Inference / Generation Phase)
            # ==========================================
            # 传入了 kv_cache，说明我们正在进行自回归生成或分块推理。
            # 这里的核心挑战是：
            # 1. 增量计算 RoPE（需要正确的绝对位置）。
            # 2. 管理有限显存下的 KV Cache（滚动/驱逐机制）。
            
            # 计算单帧的 token 数量 (Height * Width)
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            # 计算当前输入块对应的起始帧索引，确保 RoPE 的绝对位置正确
            current_start_frame = current_start // frame_seqlen 
            
            # --- 应用因果 RoPE ---
            # 使用 causal_rope_apply 并传入 start_frame，
            # 这样即使我们只输入第 N 帧的数据，计算出的位置编码也是对应第 N 帧的绝对坐标。
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            # 计算当前步骤处理后的结束位置（绝对 token 索引）
            current_end = current_start + roped_query.shape[1]
            # Attention Sink 大小：通常保留序列开头的若干 token 不被驱逐，以稳定注意力机制
            sink_tokens = self.sink_size * frame_seqlen
            
            # 获取当前 KV Cache 的容量
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            
            # --- 缓存滚动与驱逐逻辑 (Cache Rolling & Eviction) ---
            # 检查是否需要驱逐旧的 token 以腾出空间。条件如下：
            # 1. 启用了局部注意力 (local_attn_size != -1)。
            # 2. 当前进度确实在向前推进 (current_end > global_end)。
            # 3. 空间不足：新加入的 token + 现有的有效 token 位置 > 缓存总容量。
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                
                # 计算需要驱逐多少个旧 token 才能放下新 token
                # Evicted = Total Needed - Capacity
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                
                # 计算需要保留并滚动的 token 数量
                # Rolled = Current End - Evicted - Sink Tokens
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                
                # 执行滚动操作：
                # 1. 保留 Sink Tokens 不动 (索引 0:sink_tokens)。
                # 2. 将中间一部分旧数据丢弃，将较新的数据 (从 sink+evicted 开始) 移到 Sink 之后。
                # 注意：使用 .clone() 避免原地操作时的内存重叠问题。
                
                # 滚动 Key 缓存
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # 滚动 Value 缓存
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                
                # 计算更新后的局部索引
                # local_end_index 会因为移位而变小
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                
                # 将本轮新生成的 Key/Value 填入缓存末尾
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # --- 缓存空间充足 ---
                # 不需要滚动，直接追加新的 Key/Value 到当前 local_end 之后
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            
            # --- 执行注意力计算 ---
            # 从缓存中取出 Key 和 Value 进行计算。
            # 关键点：使用切片 [max(0, local_end - max_attn):local_end]
            # 这确保了注意力只作用于有效的滑动窗口内 (Window Attention)，忽略掉缓存中可能残留的陈旧数据。
            x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )
            
            # --- 更新状态 ---
            # 更新全局进度 (global_end_index) 和当前缓存内的有效位置 (local_end_index)
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # 输出投影
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):
    """
    因果 Wan 注意力块 (Causal Wan Attention Block)。
    包含自注意力、交叉注意力和前馈网络 (FFN)。
    结构: Norm1 -> Self-Attn -> Norm3 -> Cross-Attn -> Norm2 -> FFN
    """

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # 定义各个层
        self.norm1 = WanLayerNorm(dim, eps)
        # 实例化自注意力模块
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        # 根据配置决定是否在交叉注意力前使用 Norm
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        # 实例化交叉注意力模块 (从 WAN_CROSSATTENTION_CLASSES 注册表中获取类型)
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        # 前馈网络：Linear -> GELU -> Linear
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # 调制参数 (Modulation Parameters)，用于自适应调节特征
        # 形状 [1, 6, dim]，分别用于调节 Self-Attn, Cross-Attn, FFN 的输入/输出
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): 输入特征，形状 [B, L, C]
            e(Tensor): 时间嵌入/条件嵌入，形状 [B, F, 6, C]
            seq_lens(Tensor): 序列长度，形状 [B]
            grid_sizes(Tensor): 网格尺寸，形状 [B, 3]
            freqs(Tensor): RoPE 频率
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # 解包调制参数 e，分为 6 份
        # squeeze(1) 后 e 变为 [B, 6, C], 与 self.modulation 相加
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        
        # --- 自注意力 (Self-Attention) ---
        # 1. Norm1
        # 2. 应用调制参数 e[0] (偏置) 和 e[1] (缩放)
        # 3. 调用 self_attn
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)

        # 残差连接，并应用输出调制 e[2]
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # 定义交叉注意力和 FFN 的组合函数
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            # --- 交叉注意力 (Cross-Attention) ---
            # x + CrossAttn(Norm3(x), context)
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            # --- 前馈网络 (FFN) ---
            # 1. Norm2
            # 2. 应用调制参数 e[3] (偏置) 和 e[4] (缩放)
            # 3. FFN
            # 4. 应用输出调制 e[5]
            # 5. 残差连接
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):
    """
    因果预测头 (Causal Head)。
    将 Transformer 的输出投影回原始视频空间 (Patch 反向操作前的特征空间)。
    """

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # 计算输出维度：patch体积 * 输出通道数
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # 调制参数，用于调节 Norm 后的特征
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): 输入特征，形状 [B, L1, C]
            e(Tensor): 时间/条件嵌入，形状 [B, F, 1, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # 应用调制参数
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        # Norm -> Scale(e[1]) -> Shift(e[0]) -> Linear Head
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan 扩散模型骨干网络 (Backbone)，支持文生视频 (T2V) 和 图生视频 (I2V)。
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        初始化扩散模型骨干。

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                模型变体 - 't2v' (文生视频) 或 'i2v' (图生视频)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                视频嵌入的 3D Patch 尺寸 (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                文本嵌入的固定长度
            in_dim (`int`, *optional*, defaults to 16):
                输入视频通道数 (C_in，通常是 Latent 通道数)
            dim (`int`, *optional*, defaults to 2048):
                Transformer 的隐藏层维度
            ffn_dim (`int`, *optional*, defaults to 8192):
                前馈网络 (FFN) 的中间维度
            freq_dim (`int`, *optional*, defaults to 256):
                正弦时间嵌入的维度
            text_dim (`int`, *optional*, defaults to 4096):
                文本嵌入的输入维度
            out_dim (`int`, *optional*, defaults to 16):
                输出视频通道数 (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                注意力头的数量
            num_layers (`int`, *optional*, defaults to 32):
                Transformer 块的数量
            local_attn_size (`int`, *optional*, defaults to -1):
                时间局部注意力的窗口大小 (-1 表示全局注意力)
            sink_size (`int`, *optional*, defaults to 0):
                注意力 Sink 的大小，在滚动 KV 缓存时保留前 `sink_size` 帧不变
            qk_norm (`bool`, *optional*, defaults to True):
                启用 Query/Key 归一化
            cross_attn_norm (`bool`, *optional*, defaults to False):
                启用交叉注意力归一化
            eps (`float`, *optional*, defaults to 1e-6):
                归一化层的 Epsilon 值
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # 嵌入层
        # 将输入视频 patch 投影到 dim 维度
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        # 文本嵌入投影
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        # 时间嵌入 MLP
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        # 时间投影，用于生成调制参数 (dim * 6)
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # 堆叠 Attention Blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # 输出头
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # 缓冲区 (使用 register_buffer 会导致 dtype 问题，因此手动管理)
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        # 预计算 RoPE 参数
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)), # 时间
            rope_params(1024, 2 * (d // 6)),     # 高度
            rope_params(1024, 2 * (d // 6))      # 宽度
        ],
            dim=1)

        # I2V 模式下的图像嵌入投影
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # 初始化权重
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        准备分块因果注意力掩码 (Block-wise Causal Attention Mask)。
        将 token 序列划分为 [1 latent frame] [1 latent frame] ... 的格式。
        使用 flexattention 构建掩码。
        """
        total_length = num_frames * frame_seqlen

        # 右填充到 128 的倍数，以优化 flex_attention 性能
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # 分块因果掩码：当前块内的元素可以关注之前所有块的元素以及当前块内的先前元素
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            # 记录每个块的结束位置
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                # 全局因果：kv_idx 必须在当前 query 所在块的结束位置之前，或者 query 关注自身
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                # 局部因果：kv_idx 必须在 [ends[q_idx] - local_window, ends[q_idx]) 范围内
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # 双向掩码示例

        # 创建 flex_attention BlockMask
        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        准备 Teacher Forcing 注意力掩码。
        序列包含 [Clean Frames] + [Noisy Frames]。
        Noisy Frames 需要关注 Clean Frames 作为上下文。
        """
        # 调试模式相关代码...
        DEBUG = False
        if DEBUG:
            num_frames = 9
            frame_seqlen = 256

        # 总长度包含两倍的帧数 (Clean + Noisy)
        total_length = num_frames * frame_seqlen * 2

        # 右填充对齐
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # Clean context frames 的结束位置数组
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # Noisy frames 的注意力区间定义：需要两个区间
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # 设置 Clean Frames 的注意力范围
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # 设置 Noisy Frames 的注意力范围
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # 1. 关注同一块内的 Noisy Tokens (自注意力)
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # 2. 关注之前块的 Clean Tokens (交叉上下文)
            # noise_context_starts[start:end] = 0 # 从 0 开始
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # 1. Clean Frames 的掩码：标准的块内因果
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # 2. Noisy Frames 的掩码：
            # C1: 关注当前的 Noisy 块
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            # C2: 关注之前的 Clean 上下文块
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            # ... 调试代码 (可视化掩码) ...

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        准备 I2V 模式的分块因果注意力掩码。
        格式：[1 latent frame] [N latent frame] ... [N latent frame]
        第一帧被单独分出来以支持 I2V 生成 (通常第一帧是条件图像)。
        """
        total_length = num_frames * frame_seqlen

        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # 特殊处理第一帧：单独成为一个块
        ends[:frame_seqlen] = frame_seqlen

        # 后续帧按 num_frame_per_block 分块
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                 # 局部注意力逻辑
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0
    ):
        r"""
        详情参见 CausVid 论文算法 2: https://arxiv.org/abs/2412.07772
        该函数将逐帧 (或逐块) 运行 num_frame 次。
        逐个处理潜在帧 (每个 1560 tokens)。
        核心推理函数：使用 KV Cache 进行高效的自回归/分块视频生成。
        
        工作流：
        1. 输入处理：接收当前的噪声块 x (例如 3 帧)。
        2. Embedding：将像素块转为 Patch Embeddings，将文本/时间条件转为向量。
        3. Transformer 循环：逐层通过 30 个 Block。
           - 每一层都会读取并更新 kv_cache (Self-Attention)。
           - 每一层都会读取 crossattn_cache (Text-Attention)。
        4. 输出：还原为像素空间的噪声预测。

        Args:
            x (List[Tensor]): 输入视频噪声列表。虽然是列表，但在推理时通常只有一个 Tensor。
                              形状: [[C_in, F, H, W]]。例如 [[16, 3, 60, 104]]。
            t (Tensor): 扩散时间步。形状 [B]。例如 [1] (Uniform timestep)。
            context (List[Tensor]): 文本条件的 Embeddings。形状 [[L, C]] -> [[512, 4096]]。
            seq_len (int): 模型通常支持的最大位置编码长度。例如 32760。
            clip_fea (Tensor, optional): (I2V专用) CLIP 图像特征。形状 [B, 257, 1024] 或类似。
            y (List[Tensor], optional): (I2V专用) 参考原图/视频的 Latent，用于拼接。形状同 x。
            kv_cache (List[dict]): 每一层的 KV 缓存对象。包含 'k', 'v' 矩阵。
            crossattn_cache (List[dict]): 每一层的 Cross-Attention 缓存。
            current_start (int): 当前处理的这些帧在通过 Patchify 展平后，在总序列中的起始 Token 索引。
                                 例如：第 0-2 帧对应 Token 0 ~ 4679。那么处理第 3-5 帧时，current_start=4680。
            cache_start (int): 缓存写入的起始位置索引 (通常与 current_start 相关，但在某些滑动窗口机制下可能不同)。

        Returns:
            Tensor: 模型预测的去噪输出 (Velocity/Noise)。形状 [B, C_out, F, H, W]。
        """
        # I2V 模式下的必要检查: 必须提供 CLIP 特征和参考图像 Latent
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        
        # 确保频率表 (用于 RoPE 位置编码) 在正确的设备上
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # === 1. 输入数据预处理 ===
        # 如果是 I2V 模式，我们将噪声 x 和参考图像 y 在通道维度拼接
        # x shape: [16, F, H, W], y shape: [16, F, H, W] -> cat -> [32, F, H, W]
        # 这里的 dim=0 指的是 Channel 维度 (因为输入列表里没有 Batch 维度)
        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # === 2. Patch Embedding (图像块编码) ===
        # 将视频像素/Latent 切分成小块 (Patch) 并映射为向量
        # 输入 u: [C, F, H, W] -> unsqueeze(0) -> [1, C, F, H, W]
        # 输出: [1, Dim, F, H/s, W/s] (s是patch_size)
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        
        # 记录每个样本在 Patchify 之后的网格形状 (Grid Size)，用于最后还原
        # shape: [B, 3] -> [F, H_grid, W_grid]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        
        # 展平与转置 (Flatten & Transpose)
        # 1. flatten(2): 将 [1, Dim, F, H_grid, W_grid] 展平为 [1, Dim, Sequence_Length]
        # 2. transpose(1, 2): 转换为 Transformer 标准格式 [1, Sequence_Length, Dim]
        x = [u.flatten(2).transpose(1, 2) for u in x]
        
        # 计算当前输入的实际 Token 长度
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        
        # 拼接 Batch (虽然推理时通常 list 只有一个元素)
        # x 最终形状: [Batch, Sequence_Length, Dim] (例如 [1, 4680, 1536])
        x = torch.cat(x)

        # === 3. Time Embedding (时间步编码) ===
        # 处理扩散模型的 Timestep t，将其映射为高维向量，用于控制去噪程度
        # sinusoidal_embedding_1d: 生成正弦位置编码 [B, 256]
        # time_embedding: MLP 层 [B, 256] -> [B, Dim]
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        
        # time_projection: 将时间特征投影并拆分为 6 份 (用于 AdaLN 调制机制)
        # e0 shape: [B, 6, Dim]
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # === 4. Context Embedding (文本条件编码) ===
        # 处理 T5 输出的文本特征 embeddings
        context_lens = None
        # 如果 context 长度不一，这里做 padding (推理时通常已经是 padded 的 [B, 512, 4096])
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # (I2V) 如果有 CLIP 图像特征，将其作为额外的 Token 拼接到文本特征前面
        # 类似于 "Look at this image" + "Text description"
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # [B, 257, Dim]
            context = torch.concat([context_clip, context], dim=1)

        # 准备传递给 Block 的参数字典
        kwargs = dict(
            e=e0,                     # 时间步调制参数
            seq_lens=seq_lens,        # 序列长度
            grid_sizes=grid_sizes,    # 原始网格尺寸 (用于 RoPE 3D 位置编码)
            freqs=self.freqs,         # 预计算的旋转位置编码频率
            context=context,          # 文本/条件特征
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        # 辅助函数：用于梯度检查点 (节省显存)
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # === 5. Transformer Layers Loop (核心循环) ===
        # 遍历所有 Transformer Block (通常是 30 层)
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # 训练模式/开启 checkpointing: 牺牲速度换显存
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start
                    }
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                # === 推理模式 ===
                # 更新当前层的 Cache 引用
                # block 是 CausalWanBlock，它内部会处理 KV Cache 的读取和写入
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],           # 这一层的 Self-Attention Cache
                        "crossattn_cache": crossattn_cache[block_index], # 这一层的 Cross-Attention Cache
                        "current_start": current_start,              # 当前 Token 在全局序列中的位置
                        "cache_start": cache_start
                    }
                )
                # 执行当前层的前向计算
                # x shape 保持不变: [B, Seq_Len, Dim]
                x = block(x, **kwargs)

        # === 6. 输出层 ===
        # Final Layer Norm & Linear Projection
        # e.unflatten... 是将原始的时间 embedding 作为额外的条件输入
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        
        # === 7. Unpatchify (还原) ===
        # 将序列向量还原回视频的空间结构
        # 输入: [B, Seq_Len, Dim]
        # 输出列表: [[C_out, F, H, W], ...]
        x = self.unpatchify(x, grid_sizes)
        
        # 堆叠回 Tensor
        # 最终输出: [B, C_out, F, H, W]
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        训练阶段的扩散模型前向传播。

        Args:
            x (List[Tensor]): 输入视频张量列表
            t (Tensor): 扩散时间步
            context (List[Tensor]): 文本嵌入
            seq_len (`int`): 最大序列长度
            clean_x (List[Tensor], optional): 干净的视频帧，用于 Teacher Forcing 构建 mask
            aug_t (Tensor, optional): 增强的时间步
            clip_fea (Tensor, optional): CLIP 特征
            y (List[Tensor], optional): 条件输入
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # 参数
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # --- 构建分块因果注意力掩码 ---
        if self.block_mask is None:
            if clean_x is not None:
                # 有 clean_x 表示正在进行 Self-Forcing / Teacher Forcing 训练
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                # 标准训练流程
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # --- 嵌入 ---
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        # 批次拼接
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # --- 时间嵌入 ---
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

        # --- 上下文嵌入 ---
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # --- 处理 Clean 数据 (用于 Self-Forcing) ---
        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            # 将 Clean 和 Noisy 数据拼接在一起
            x = torch.cat([clean_x, x], dim=1)
            
            # 处理 Clean 数据的对齐时间步
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # 构造参数
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        # 执行 Transformer Blocks
        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # 如果拼接了 Clean 数据，最后需要将其切分掉，只保留 Noisy 部分的输出
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # 输出头
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # 反 Patch 化
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        # 根据是否存在 kv_cache 自动分发到 推理 或 训练 逻辑
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        从 patch embeddings 重构视频张量。

        Args:
            x (List[Tensor]): Patch 特征列表
            grid_sizes (Tensor): 原始网格尺寸 [F, H, W]

        Returns:
            List[Tensor]: 重构的视频张量 [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # 1. 截取有效长度
            # 2. Reshape 为 (F, H, W, P_t, P_h, P_w, C)
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            # 3. 维度重排: 'fhwpqrc' -> 'cfphqwr' (Channel, F, P_t, H, P_h, W, P_w)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            # 4. Reshape 合并 Patch 维度到空间时间维度
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        使用 Xavier 初始化模型参数。
        """

        # 基础初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 初始化 Embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # 初始化输出层 (Bias 初始化为 0，防止训练初期产生大梯度)
        nn.init.zeros_(self.head.head.weight)
