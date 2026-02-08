"""
================================================================================
trainer/distillation.py - 分数蒸馏训练器
================================================================================

【文件作用】
这个文件定义了 ScoreDistillationTrainer (SDTrainer) 类。
它是 Self-Forcing with DMD (Distribution Matching Distillation) 算法的训练中枢。
负责：
1. 初始化分布式环境、模型、优化器
2. 管理数据加载（实际上只加载文本Prompt）
3. 执行交替训练循环（生成器 vs Critic）
4. 保存检查点和记录日志

【核心逻辑】
DMD 训练包含两个相互竞争的部分：
1. Generator (生成的视频): 试图欺骗 Critic，并匹配真实视频的分布
2. Critic (Fake Score): 试图区分生成的视频和真实视频（实际上是去噪）

训练循环中：
- 每 5 步训练一次生成器 (Generator)
- 每 1 步训练一次 Critic
================================================================================
"""

import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD
import torch
import wandb
import time
import os


class Trainer:
    """
    Score Distillation Trainer 类
    负责执行 DMD 算法的训练循环。
    """
    def __init__(self, config):
        """
        初始化 Trainer
        """
        self.config = config
        self.step = 0

        # ================================================================================
        # Step 1: 初始化分布式训练环境
        # ================================================================================
        
        # 允许 TF32 精度（在 Ampere 架构 GPU 上加速矩阵乘法）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # 启动分布式任务 (PyTorch Distributed Data Parallel)
        # 这会设置 RANK, WORLD_SIZE 等环境变量
        launch_distributed_job()
        
        # 获取当前进程的全局 rank (0 到 63) 和总进程数 (64)
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 设置精度：如果是混合精度训练，使用 BFloat16，否则使用 Float32
        # BFloat16 在保持数值范围的同时减少了显存占用，适合训练大模型
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0  # 只有 rank 0 负责打印日志和保存模型
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # 设置随机种子，确保可复现性
        if config.seed == 0:
            # 如果没指定种子，随机生成一个并广播给所有进程，确保大家用的一样
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        # 每个进程使用 seed + rank 作为种子，保证数据采样的随机性但又可控
        set_seed(config.seed + global_rank)

        # 只有主进程负责初始化 Weights & Biases
        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # ================================================================================
        # Step 2: 初始化模型 (DMD)
        # ================================================================================
        
        # 根据配置选择分布匹配损失类型
        # 这里我们关注 "dmd" (Self-Forcing with DMD)
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd":
            # 初始化 DMD 模型
            # 这是一个包含 Generator, Real Score(教师), Fake Score(Critic) 的复合模型
            # 详见 model/dmd.py
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # 此时模型还在 CPU 上或部分在 GPU 上，尚未被 FSDP 包装

        # 备份 Fake Score 的初始权重到 CPU (调试用)
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        # 使用 FSDP (Fully Sharded Data Parallel) 包装各个子模型
        # FSDP 会将模型参数切分到各个 GPU 上，极大地节省显存，从而能训练像 Wan2.1-14B 这样的大模型
        
        # 1. 包装生成器 (Generator) - 需要训练
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        # 2. 包装真实评分网络 (Real Score / Teacher) - 冻结不训练
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        # 3. 包装伪造评分网络 (Fake Score / Critic) - 需要训练
        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        # 4. 包装文本编码器 (Text Encoder) - 冻结不训练，可 Offload 到 CPU
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        # 可选：初始化 VAE (用于可视化)
        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # 初始化生成器优化器 (AdamW)
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # 初始化 Critic 优化器 (AdamW)
        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # ================================================================================
        # Step 3: 初始化数据加载器
        # ================================================================================
        
        # Self-Forcing 是 Data-Free 的，不需要视频数据
        # 所以这里的 TextDataset 仅加载文本 Prompt 列表
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            # 加载纯文本 Prompt
            dataset = TextDataset(config.data_path)
            
        # 分布式采样器，确保每个 GPU 拿到不同的数据子集
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
            
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size, # 每个 GPU 的 Batch Size (通常为 1)
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        
        # cycle() 函数让 dataloader 可以无限循环，不需要手动处理 epoch 结束
        self.dataloader = cycle(dataloader)

        # ================================================================================
        # Step 4: 此处省略 EMA (指数移动平均) 设置代码，这是为了让模型权重更平滑
        # ================================================================================
        
        # ... (EMA相关代码) ...
        # 计算需要进行 EMA 的参数
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
             if not p.requires_grad:
                 continue
             renamed_n = rename_param(n)
             self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
             print(f"Setting up EMA with weight {ema_weight}")
             # EMA_FSDP 是支持分布式并行的 EMA 实现
             self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        # ================================================================================
        # Step 5: 加载预训练权重 (ODE Initialization)
        # ================================================================================
        # Self-Forcing 需要从一个已经有一定生成能力的模型开始训练
        # generator_ckpt 通常是经过 ODE 初始化微调过的 Wan2.1 检查点
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            # 严格加载权重
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        # 如果还没到 EMA 开始的步数，先不启用 EMA 以节省计算
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        """
        保存模型检查点
        """
        print("Start gathering distributed model states...")
        # 从所有 GPU 收集完整的模型参数
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        """
        执行一步前向传播 + 反向传播
        参数:
            batch: 数据批次 (包含 prompts)
            train_generator: 布尔值，True表示训练生成器，False表示训练Critic
        """
        self.model.eval()  # 设置为 eval 模式，关闭 Dropout 带来的随机性

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: 获取文本 prompts
        text_prompts = batch["prompts"] # e.g., ["A cat running"]
        
        # Self-Forcing 目前主要做 T2V (文本生视频), 因此 I2V 相关变量设为 None
        if self.config.i2v:
            # ... (I2V 逻辑)
            pass
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape) # [1, 21, 16, 60, 104]
        image_or_video_shape[0] = batch_size

        # Step 2: 提取条件信息 (Text Embeddings)
        with torch.no_grad():
            # 使用 T5 编码器获取文本嵌入
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            # 获取负面提示的嵌入 (用于 CFG - Classifier Free Guidance)
            # 同样也是缓存起来，避免重复计算
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: 这里的 train_generator 开关决定了走哪条训练路径
        
        if train_generator:
            # ──────────────────────────────────────────────────
            # 路径 A: 训练生成器 (Generator)
            # ──────────────────────────────────────────────────
            # 调用 model/dmd.py 中的 generator_loss
            # 这会触发 Self-Forcing 的自回归生成过程
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            # 反向传播计算梯度
            generator_loss.backward()
            
            # 梯度裁剪 (防止梯度爆炸)
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            # ──────────────────────────────────────────────────
            # 路径 B: 训练 Critic (Fake Score)
            # ──────────────────────────────────────────────────
            # 调用 model/dmd.py 中的 critic_loss
            critic_log_dict = {}
            
            # train_generator=False 时，不一定要计算 Critic Loss，
            # 这里调用的是 self.model.critic_loss, 它内部会先生成假样本，再计算 Critic 的去噪 Loss
            critic_loss, critic_log_dict = self.model.critic_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            # 反向传播
            critic_loss.backward()
            
            # 梯度裁剪
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)

            critic_log_dict.update({"critic_loss": critic_loss,
                                    "critic_grad_norm": critic_grad_norm})

            return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        # ... (用于在验证阶段生成可视化的函数，这里略过) ...
        pass

    def train(self):
        """
        训练主循环
        """
        start_step = self.step

        while True:
            # 决定当前步骤是否训练生成器
            # dfake_gen_update_ratio=5, 意味着每 5 步只训练 1 次生成器
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # ──────────────────────────────────────────────────
            # 阶段 1: 训练生成器 (Generator) (如果轮到)
            # ──────────────────────────────────────────────────
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                
                # 执行前向+反向传播
                extra = self.fwdbwd_one_step(batch, True)
                
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                
                # 更新权重
                self.generator_optimizer.step()
                
                # 更新 EMA 权重
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # ──────────────────────────────────────────────────
            # 阶段 2: 训练 Critic (Fake Score) (每步都做)
            # ──────────────────────────────────────────────────
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            
            # 执行前向+反向传播
            extra = self.fwdbwd_one_step(batch, False)
            
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            
            # 更新权重
            self.critic_optimizer.step()

            # 步数加 1
            self.step += 1

            # ... (如果到了 EMA 启动步数，初始化 EMA) ...
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # ──────────────────────────────────────────────────
            # 阶段 3: 保存模型 (如果到了间隔)
            # ──────────────────────────────────────────────────
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # ──────────────────────────────────────────────────
            # 阶段 4: 记录日志 (WandBH等)
            # ──────────────────────────────────────────────────
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            # ... (垃圾回收和计时逻辑) ...
            if self.step % self.config.gc_interval == 0:
                # ...
                gc.collect()
                torch.cuda.empty_cache()

            # ...
