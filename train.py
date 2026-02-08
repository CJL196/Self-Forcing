"""
================================================================================
train.py - Self-Forcing 训练入口文件
================================================================================

【文件作用】
这是整个 Self-Forcing 训练系统的入口点。当你运行训练命令时，Python 解释器首先执行这个文件。
它的主要职责是：
1. 解析命令行参数（如配置文件路径、日志目录等）
2. 加载并合并配置文件
3. 根据配置选择合适的 Trainer 类
4. 启动训练循环

【执行命令示例】
torchrun --nnodes=8 --nproc_per_node=8 train.py \
    --config_path configs/self_forcing_dmd.yaml \
    --logdir logs/self_forcing_dmd \
    --disable-wandb

【代码执行流程】
main() 函数会：
1. 解析命令行参数
2. 加载 configs/default_config.yaml（默认配置）
3. 加载 configs/self_forcing_dmd.yaml（特定配置）
4. 合并两个配置（特定配置覆盖默认配置）
5. 根据 config.trainer == "score_distillation" 选择 ScoreDistillationTrainer
6. 调用 trainer.train() 开始训练
================================================================================
"""

# ================================================================================
# 导入部分
# ================================================================================

import argparse  # Python 标准库，用于解析命令行参数
import os  # Python 标准库，用于文件路径操作
from omegaconf import OmegaConf  # OmegaConf 是一个强大的配置管理库，支持 YAML 文件加载和配置合并
import wandb  # Weights & Biases，一个流行的机器学习实验跟踪工具

# 从 trainer 包中导入四种不同的 Trainer 类
# 根据配置文件中的 trainer 字段，会选择其中一个来执行训练
from trainer import DiffusionTrainer, GANTrainer, ODETrainer, ScoreDistillationTrainer
# - DiffusionTrainer: 标准 Diffusion 训练
# - GANTrainer: 使用 GAN 损失的训练
# - ODETrainer: ODE (常微分方程) 回归训练
# - ScoreDistillationTrainer: 分数蒸馏训练 ← Self-Forcing with DMD 使用的就是这个！


def main():
    """
    主函数 - 训练入口
    
    这个函数是整个训练流程的起点。它会：
    1. 解析命令行参数
    2. 加载和合并配置
    3. 创建相应的 Trainer
    4. 启动训练
    """
    
    # ================================================================================
    # Step 1: 创建命令行参数解析器
    # ================================================================================
    # argparse.ArgumentParser() 创建一个参数解析器对象
    # 它可以自动解析命令行中 --xxx 形式的参数
    parser = argparse.ArgumentParser()
    
    # --config_path: 配置文件路径（必需参数）
    # 例如: --config_path configs/self_forcing_dmd.yaml
    # 这个配置文件包含了训练的所有超参数设置
    parser.add_argument("--config_path", type=str, required=True)
    
    # --no_save: 如果设置，则不保存模型检查点
    # action="store_true" 意味着这是一个开关参数，出现则为 True，不出现则为 False
    # 主要用于调试，避免生成大量的检查点文件
    parser.add_argument("--no_save", action="store_true")
    
    # --no_visualize: 如果设置，则不生成可视化结果（如生成的视频样本）
    # 可以加速训练，因为视频解码很耗时
    parser.add_argument("--no_visualize", action="store_true")
    
    # --logdir: 日志和模型保存的目录路径
    # 例如: --logdir logs/self_forcing_dmd
    # 训练过程中的检查点会保存在 {logdir}/checkpoint_model_000050/ 等子目录
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    
    # --wandb-save-dir: Weights & Biases 日志保存目录
    # WandB 会在本地缓存一些数据，这个参数指定缓存位置
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    
    # --disable-wandb: 禁用 Weights & Biases 实验跟踪
    # 如果你没有 WandB 账号或不想记录实验，可以添加这个参数
    parser.add_argument("--disable-wandb", action="store_true")

    # 解析命令行参数
    # args 是一个 Namespace 对象，包含了所有解析后的参数
    # 例如: args.config_path = "configs/self_forcing_dmd.yaml"
    args = parser.parse_args()

    # ================================================================================
    # Step 2: 加载并合并配置文件
    # ================================================================================
    
    # OmegaConf.load() 从 YAML 文件加载配置
    # 用户指定的配置文件（如 self_forcing_dmd.yaml）包含特定实验的设置
    config = OmegaConf.load(args.config_path)
    
    # 加载默认配置文件
    # default_config.yaml 包含所有参数的默认值
    # 这样用户的配置文件只需要覆盖需要修改的参数
    default_config = OmegaConf.load("configs/default_config.yaml")
    
    # OmegaConf.merge() 合并两个配置
    # 规则：后面的配置会覆盖前面的配置
    # 所以 config（用户配置）会覆盖 default_config（默认配置）中的同名字段
    config = OmegaConf.merge(default_config, config)
    
    # 将命令行参数中的布尔开关添加到配置对象中
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # ================================================================================
    # Step 3: 设置配置的附加字段
    # ================================================================================
    
    # 从配置文件路径中提取配置名称
    # os.path.basename() 获取文件名（如 "self_forcing_dmd.yaml"）
    # .split(".")[0] 去掉扩展名，得到 "self_forcing_dmd"
    # 这个名字会用于 WandB 实验命名和日志目录
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    
    # 将其他命令行参数也添加到配置中
    config.logdir = args.logdir  # 日志目录
    config.wandb_save_dir = args.wandb_save_dir  # WandB 缓存目录
    config.disable_wandb = args.disable_wandb  # 是否禁用 WandB

    # ================================================================================
    # Step 4: 根据配置选择并实例化 Trainer
    # ================================================================================
    # config.trainer 字段决定使用哪种训练器
    # 这个字段在配置文件中定义，例如 self_forcing_dmd.yaml 中设置了 trainer: score_distillation
    
    if config.trainer == "diffusion":
        # 标准 Diffusion 训练器
        # 使用传统的扩散模型训练方法
        trainer = DiffusionTrainer(config)
    elif config.trainer == "gan":
        # GAN 训练器
        # 使用对抗训练（判别器 + 生成器）
        trainer = GANTrainer(config)
    elif config.trainer == "ode":
        # ODE 回归训练器
        # 用于 ODE 初始化阶段的训练
        trainer = ODETrainer(config)
    elif config.trainer == "score_distillation":
        # 分数蒸馏训练器 ← Self-Forcing with DMD 使用的就是这个！
        # ScoreDistillationTrainer 定义在 trainer/distillation.py 中
        # 它实现了 DMD (Distribution Matching Distillation) 损失
        # 以及 Self-Forcing 的核心逻辑（训练时模拟推理过程）
        trainer = ScoreDistillationTrainer(config)
    
    # ================================================================================
    # Step 5: 开始训练
    # ================================================================================
    # 调用 trainer.train() 方法开始训练循环
    # 这个方法会一直运行，直到满足停止条件（如达到最大步数）
    # 在 trainer/distillation.py 中可以看到 train() 方法的具体实现
    trainer.train()

    # ================================================================================
    # Step 6: 清理 WandB
    # ================================================================================
    # wandb.finish() 告诉 WandB 实验结束了
    # 这会上传所有待处理的日志并关闭连接
    # 即使 disable_wandb=True，调用这个也是安全的（不会报错）
    wandb.finish()


# ================================================================================
# Python 入口点
# ================================================================================
# 这是 Python 的标准入口点模式
# 当直接运行这个文件时（而不是被 import），__name__ 会等于 "__main__"
# 这时就会调用 main() 函数

if __name__ == "__main__":
    main()
