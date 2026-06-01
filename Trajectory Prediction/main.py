# -*- coding: utf-8 -*-
"""
轨迹预测 —— 训练主脚本
======================
使用 Seq2Seq + 注意力模型进行无人机轨迹预测。
支持混合精度训练、warmup + 自适应学习率衰减、早停。
"""

import os
import sys
import torch
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm

# 将项目根目录加入搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mydataloader2 import StandardScaler, MyDataLoader, train_time_seq, \
    val_time_seq, lr_warmup
from seq2seq import Seq2Seq
from animation_marking import plot_loss_curve


# =============================================================================
# 配置
# =============================================================================

torch.set_float32_matmul_precision('medium')

# 数据路径
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "posearray.pkl"
)

# 超参数
BATCH_SIZE = 64
TRAIN_STEPS = 20
INPUT_SIZE = 3            # 编码器输入: x, y, z 位置
DE_INPUT_SIZE = 3         # 解码器输入: x, y, z 位置
HIDDEN_SIZE = 256
LINEAR_SIZE = 128         # 注意力输出后的线性层维度
OUTPUT_SIZE = 3           # 预测: x, y, z 位置
LEARNING_RATE = 0.005
NUM_LAYERS = 2
PRED_STEPS = 10           # 预测未来 10 步
TEST_NUM = 0
TRAIN_RATIO = 0.8
EPOCHS = 500
RNN_TYPE = "lstm"

# 训练控制
PATIENCE = 30             # 早停耐心值
WARMUP_EPOCHS = 5         # 学习率 warmup 轮数
SEED = 42

# =============================================================================
# 主程序
# =============================================================================

if __name__ == '__main__':
    # ---- 随机种子 ----
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ---- 设备 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch CUDA 版本: {torch.version.cuda}")
    print(f"CUDA 可用: {torch.cuda.is_available()}, 设备: {device}")

    # ---- 数据加载与标准化 ----
    with open(DATA_PATH, "rb") as f:
        trajectory_data = pickle.load(f)

    scaler = StandardScaler()
    scaler.fit(trajectory_data)
    trajectory_transformed = scaler.transform(trajectory_data)

    train_loader = MyDataLoader(trajectory_transformed, TRAIN_RATIO, TEST_NUM,
                                BATCH_SIZE, TRAIN_STEPS, PRED_STEPS)
    val_loader = MyDataLoader(trajectory_transformed, TRAIN_RATIO, TEST_NUM,
                              BATCH_SIZE, TRAIN_STEPS, PRED_STEPS, mode="val")

    # ---- 模型 ----
    net = Seq2Seq(INPUT_SIZE, DE_INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
                  NUM_LAYERS, LINEAR_SIZE, PRED_STEPS, TRAIN_STEPS,
                  RNN_TYPE).to(device)

    # ---- 损失函数 ----
    criterion = nn.MSELoss(reduction="none")

    # ---- 优化器与调度器 ----
    optimizer = torch.optim.AdamW(net.parameters(), LEARNING_RATE,
                                  weight_decay=1e-4)
    scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_warmup
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=2
    )

    # ---- 混合精度 ----
    scaler_amp = torch.amp.GradScaler('cuda')

    # ---- 训练循环 ----
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    stop_count = 0

    tqdm_iter = tqdm(range(EPOCHS))
    for epoch in tqdm_iter:
        current_lr = optimizer.param_groups[0]['lr']

        train_loss = train_time_seq(
            train_loader, net, epoch, HIDDEN_SIZE, NUM_LAYERS,
            criterion, optimizer, device, scaler_amp, RNN_TYPE
        )
        val_loss = val_time_seq(
            val_loader, net, HIDDEN_SIZE, NUM_LAYERS,
            criterion, device, RNN_TYPE
        ).cpu().item()
        train_loss = train_loss.cpu().item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 前 warmup_epochs 轮线性 warmup，之后按验证损失自适应衰减
        if epoch < WARMUP_EPOCHS:
            scheduler_warmup.step()
        else:
            scheduler.step(val_loss)

        # 早停逻辑
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_count = 0
            torch.save(net.state_dict(), 'time_seq_best.opt')
        else:
            stop_count += 1

        if stop_count >= PATIENCE:
            tqdm_iter.set_postfix(
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.9f}",
                early_stop=f"best={best_val_loss:.9f}"
            )
            print(f"\n早停于 epoch {epoch}，最佳 val_loss: {best_val_loss:.9f}")
            break

        print(f"epoch 结束: {epoch}")
        tqdm_iter.set_postfix(
            train_loss=f"{train_loss:.8f}",
            val_loss=f"{val_loss:.9f}",
            lr=f"{current_lr:.8f}"
        )

    # ---- 保存最终模型 ----
    torch.save(net.state_dict(), 'time_seq_final.opt')
    plot_loss_curve(train_losses, val_losses)
