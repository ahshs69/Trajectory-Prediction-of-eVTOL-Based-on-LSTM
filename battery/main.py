# -*- coding: utf-8 -*-
"""
电池 SOC 估计与工况分类 —— 训练主脚本
======================================
使用多任务 LSTM 模型同时预测电池 SOC 和识别机动段类别。
支持混合精度训练、早停和学习率衰减。
"""

import os
import sys
import torch
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm

# 将项目根目录加入搜索路径，以便导入根目录下的工具模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mydataloader2 import StandardScaler, MyDataLoader, train_battery_seq, \
    validate_battery_seq, lr_down
from SOC_lstm import SOC_LSTM
from animation_marking import plot_loss_curve


# =============================================================================
# 配置
# =============================================================================

torch.set_float32_matmul_precision('medium')

# 数据路径
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "dataset1.pkl"
)

# 超参数
BATCH_SIZE = 64
TRAIN_STEPS = 20
INPUT_SIZE = 3           # LSTM 输入: 电压、电流、温度
HIDDEN_SIZE = 256
LEARNING_RATE = 0.001655
NUM_LAYERS = 2
NUM_CLASSES = 4          # 机动段类别数
PRED_STEPS = 1           # SOC 预测未来 1 步
TEST_NUM = 60
TRAIN_RATIO = 0.8
EPOCHS = 500
RNN_TYPE = "lstm"

# 训练控制
PATIENCE = 80            # 早停耐心值
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
        battery_data = pickle.load(f)

    scaler = StandardScaler()
    scaler.battery_fit(battery_data)
    battery_transformed = scaler.battery_transform(battery_data)

    train_loader = MyDataLoader(battery_transformed, TRAIN_RATIO, TEST_NUM,
                                BATCH_SIZE, TRAIN_STEPS, PRED_STEPS)
    val_loader = MyDataLoader(battery_transformed, TRAIN_RATIO, TEST_NUM,
                              BATCH_SIZE, TRAIN_STEPS, PRED_STEPS, mode="val")

    # ---- 模型 ----
    net = SOC_LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)

    # 加载预训练权重（可选）
    # net.load_state_dict(
    #     torch.load("time_seq_best.opt", map_location=device, weights_only=False)
    # )

    # ---- 损失函数 ----
    criterion_soc = nn.MSELoss(reduction="none")
    criterion_cls = nn.CrossEntropyLoss()

    # ---- 优化器与调度器 ----
    optimizer = torch.optim.AdamW(net.parameters(), LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_down)

    # ---- 混合精度 ----
    scaler_amp = torch.amp.GradScaler('cuda')

    # ---- 训练循环 ----
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    stop_count = 0

    tqdm_iter = tqdm(range(EPOCHS))
    for epoch in tqdm_iter:
        epoch += 110  # 从预训练断点继续
        current_lr = optimizer.param_groups[0]['lr']

        train_loss = train_battery_seq(
            train_loader, net, epoch, HIDDEN_SIZE, NUM_LAYERS,
            criterion_soc, criterion_cls, optimizer, device,
            scaler_amp, RNN_TYPE
        )
        val_loss, loss_cls, loss_soc = validate_battery_seq(
            val_loader, net, HIDDEN_SIZE, NUM_LAYERS,
            criterion_soc, criterion_cls, device, RNN_TYPE
        )

        train_loss = train_loss.cpu().item()
        val_loss = val_loss.cpu().item()
        loss_cls = loss_cls.cpu().item()
        loss_soc = loss_soc.cpu().item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

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
                val_loss=f"{val_loss:.6f}",
                early_stop=f"best={best_val_loss:.6f}"
            )
            print(f"\n早停于 epoch {epoch}，最佳 val_loss: {best_val_loss:.6f}")
            break

        print(f"epoch 结束: {epoch}")
        tqdm_iter.set_postfix(
            train_loss=f"{train_loss:.6f}",
            val_loss=f"{val_loss:.6f}",
            loss_soc=f"{loss_soc:.6f}",
            loss_cls=f"{loss_cls:.6f}",
            lr=f"{current_lr:.6f}"
        )

    # ---- 保存最终模型 ----
    torch.save(net.state_dict(), 'time_seq_final.opt')
    plot_loss_curve(train_losses, val_losses)
