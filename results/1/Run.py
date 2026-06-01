# -*- coding: utf-8 -*-
import os
import torch
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
from mydataloader2 import *
from animation_marking import plot_loss_curve
from seq2seq import Seq2Seq

torch.set_float32_matmul_precision('medium')


with open(r"results\posearray.pkl", "rb") as f:
    circular = pickle.load(f)


if __name__ == '__main__':
    # ===========================================================================
    # [准确度 P0] val_loss 在 0.000255 处停滞，核心原因是学习率衰减过快
    #   ReduceLROnPlateau(patience=2, factor=0.5) 导致 lr 每 2~4 epoch 折半
    #   epoch 30 时 lr=0.000625，epoch 40 时 lr=0.000156，epoch 50 时 lr=2e-5
    #   这么小的 lr 几乎无法更新参数，模型陷入局部最优出不来
    #   方案A: patience 改为 8~10，factor 改为 0.7~0.8(温和衰减)
    #   方案B: 用 CosineAnnealingWarmRestarts(T_0=20, T_mult=2) 周期性重启
    #   方案C: 前 5 epoch warmup，之后 CosineAnnealingLR(T_max=95)
    #   Cosine 天然让 lr 在高低间循环，给模型多次跳出局部最优的机会
    # ===========================================================================
    # [准确度] type="lstm"(单向) → "bi-lstm"(双向) 是最直接的精度提升
    #   双向 Encoder 能同时看前后文，对拐点/趋势变化预测明显更好
    #   需同步: 1) init_state 调用改为 type="bi-lstm"
    #           2) 创建 (num_layer*2, B, H) 的初始状态
    # ===========================================================================
    # [准确度] AdamW 缺少 weight_decay 参数
    #   optimizer = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-5)
    #   weight_decay 提供 L2 正则化，缩小 train/val loss 差距(当前 ~3x)
    # ===========================================================================
    # [准确度] 可启用 SWA(Stochastic Weight Averaging) 进一步提升泛化:
    #   from torch.optim.swa_utils import AveragedModel, SWALR
    #   swa_model = AveragedModel(net)
    #   swa_scheduler = SWALR(optimizer, swa_lr=0.001)
    #   每轮或每几轮更新 swa_model，最终用 swa_model 推理
    # ===========================================================================
    # [速度] stride=1 每 epoch ~7.5 分钟，改为 5 可降至 ~1.5 分钟
    #   同时减少相邻窗口过拟合，让模型学真正的轨迹动态
    # ===========================================================================
    # [速度] torch.compile 可提速 30~50%:
    #   if hasattr(torch, 'compile'):
    #       net = torch.compile(net, mode="reduce-overhead")
    # ===========================================================================
    # [准确度] epochs=100 看起来多，但当前 lr 衰减过快导致 30+ epoch 白费
    #   修 lr 调度后建议 epochs 设为 150~200，配合 Cosine 周期让模型充分训练
    # ===========================================================================
    batch_size = 64
    train_steps = 20
    input_size = 6
    de_input_size = 3
    hidden_size = 256
    linear_size = 128
    output_size = 3
    lr = 0.005
    num_layer = 2
    pred_steps = 10
    test_num = 0
    train_ratio = 0.8
    epochs = 500
    type = "lstm"

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.version.cuda) 
    print(f"CUDA: {torch.cuda.is_available()}, Device: {device}")

    scaler = StandardScaler()
    scaler.fit(circular)
    circular_trans = scaler.transform(circular)

    data_loder = MyDataLoader(circular_trans, train_ratio, test_num, batch_size, train_steps, pred_steps)
    val_loder = MyDataLoader(circular_trans, train_ratio, test_num, batch_size, train_steps, pred_steps, type="val")

    net = Seq2Seq(input_size, de_input_size, hidden_size, output_size, num_layer, linear_size, pred_steps, train_steps,type).to(device)

    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)

    scheduler_warm = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_warmup)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=2
    )

    scaler_amp = torch.amp.GradScaler('cuda')

    loss_list = []
    val_loss_list = []
    # pos_loss_list = []
    val_loss_best = float('inf')
    patience = 30
    stop_count = 0
    # net = torch.load("model.pth", map_location=device, weights_only=False

    tqdm_iter = tqdm(range(epochs))
    for epoch in tqdm_iter:
        current_lr = optimizer.param_groups[0]['lr']

        train_loss = train_time_seq(data_loder, net, epoch, hidden_size, num_layer,
                                     criterion, optimizer, device, scaler_amp)
        val_loss = val_time_seq(val_loder, net, hidden_size, num_layer,
                                 criterion, device).cpu().item()
        train_loss = train_loss.cpu().item()
        # pos_loss = pos_loss.cpu().item()

        loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        # pos_loss_list.append(pos_loss)

        if epoch < 5:
            scheduler_warm.step()
        else:
            scheduler.step(val_loss)

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            stop_count = 0
            torch.save(net.state_dict(), 'time_seq_best.opt')
        else:
            stop_count += 1

        if stop_count >= patience:
            tqdm_iter.set_postfix(
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.9f}",
                early_stop=f"best={val_loss_best:.9f}"
            )
            print(f"\nEarly stopping at epoch {epoch}, best val_loss: {val_loss_best:.9f}")
            break
        
        print(f"epoch ended: {epoch}")
        tqdm_iter.set_postfix(
            train_loss=f"{train_loss:.8f}",
            val_loss=f"{val_loss:.9f}",
            lr=f"{current_lr:.8f}"
            
        )

    torch.save(net.state_dict(), 'time_seq_final.opt')
    plot_loss_curve(loss_list, val_loss_list)
