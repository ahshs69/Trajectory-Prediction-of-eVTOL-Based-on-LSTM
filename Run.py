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

torch.set_float32_matmul_precision('high')


with open(r"E:\bishe\work\results\posearray.pkl", "rb") as f:
    circular = pickle.load(f)


if __name__ == '__main__':
    # ===========================================================================
    # [Bug] train_time_seq 调用 net 时 epoch=0 硬编码，teacher forcing 永不衰减
    #   见 mydataloader2.py 第207行: net(x, state, epoch=0, y=y, type="train")
    #   应改为: net(x, state, epoch=epoch, y=y, type="train")
    # ===========================================================================
    # [速度] stride=1 仍是大瓶颈：每 epoch 处理大量重叠窗口
    #   改为 stride=5 可提速 5x，同时减少过拟合（见 mydataloader2.py:77）
    # ===========================================================================
    # [优化] type="lstm" 使用单向 Encoder，改为 "bi-lstm" 可捕获上下文
    #   需同步: init_state 调用改为 type="bi-lstm"，创建 (num_layer*2, B, H) 状态
    # ===========================================================================
    # [优化] lr=0.005 对 2 层 LSTM 仍偏高，建议 0.001，配合 AdamW 更稳定
    # ===========================================================================
    # [优化] scheduler.patience=2 太激进，建议 5~8
    #   当前 val_loss 即 loss_pos，比之前复合损失更可靠，可适当放宽 patience
    # ===========================================================================
    # [速度] 添加 torch.compile 可再提速 30~50%:
    #   if hasattr(torch, 'compile'):
    #       net = torch.compile(net, mode="reduce-overhead")
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
    test_num = 20
    train_ratio = 0.8
    epochs = 100
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
    optimizer = torch.optim.AdamW(net.parameters(), lr)

    scheduler_warm = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_warmup)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    scaler_amp = torch.amp.GradScaler('cuda')

    loss_list = []
    val_loss_list = []
    pos_loss_list = []
    val_loss_best = float('inf')
    patience = 15
    stop_count = 0

    tqdm_iter = tqdm(range(epochs))
    for epoch in tqdm_iter:

        current_lr = optimizer.param_groups[0]['lr']

        train_loss,pos_loss = train_time_seq(data_loder, net, epoch, hidden_size, num_layer,
                                     criterion, optimizer, device, scaler_amp)
        val_loss = val_time_seq(val_loder, net, hidden_size, num_layer,
                                 criterion, device).cpu().item()
        train_loss = train_loss.cpu().item()
        # pos_loss = pos_loss.cpu().item()

        loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        pos_loss_list.append(pos_loss)

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
                val_loss=f"{val_loss:.6f}",
                early_stop=f"best={val_loss_best:.6f}"
            )
            print(f"\nEarly stopping at epoch {epoch}, best val_loss: {val_loss_best:.6f}")
            break
        
        print(f"epoch ended: {epoch}")
        tqdm_iter.set_postfix(
            train_loss=f"{train_loss:.6f}",
            val_loss=f"{val_loss:.6f}",
            train_pos_loss=f"{pos_loss:.6f}",
            lr=f"{current_lr:.6f}"
            
        )

    torch.save(net.state_dict(), 'time_seq_final.opt')
    plot_loss_curve(loss_list, val_loss_list)
