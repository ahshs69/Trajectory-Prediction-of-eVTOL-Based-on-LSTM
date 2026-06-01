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
    type = "gru"

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
    optimizer = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-5)

    # scheduler_warm = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_warmup)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.75, patience=2
    )

    scaler_amp = torch.amp.GradScaler('cuda')

    loss_list = []
    val_loss_list = []
    pos_loss_list = []
    val_loss_best = float('inf')
    patience = 30
    stop_count = 0


    tqdm_iter = tqdm(range(epochs))
    for epoch in tqdm_iter:


        current_lr = optimizer.param_groups[0]['lr']

        train_loss = train_time_seq(data_loder, net, epoch, hidden_size, num_layer,
                                     criterion, optimizer, device, scaler_amp,type)
        val_loss = val_time_seq(val_loder, net, hidden_size, num_layer,
                                 criterion, device,type).cpu().item()
        train_loss = train_loss.cpu().item()
        # pos_loss = pos_loss.cpu().item()

        loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        # pos_loss_list.append(pos_loss)

        # if epoch < 5:
        #     scheduler_warm.step()
        # else:
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
            train_loss=f"{train_loss:.8f}",
            val_loss=f"{val_loss:.8f}",
            lr=f"{current_lr:.8f}"
            
        )

    torch.save(net.state_dict(), 'time_seq_final.opt')
    plot_loss_curve(loss_list, val_loss_list)
