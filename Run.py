# -*- coding: utf-8 -*-
import os
import torch
import pickle
import numpy as np
from torch import nn
from tqdm import tqdm
from mydataloader2 import *
from SOC_lstm import *
from animation_marking import plot_loss_curve


torch.set_float32_matmul_precision('medium')


with open(r"E:\bishe\work\BMS\dataset1.pkl", "rb") as f:
    circular = pickle.load(f)


if __name__ == '__main__':

    batch_size = 64
    train_steps = 20
    input_size = 3
    hidden_size = 256
    lr = 0.005
    num_layers = 2
    num_classes = 4
    pred_steps = 1
    test_num = 60
    train_ratio = 0.8
    epochs = 500

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.version.cuda) 
    print(f"CUDA: {torch.cuda.is_available()}, Device: {device}")

    scaler = StandardScaler()
    scaler.battery_fit(circular)
    circular_trans = scaler.battery_transform(circular)

    data_loder = MyDataLoader(circular_trans, train_ratio, test_num, batch_size, train_steps, pred_steps)
    val_loder = MyDataLoader(circular_trans, train_ratio, test_num, batch_size, train_steps, pred_steps, type="val")

    net = SOC_LSTM(input_size, hidden_size, num_layers,num_classes).to(device)

    criterion_soc = nn.MSELoss(reduction="none")
    criterion_cls = nn.CrossEntropyLoss()  
    optimizer = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_down)


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

        train_loss = train_battery_seq(data_loder, net, epoch, hidden_size, num_layers,
                                     criterion_soc, criterion_cls,optimizer, device, scaler_amp,type)
        val_loss = val_battery_seq(val_loder, net, hidden_size, num_layers,
                                 criterion_soc,criterion_cls, device,type).cpu().item()
        train_loss = train_loss.cpu().item()


        loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        scheduler.step()
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
