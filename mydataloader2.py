# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils.data import DataLoader


class StandardScaler():

    def __init__(self):
        self.pos_mean = 0.
        self.vel_mean = 0.
        self.pos_std = 1.
        self.vel_std = 1.

    def fit(self, dataset, train_ratio=0.8):
        train_data = dataset[:int(len(dataset) * train_ratio)]
        if len(train_data) == 0:
            raise ValueError("Train dataset is empty, check train_ratio or dataset size")

        all_data = np.concatenate(train_data, axis=0)
        # 位置和速度分开标准化，量纲不同
        self.pos_mean = np.mean(all_data[:, 1:4], axis=0)
        self.vel_mean = np.mean(all_data[:, 4:7], axis=0)
        self.pos_std = np.std(all_data[:, 1:4], axis=0)
        self.vel_std = np.std(all_data[:, 4:7], axis=0)
        # self.bear_mean = np.mean(all_data[:, 7:], axis=0)
        # self.bear_std = np.std(all_data[:, 7:], axis=0)

        self.pos_std[self.pos_std == 0] = 1e-8
        self.vel_std[self.vel_std == 0] = 1e-8
        # self.bear_std[self.bear_std == 0] = 1e-8

        np.savez("scaler.npz", pos_mean=self.pos_mean, vel_mean=self.vel_mean,
                 pos_std=self.pos_std, vel_std=self.vel_std)
        
    def battery_fit(self, dataset, train_ratio=0.8):
        train_data = dataset[:int(len(dataset) * train_ratio)]
        if len(train_data) == 0:
            raise ValueError("Train dataset is empty, check train_ratio or dataset size")

        all_data = np.concatenate(train_data, axis=0)
        # 位置和速度分开标准化，量纲不同
        self.v_mean = np.mean(all_data[:, 1], axis=0)
        self.i_mean = np.mean(all_data[:, 2], axis=0)
        self.t_mean = np.mean(all_data[:, 3], axis=0)
        self.v_std = np.std(all_data[:, 1], axis=0)
        self.i_std = np.std(all_data[:, 2], axis=0)
        self.t_std = np.std(all_data[:, 3], axis=0)

        if self.v_std == 0:
            self.v_std = 1e-8
        if self.i_std == 0:
            self.i_std = 1e-8
        if self.t_std == 0:
            self.t_std = 1e-8
        np.savez("scaler_battery.npz", v_mean=self.v_mean, v_std=self.v_std, 
                 i_mean=self.i_mean ,i_std=self.i_std,
                 t_mean=self.t_mean, t_std=self.t_std)

    def transform(self, dataset):
        # 向量化：一次性 concat + 标准化，再按原始长度 split
        lengths = [len(seg) for seg in dataset]
        all_data = np.concatenate([np.array(seg)[:, 1:] for seg in dataset], axis=0)

        all_data[:, :3] = (all_data[:, :3] - self.pos_mean) / self.pos_std
        all_data[:, 3:6] = (all_data[:, 3:6] - self.vel_mean) / self.vel_std
        # all_data[:, 6:] = (all_data[:, 6:] - self.bear_mean) / self.bear_std

        trans_data = []
        start = 0
        for length in lengths:
            trans_data.append(torch.tensor(all_data[start:start + length]))
            start += length

        return trans_data
    
    def battery_transform(self, dataset):
        # 向量化：一次性 concat + 标准化，再按原始长度 split
        lengths = [len(seg) for seg in dataset]
        all_data = np.concatenate([np.array(seg)[:, 1:] for seg in dataset], axis=0)

        all_data[:, 0] = (all_data[:, 0] - self.v_mean) / self.v_std
        all_data[:, 1] = (all_data[:, 1] - self.i_mean) / self.i_std
        all_data[:, 2] = (all_data[:, 2] - self.t_mean) / self.t_std

        trans_data = []
        start = 0
        for length in lengths:
            trans_data.append(torch.tensor(all_data[start:start + length]))
            start += length

        return trans_data

    def inverse_transform(self, mutil_steps_pred, train_steps, pred_steps,
                          output_size, test_num=10):
        # [Bug] pad_first 前置 train_steps 行零点 → 动画前 20 帧预测线显示在原点
        #   且 animation_double 中 pred_data[frame] 每帧取不同窗口的预测
        #   导致预测线不断变化、始终从当前位置出发，与真实轨迹"贴在一起"
        # [修改] 去掉 pad_first，让 pred_data 与预测窗口一一对应(无填充)
        #   在 animation_double 中用单一固定窗口(如 pred_data[0])作为静态延续
        datas = []
        pad_first = np.zeros((train_steps, pred_steps, 3))

        for i in range(test_num):
            data = mutil_steps_pred[i].cpu().numpy() * self.pos_std
            data = data + self.pos_mean
            data = np.concatenate([pad_first, data], axis=0)
            datas.append(data)

        return datas


def prepare_data(dataset, test_num=10, train_ratio=0.8, batch_size=32,
                 train_steps=20, pred_steps=12, type="train"):

    stride = 1

    data_list_x = []
    data_list_y = []

    if type == "train":
        min_idx = 0
        max_idx = int(len(dataset) * train_ratio)
    elif type == "val":
        min_idx = int(len(dataset) * train_ratio)
        max_idx = len(dataset) - test_num
    else:
        raise ValueError(f"unknown type: {type}")

    for i in range(min_idx, max_idx):
        data_tensor = dataset[i]
        if len(data_tensor) <= train_steps + pred_steps:
            continue
        # unfold 步长改为 stride，减少相邻窗口重叠
        windows = data_tensor.unfold(0, train_steps + pred_steps, stride)
        windows = windows.permute(0, 2, 1)
        x_windows = windows[:, :train_steps, :]
        y_windows = windows[:, train_steps:, :]
        data_list_x.append(x_windows)
        data_list_y.append(y_windows)

    if not data_list_x:
        raise ValueError(f"No valid windows for {type} split, check trajectory lengths")

    train_data_x = torch.cat(data_list_x, dim=0)
    train_data_y = torch.cat(data_list_y, dim=0)

    dataloader = DataLoader(
        list(zip(train_data_x, train_data_y)),
        batch_size=batch_size,
        shuffle=(type == "train"),
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    return dataloader


def multi_steps_data(dataset, test_num=10, train_step=20, input_size=3):
    data_true_list = []

    for i in range(int(len(dataset) - test_num), len(dataset)):
        data_tensor = dataset[i]
        end_idx = len(data_tensor) - train_step
        data_true = torch.zeros((end_idx, train_step, input_size))

        for j in range(end_idx):
            data_true[j] = data_tensor[j:j + train_step]

        data_true_list.append(data_true)

    return data_true_list


class MyDataLoader(object):

    def __init__(self, dataset, train_ratio=0.8, test_num=10, batch_size=32,
                 train_steps=20, pred_steps=12, type="train"):
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.test_num = test_num
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.pred_steps = pred_steps
        self.type = type
        self.dataloder = prepare_data(self.dataset, self.test_num,
                                       self.train_ratio, self.batch_size,
                                       self.train_steps, self.pred_steps,
                                       self.type)

    def __iter__(self):
        return iter(self.dataloder)

    def test_iter(self, input_size):
        data_true_list = multi_steps_data(self.dataset, self.test_num,
                                           self.train_steps, input_size)
        return data_true_list
    


def init_state(batch_size, hidden_dim, device, num_layer, type="lstm"):
    # [注意] 当前 train_time_seq/val_time_seq 调用 init_state 时未传 type 参数
    #   默认 type="lstm" 创建 (num_layer, B, H) 状态
    #   若 Encoder 改为 bi-lstm，此处需改为 type="bi-lstm" 创建 (num_layer*2, B, H)
    #   建议: 将 type 参数传入 train/val 函数，透传至 init_state
    if type == "rnn":
        return torch.zeros((num_layer, batch_size, hidden_dim)).to(device)
    elif type == "lstm":
        return (torch.zeros((num_layer , batch_size, hidden_dim)).to(device),
                torch.zeros((num_layer , batch_size, hidden_dim)).to(device))
    elif type == "bi-lstm":
        return (torch.zeros((num_layer * 2, batch_size, hidden_dim)).to(device),
                torch.zeros((num_layer * 2, batch_size, hidden_dim)).to(device))



def train_time_seq(time_seq_loader, net, epoch, hidden_size, num_layer,
                   criterion, optimizer, device, scaler_amp,type):

    net.train()
    total_loss = []
    # total_loss_pos = []
    for x, y in time_seq_loader:
        batch_size = x.shape[0]
        state = init_state(batch_size, hidden_size, device, num_layer,type)
        x = x.float().to(device)
        y = y.float().to(device)

        with torch.amp.autocast('cuda'):
            
            y_hat,_ = net(x, state, epoch=epoch, y=y, type="train")
            y_pos = y[:, :, :3]
            loss_pos = criterion(y_hat, y_pos).mean()

            
            loss = loss_pos 

        optimizer.zero_grad()
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss.append(loss.detach())
        # total_loss_pos.append(loss_pos.detach())

    return torch.stack(total_loss).mean()


def val_time_seq(time_seq_loader, net, hidden_size, num_layer, criterion, device,type):
    # [准确度] 当前 val loss 只是平均位置 MSE，不反映多步预测的误差累积
    #   预测第 1 步 vs 第 10 步的误差可能差一个数量级
    #   建议分开记录: loss_step_0_3, loss_step_4_6, loss_step_7_9
    #   观察模型预测能力随步长衰减的规律，针对性优化远步预测
    # ===========================================================================
    net.eval()
    total_loss = []
    
    with torch.no_grad():
        for x, y in time_seq_loader:
            batch_size = x.shape[0]
            state = init_state(batch_size, hidden_size, device, num_layer,type)
            x = x.float().to(device)
            y = y.float().to(device)

            y_hat,_ = net(x, state, epoch=0, y=y, type="val")
            y_pos = y[:, :, :3]
            loss_pos = criterion(y_hat, y_pos).mean()
            
            total_loss.append(loss_pos.detach())

    return torch.stack(total_loss).mean()

def pred_time_seq(time_seq_loader, net, device, test_num, num_layer,
                  hidden_size=128):
    net.eval()
    pred_list = []

    with torch.no_grad():
        for i in range(test_num):
            x = time_seq_loader[i].to(device)
            state = init_state(len(x), hidden_size, device, num_layer)
            y_hat, _ = net(x, state, epoch=0, type="val")
            # y_hat = y_hat[:,:, :3]
            pred_list.append(y_hat)

    return pred_list


def train_battery_seq(time_seq_loader, net, epoch, hidden_size, num_layer,
                   criterion_soc,criterion_cls, optimizer, device, scaler_amp,type):
    net.train()
    total_loss = []
    for x, y in time_seq_loader:
        batch_size = x.shape[0]
        state = init_state(batch_size, hidden_size, device, num_layer,type)
        x = x.float().to(device)
        y = y.float().to(device)

        with torch.amp.autocast('cuda'):
            
            soc_hat,cls_hat,a,b = net(x, state)
            y_true = y[:, :, 5]
            loss_soc = criterion_soc(soc_hat, y_true).mean()
            y_cls = y[:,:,4].squeeze().long()
            loss_cls = criterion_cls(cls_hat, y_cls).mean()
            precision_soc = torch.exp(-a)
            precision_cls = torch.exp(-b)
            loss = precision_soc*loss_soc+precision_cls*loss_cls


        optimizer.zero_grad()
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss.append(loss.detach())
    return torch.stack(total_loss).mean()


def val_battery_seq(time_seq_loader, net, hidden_size, num_layer, criterion_soc,criterion_cls, device,type):

    net.eval()
    total_loss = []
    
    with torch.no_grad():
        for x, y in time_seq_loader:
            batch_size = x.shape[0]
            state = init_state(batch_size, hidden_size, device, num_layer,type)
            x = x.float().to(device)
            y = y.float().to(device)
            
            soc_hat,cls_hat,a,b = net(x, state)
            y_true = y[:, :, 5]
            loss_soc = criterion_soc(soc_hat, y_true).mean()
            y_cls = y[:,:,4].squeeze().long()
            loss_cls = criterion_cls(cls_hat, y_cls).mean()
            precision_soc = torch.exp(-a)
            precision_cls = torch.exp(-b)
            loss = precision_soc*loss_soc+precision_cls*loss_cls
            
            total_loss.append(loss.detach())

    return torch.stack(total_loss).mean()


def lr_warmup(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    else:
        return 1.0
    
def lr_down(epoch):

    return 0.99**epoch



__all__ = ["prepare_data", "multi_steps_data", "MyDataLoader", "init_state",
           "train_time_seq", "val_time_seq", "pred_time_seq", "StandardScaler",
           "lr_warmup","train_battery_seq","val_battery_seq","lr_down"]
