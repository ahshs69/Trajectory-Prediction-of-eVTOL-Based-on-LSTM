# -*- coding: utf-8 -*-
"""
数据加载与训练工具模块（冲突检测版本）
======================================
包含数据标准化器、滑动窗口数据加载器和训练/验证/预测函数。
专用于轨迹预测任务（位置+速度输入，未来位置输出）。
"""

import torch
import numpy as np
from torch.utils.data import DataLoader


# =============================================================================
# 数据标准化器
# =============================================================================

class StandardScaler:
    """轨迹数据标准化器。

    位置和速度分开标准化以处理量纲差异。
    拟合统计量保存在 scaler.npz 中以便后续逆变换。
    """

    def __init__(self):
        self.pos_mean = 0.0
        self.vel_mean = 0.0
        self.pos_std = 1.0
        self.vel_std = 1.0

    def fit(self, dataset, train_ratio=0.8):
        """在训练集上拟合标准化参数。

        Args:
            dataset: 轨迹片段列表，每条 shape (T, D)，列 0 为时间戳，列 1:4 为位置，列 4:7 为速度
            train_ratio: 训练集比例
        """
        train_data = dataset[:int(len(dataset) * train_ratio)]
        if len(train_data) == 0:
            raise ValueError("训练集为空，请检查 train_ratio 或数据集大小")

        all_data = np.concatenate(train_data, axis=0)
        self.pos_mean = np.mean(all_data[:, 1:4], axis=0)
        self.vel_mean = np.mean(all_data[:, 4:7], axis=0)
        self.pos_std = np.std(all_data[:, 1:4], axis=0)
        self.vel_std = np.std(all_data[:, 4:7], axis=0)

        # 防止除零
        self.pos_std[self.pos_std == 0] = 1e-8
        self.vel_std[self.vel_std == 0] = 1e-8

        np.savez("scaler.npz",
                 pos_mean=self.pos_mean, vel_mean=self.vel_mean,
                 pos_std=self.pos_std, vel_std=self.vel_std)

    def transform(self, dataset):
        """对轨迹数据集进行标准化。

        Args:
            dataset: 轨迹片段列表

        Returns:
            标准化后的 tensor 列表，去除时间戳列
        """
        lengths = [len(seg) for seg in dataset]
        all_data = np.concatenate([np.array(seg)[:, 1:] for seg in dataset], axis=0)

        all_data[:, :3] = (all_data[:, :3] - self.pos_mean) / self.pos_std
        all_data[:, 3:6] = (all_data[:, 3:6] - self.vel_mean) / self.vel_std

        trans_data = []
        start = 0
        for length in lengths:
            trans_data.append(torch.tensor(all_data[start:start + length]))
            start += length
        return trans_data

    def inverse_transform(self, multistep_preds, train_steps, pred_steps,
                          output_size, test_num=10):
        """对多步预测结果进行逆标准化。

        Args:
            multistep_preds: 模型预测输出列表
            train_steps: 输入序列长度
            pred_steps: 预测步数
            output_size: 输出维度
            test_num: 测试样本数

        Returns:
            逆标准化后的数据列表（numpy 数组）
        """
        datas = []
        pad_first = np.zeros((train_steps, pred_steps, 3))

        for i in range(test_num):
            data = multistep_preds[i].cpu().numpy() * self.pos_std
            data = data + self.pos_mean
            data = np.concatenate([pad_first, data], axis=0)
            datas.append(data)
        return datas


# =============================================================================
# 滑动窗口数据准备
# =============================================================================

def prepare_data(dataset, test_num=10, train_ratio=0.8, batch_size=32,
                 train_steps=20, pred_steps=12, mode="train"):
    """将轨迹片段数据集切分为滑动窗口并创建 DataLoader。

    Args:
        dataset: 标准化后的 tensor 列表
        test_num: 测试集末尾样本数
        train_ratio: 训练集比例
        batch_size: 批次大小
        train_steps: 输入（历史）时间步数
        pred_steps: 预测（未来）时间步数
        mode: "train" / "val"

    Returns:
        torch DataLoader 对象
    """
    stride = 1

    if mode == "train":
        min_idx = 0
        max_idx = int(len(dataset) * train_ratio)
    elif mode == "val":
        min_idx = int(len(dataset) * train_ratio)
        max_idx = len(dataset) - test_num
    else:
        raise ValueError(f"未知的数据划分模式: {mode}，可选 'train'/'val'")

    data_list_x = []
    data_list_y = []

    for i in range(min_idx, max_idx):
        data_tensor = dataset[i]
        if len(data_tensor) <= train_steps + pred_steps:
            continue
        windows = data_tensor.unfold(0, train_steps + pred_steps, stride)
        windows = windows.permute(0, 2, 1)
        x_windows = windows[:, :, :train_steps]
        y_windows = windows[:, :, train_steps:]
        x_windows = x_windows.permute(0, 2, 1)
        y_windows = y_windows.permute(0, 2, 1)
        data_list_x.append(x_windows)
        data_list_y.append(y_windows)

    if not data_list_x:
        raise ValueError(
            f"{mode} 划分中无有效窗口，请检查轨迹长度是否 > train_steps+pred_steps"
        )

    train_data_x = torch.cat(data_list_x, dim=0)
    train_data_y = torch.cat(data_list_y, dim=0)

    dataloader = DataLoader(
        list(zip(train_data_x, train_data_y)),
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    return dataloader


def make_multistep_ground_truth(dataset, test_num=10, train_steps=20, input_size=3):
    """为测试集生成多步预测的真值序列。"""
    data_true_list = []

    for i in range(int(len(dataset) - test_num), len(dataset)):
        data_tensor = dataset[i]
        end_idx = len(data_tensor) - train_steps
        data_true = torch.zeros((end_idx, train_steps, input_size))

        for j in range(end_idx):
            data_true[j] = data_tensor[j:j + train_steps]

        data_true_list.append(data_true)
    return data_true_list


class MyDataLoader:
    """统一的数据加载器封装。"""

    def __init__(self, dataset, train_ratio=0.8, test_num=10, batch_size=32,
                 train_steps=20, pred_steps=12, mode="train"):
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.test_num = test_num
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.pred_steps = pred_steps
        self.mode = mode
        self.dataloader = prepare_data(
            self.dataset, self.test_num, self.train_ratio, self.batch_size,
            self.train_steps, self.pred_steps, self.mode
        )

    def __iter__(self):
        return iter(self.dataloader)

    def test_iter(self, input_size):
        """返回测试集真值序列。"""
        return make_multistep_ground_truth(
            self.dataset, self.test_num, self.train_steps, input_size
        )


# =============================================================================
# RNN 隐状态初始化
# =============================================================================

def init_state(batch_size, hidden_dim, device, num_layers, rnn_type="lstm"):
    """根据 RNN 类型创建初始隐状态。"""
    if rnn_type == "rnn":
        return torch.zeros((num_layers, batch_size, hidden_dim)).to(device)
    elif rnn_type == "lstm":
        return (torch.zeros((num_layers, batch_size, hidden_dim)).to(device),
                torch.zeros((num_layers, batch_size, hidden_dim)).to(device))
    elif rnn_type == "bi-lstm":
        return (torch.zeros((num_layers * 2, batch_size, hidden_dim)).to(device),
                torch.zeros((num_layers * 2, batch_size, hidden_dim)).to(device))
    else:
        raise ValueError(f"不支持的 RNN 类型: {rnn_type}")


# =============================================================================
# 训练 / 验证 / 预测
# =============================================================================

def train_time_seq(time_seq_loader, net, epoch, hidden_size, num_layers,
                   criterion, optimizer, device, scaler_amp, rnn_type="lstm"):
    """轨迹预测模型单轮训练。"""
    net.train()
    total_loss = []

    for x, y in time_seq_loader:
        batch_size = x.shape[0]
        state = init_state(batch_size, hidden_size, device, num_layers, rnn_type)
        x = x.float().to(device)
        y = y.float().to(device)

        with torch.amp.autocast('cuda'):
            y_hat, _ = net(x, state, epoch=epoch, y=y, mode="train")
            y_pos = y[:, :, :3]
            loss = criterion(y_hat, y_pos).mean()

        optimizer.zero_grad()
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss.append(loss.detach())

    return torch.stack(total_loss).mean()


def val_time_seq(time_seq_loader, net, hidden_size, num_layers,
                 criterion, device, rnn_type="lstm"):
    """轨迹预测模型验证。"""
    net.eval()
    total_loss = []

    with torch.no_grad():
        for x, y in time_seq_loader:
            batch_size = x.shape[0]
            state = init_state(batch_size, hidden_size, device, num_layers, rnn_type)
            x = x.float().to(device)
            y = y.float().to(device)

            y_hat, _ = net(x, state, epoch=0, y=y, mode="val")
            y_pos = y[:, :, :3]
            loss = criterion(y_hat, y_pos).mean()
            total_loss.append(loss.detach())

    return torch.stack(total_loss).mean()


def pred_time_seq(time_seq_loader, net, device, test_num, num_layers,
                  hidden_size=128, rnn_type="lstm"):
    """轨迹预测模型推理。"""
    net.eval()
    pred_list = []

    with torch.no_grad():
        for i in range(test_num):
            x = time_seq_loader[i].to(device)
            state = init_state(len(x), hidden_size, device, num_layers, rnn_type)
            y_hat, _ = net(x, state, epoch=0, y=None, mode="val")
            pred_list.append(y_hat)

    return pred_list


def lr_warmup(epoch):
    """前 5 轮线性 warmup。"""
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0


__all__ = [
    "StandardScaler",
    "MyDataLoader",
    "prepare_data",
    "make_multistep_ground_truth",
    "init_state",
    "train_time_seq",
    "val_time_seq",
    "pred_time_seq",
    "lr_warmup",
]
