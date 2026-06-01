# -*- coding: utf-8 -*-
"""
数据加载与训练工具模块
======================
包含数据标准化器、滑动窗口数据加载器、训练/验证/测试函数和评估指标。
支持两类任务：
  - 轨迹预测（位置+速度输入，未来位置输出）
  - 电池 SOC 估计与工况分类（电压/电流/温度输入，SOC 回归 + 机动段分类输出）
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# =============================================================================
# 数据标准化器
# =============================================================================

class StandardScaler:
    """支持轨迹数据和电池数据的标准化器。

    轨迹数据使用位置和速度分别标准化（量纲不同），
    电池数据使用电压、电流、温度分别标准化（量纲不同）。
    拟合统计量保存在 .npz 文件中以便后续逆变换。
    """

    def __init__(self):
        # 轨迹数据统计量
        self.pos_mean = 0.0
        self.vel_mean = 0.0
        self.pos_std = 1.0
        self.vel_std = 1.0
        # 电池数据统计量
        self.v_mean = 0.0
        self.i_mean = 0.0
        self.t_mean = 0.0
        self.v_std = 1.0
        self.i_std = 1.0
        self.t_std = 1.0

    # -------------------------------------------------------------------------
    # 轨迹数据拟合与变换
    # -------------------------------------------------------------------------

    def fit(self, dataset, train_ratio=0.8):
        """在训练集上拟合轨迹数据标准化参数。

        Args:
            dataset: 轨迹片段列表，每条 shape (T, D)，列 0 为时间戳，列 1:4 为位置，列 4:7 为速度
            train_ratio: 训练集比例
        """
        train_data = dataset[:int(len(dataset) * train_ratio)]
        if len(train_data) == 0:
            raise ValueError("训练集为空，请检查 train_ratio 或数据集大小")

        all_data = np.concatenate(train_data, axis=0)
        # 位置和速度分开标准化，量纲不同
        self.pos_mean = np.mean(all_data[:, 1:4], axis=0)
        self.pos_std = np.std(all_data[:, 1:4], axis=0)
        self.vel_mean = np.mean(all_data[:, 4:7], axis=0)
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

        在预测结果前填充 train_steps 行零值，使动画中预测线与真实轨迹对齐呈现。

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

    # -------------------------------------------------------------------------
    # 电池数据拟合与变换
    # -------------------------------------------------------------------------

    def battery_fit(self, dataset, train_ratio=0.8):
        """在训练集上拟合电池数据标准化参数。

        Args:
            dataset: 电池数据片段列表，列 0 为时间戳，列 1 为电压，列 2 为电流，列 3 为温度
            train_ratio: 训练集比例
        """
        train_data = dataset[:int(len(dataset) * train_ratio)]
        if len(train_data) == 0:
            raise ValueError("训练集为空，请检查 train_ratio 或数据集大小")

        all_data = np.concatenate(train_data, axis=0)
        self.v_mean = np.mean(all_data[:, 1], axis=0)
        self.i_mean = np.mean(all_data[:, 2], axis=0)
        self.t_mean = np.mean(all_data[:, 3], axis=0)
        self.v_std = np.std(all_data[:, 1], axis=0)
        self.i_std = np.std(all_data[:, 2], axis=0)
        self.t_std = np.std(all_data[:, 3], axis=0)

        # 防止除零
        for attr in ('v_std', 'i_std', 't_std'):
            if getattr(self, attr) == 0:
                setattr(self, attr, 1e-8)

        np.savez("scaler_battery.npz",
                 v_mean=self.v_mean, v_std=self.v_std,
                 i_mean=self.i_mean, i_std=self.i_std,
                 t_mean=self.t_mean, t_std=self.t_std)

    def battery_transform(self, dataset):
        """对电池数据集进行标准化。

        Args:
            dataset: 电池数据片段列表

        Returns:
            标准化后的 tensor 列表，去除时间戳列
        """
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


# =============================================================================
# 滑动窗口数据准备
# =============================================================================

def prepare_data(dataset, test_num=10, train_ratio=0.8, batch_size=32,
                 train_steps=20, pred_steps=12, mode="train"):
    """将轨迹片段数据集切分为滑动窗口并创建 DataLoader。

    使用 unfold 操作高效生成 (x, y) 窗口对，x 为历史步，y 为未来步。

    Args:
        dataset: 标准化后的 tensor 列表
        test_num: 测试集末尾样本数
        train_ratio: 训练集比例
        batch_size: 批次大小
        train_steps: 输入（历史）时间步数
        pred_steps: 预测（未来）时间步数
        mode: "train" / "val" / "test"

    Returns:
        torch DataLoader 对象

    Note:
        stride=1 时相邻窗口高度重叠，适合小数据集；
        若数据充足，可增大 stride（如 train_steps//2）减少冗余。
    """
    stride = 1

    if mode == "train":
        min_idx = 0
        max_idx = int(len(dataset) * train_ratio)
    elif mode == "val":
        min_idx = int(len(dataset) * train_ratio)
        max_idx = len(dataset) - test_num
    elif mode == "test":
        min_idx = int(len(dataset) * train_ratio) - test_num
        max_idx = len(dataset)
    else:
        raise ValueError(f"未知的数据划分模式: {mode}，可选 'train'/'val'/'test'")

    data_list_x = []
    data_list_y = []

    for i in range(min_idx, max_idx):
        data_tensor = dataset[i]
        if len(data_tensor) <= train_steps + pred_steps:
            continue
        windows = data_tensor.unfold(0, train_steps + pred_steps, stride)
        windows = windows.permute(0, 2, 1)                  # (N, features, steps)
        x_windows = windows[:, :, :train_steps]             # 历史窗口
        y_windows = windows[:, :, train_steps:]             # 未来窗口
        # 恢复为 (N, steps, features) 形状
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
    """为测试集生成多步预测的真值序列。

    对每条测试轨迹，生成所有可能的 (train_steps, input_size) 真值窗口。

    Args:
        dataset: 标准化后的 tensor 列表
        test_num: 测试集样本数
        train_steps: 输入时间步数
        input_size: 特征维度

    Returns:
        真值窗口列表，每个元素 shape (num_windows, train_steps, input_size)
    """
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
    """统一的数据加载器封装。

    根据 mode 参数自动划分训练/验证/测试集并创建对应的 DataLoader。
    """

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
        """返回测试集真值序列（用于与预测对比）。"""
        return make_multistep_ground_truth(
            self.dataset, self.test_num, self.train_steps, input_size
        )


# =============================================================================
# RNN 隐状态初始化
# =============================================================================

def init_state(batch_size, hidden_dim, device, num_layers, rnn_type="lstm"):
    """根据 RNN 类型创建初始隐状态。

    Args:
        batch_size: 批次大小
        hidden_dim: 隐层维度
        device: 计算设备
        num_layers: RNN 层数
        rnn_type: "rnn" / "lstm" / "bi-lstm"

    Returns:
        RNN: (num_layers, B, H) 张量
        LSTM: ((num_layers, B, H), (num_layers, B, H)) 元组
        Bi-LSTM: ((num_layers*2, B, H), (num_layers*2, B, H)) 元组
    """
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
# 轨迹预测 —— 训练 / 验证 / 测试 / 预测
# =============================================================================

def train_trajectory(time_seq_loader, net, epoch, hidden_size, num_layers,
                     criterion, optimizer, device, scaler_amp, rnn_type="lstm"):
    """轨迹预测模型单轮训练。

    使用混合精度训练 + 梯度裁剪。损失仅为位置 MSE。
    """
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


def validate_trajectory(time_seq_loader, net, hidden_size, num_layers,
                        criterion, device, rnn_type="lstm"):
    """轨迹预测模型验证。

    使用纯自回归模式（teacher forcing ratio=0）评估多步预测能力。
    """
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


def predict_trajectory(time_seq_loader, net, device, test_num, num_layers,
                       hidden_size, rnn_type="lstm"):
    """轨迹预测模型推理。

    对 test_num 条测试轨迹逐一生成多步预测。

    Returns:
        预测结果列表，每个元素 shape (1, pred_steps, 3)
    """
    net.eval()
    pred_list = []

    with torch.no_grad():
        for i in range(test_num):
            x = time_seq_loader[i].to(device)
            state = init_state(len(x), hidden_size, device, num_layers, rnn_type)
            y_hat, _ = net(x, state, epoch=0, y=None, mode="val")
            pred_list.append(y_hat)

    return pred_list


# =============================================================================
# 电池 SOC —— 训练 / 验证 / 测试
# =============================================================================

def train_battery(time_seq_loader, net, epoch, hidden_size, num_layers,
                  criterion_soc, criterion_cls, optimizer, device,
                  scaler_amp, rnn_type="lstm"):
    """电池 SOC 多任务模型单轮训练。

    同时优化 SOC 回归（MSE）和机动段分类（交叉熵），
    使用同方差不确定性加权自动平衡两个任务的损失。
    """
    net.train()
    total_loss = []

    for x, y in time_seq_loader:
        batch_size = x.shape[0]
        state = init_state(batch_size, hidden_size, device, num_layers, rnn_type)
        x = x.float().to(device)
        y = y.float().to(device)

        with torch.amp.autocast('cuda'):
            soc_hat, cls_hat, log_sigma_soc, log_sigma_cls = net(x, state)
            # SOC 回归损失
            y_soc_true = y[:, :, 5]
            loss_soc = criterion_soc(soc_hat, y_soc_true).mean()
            # 机动段分类损失
            y_cls_true = y[:, :, 4].squeeze().long()
            loss_cls = criterion_cls(cls_hat, y_cls_true).mean()
            # 同方差不确定性加权
            precision_soc = torch.exp(-log_sigma_soc)
            precision_cls = torch.exp(-log_sigma_cls)
            loss = (precision_soc * loss_soc +
                    precision_cls * loss_cls +
                    log_sigma_soc + log_sigma_cls)

        optimizer.zero_grad()
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()

        total_loss.append(loss.detach())

    return torch.stack(total_loss).mean()


def validate_battery(time_seq_loader, net, hidden_size, num_layers,
                     criterion_soc, criterion_cls, device, rnn_type="lstm"):
    """电池 SOC 多任务模型验证。

    Returns:
        (总损失, 分类损失, SOC 回归损失)
    """
    net.eval()
    total_loss = []

    with torch.no_grad():
        for x, y in time_seq_loader:
            batch_size = x.shape[0]
            state = init_state(batch_size, hidden_size, device, num_layers, rnn_type)
            x = x.float().to(device)
            y = y.float().to(device)

            soc_hat, cls_hat, _, _ = net(x, state)
            # SOC 回归损失
            y_soc_true = y[:, :, 5]
            loss_soc = criterion_soc(soc_hat, y_soc_true).mean()
            # 机动段分类损失
            y_cls_true = y[:, :, 4].squeeze().long()
            loss_cls = criterion_cls(cls_hat, y_cls_true).mean()
            # 验证时直接求和（两个损失量纲不同，建议分任务监控）
            loss = loss_soc + loss_cls
            total_loss.append(loss.detach())

    return (torch.stack(total_loss).mean(),
            loss_cls.detach(),
            loss_soc.detach())


# =============================================================================
# 评估指标
# =============================================================================

def mae(y_pred, y_true):
    """平均绝对误差 (Mean Absolute Error)"""
    return torch.mean(torch.abs(y_pred - y_true))


def rmse(y_pred, y_true):
    """均方根误差 (Root Mean Squared Error)"""
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def r2_score_torch(y_pred, y_true):
    """决定系数 (R² Score)"""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def mse(y_pred, y_true):
    """均方误差 (Mean Squared Error)"""
    return torch.mean((y_pred - y_true) ** 2)


def endpoint_error(y_pred, y_true):
    """计算平均端点误差和最后一步端点误差。

    Returns:
        (平均各步欧氏距离, 最后一步欧氏距离)
    """
    avg_dist = torch.sqrt(torch.sum((y_pred - y_true) ** 2, dim=-1)).mean()
    last_dist = torch.sqrt(
        torch.sum((y_pred[:, -1, :3] - y_true[:, -1, :3]) ** 2, dim=-1)
    ).mean()
    return avg_dist, last_dist


def compute_classification_metrics(y_true, y_pred_probs):
    """计算分类指标。

    Args:
        y_true: 真实标签 (numpy array)
        y_pred_probs: 预测概率或 logits (numpy array)，shape (N, num_classes)

    Returns:
        (accuracy, f1_score, precision_score)
    """
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    return accuracy, f1, precision


# =============================================================================
# 轨迹预测测试
# =============================================================================

def test_trajectory(time_seq_loader, net, hidden_size, num_layers, device,
                    rnn_type="lstm"):
    """轨迹预测模型测试 —— 计算全部评估指标。

    Returns:
        (MSE, MAE, RMSE, R², 平均端点误差, 最后一步端点误差)
    """
    net.eval()
    total_mse = []
    total_mae = []
    total_rmse = []
    total_r2 = []
    total_avg_ee = []
    total_last_ee = []

    with torch.no_grad():
        for x, y in time_seq_loader:
            batch_size = x.shape[0]
            state = init_state(batch_size, hidden_size, device, num_layers, rnn_type)
            x = x.float().to(device)
            y = y.float().to(device)

            y_hat, _ = net(x, state, epoch=0, y=y, mode="val")
            y_pos = y[:, :, :3]

            total_mse.append(mse(y_hat, y_pos).detach())
            total_mae.append(mae(y_hat, y_pos).detach())
            total_rmse.append(rmse(y_hat, y_pos).detach())
            total_r2.append(r2_score_torch(y_hat, y_pos).detach())
            avg_ee, last_ee = endpoint_error(y_hat, y_pos)
            total_avg_ee.append(avg_ee.detach())
            total_last_ee.append(last_ee.detach())

    return (torch.stack(total_mse).mean(),
            torch.stack(total_mae).mean(),
            torch.stack(total_rmse).mean(),
            torch.stack(total_r2).mean(),
            torch.stack(total_avg_ee).mean(),
            torch.stack(total_last_ee).mean())


# =============================================================================
# 电池 SOC 测试
# =============================================================================

def test_battery(time_seq_loader, net, hidden_size, num_layers, device,
                 rnn_type="lstm"):
    """电池 SOC 多任务模型测试 —— 回归 + 分类评估。

    Returns:
        (MSE, MAE, RMSE, R², Accuracy, F1, Precision)
    """
    net.eval()
    total_mse = []
    total_mae = []
    total_rmse = []
    total_r2 = []
    total_acc = []
    total_f1 = []
    total_prec = []

    with torch.no_grad():
        for x, y in time_seq_loader:
            batch_size = x.shape[0]
            state = init_state(batch_size, hidden_size, device, num_layers, rnn_type)
            x = x.float().to(device)
            y = y.float().to(device)

            soc_hat, cls_hat, _, _ = net(x, state)
            y_soc_true = y[:, :, 5]
            y_cls_true = x[:, -1, 4].squeeze()

            # 分类指标
            probs = torch.softmax(cls_hat, dim=1)
            pred_indices = torch.argmax(probs, dim=1).detach().cpu().numpy()
            y_cls_np = y_cls_true.detach().cpu().numpy()
            accuracy = accuracy_score(y_cls_np, pred_indices)
            f1 = f1_score(y_cls_np, pred_indices, average="weighted", zero_division=0)
            prec = precision_score(y_cls_np, pred_indices, average="weighted", zero_division=0)

            # 回归指标
            total_mse.append(mse(soc_hat, y_soc_true).detach())
            total_mae.append(mae(soc_hat, y_soc_true).detach())
            total_rmse.append(rmse(soc_hat, y_soc_true).detach())
            total_r2.append(r2_score_torch(soc_hat, y_soc_true).detach())
            total_acc.append(accuracy)
            total_f1.append(f1)
            total_prec.append(prec)

    return (torch.stack(total_mse).mean(),
            torch.stack(total_mae).mean(),
            torch.stack(total_rmse).mean(),
            torch.stack(total_r2).mean(),
            sum(total_acc) / len(total_acc),
            sum(total_f1) / len(total_f1),
            sum(total_prec) / len(total_prec))


# =============================================================================
# 学习率调度辅助函数
# =============================================================================

def lr_warmup(epoch):
    """前 warmup_epochs 轮线性 warmup，之后返回 1.0。"""
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0


def lr_decay(epoch):
    """指数衰减: 0.99^epoch。"""
    return 0.99 ** epoch


# =============================================================================
# 兼容旧接口的别名
# =============================================================================

# 保持向后兼容的旧函数名
multi_steps_data = make_multistep_ground_truth
train_time_seq = train_trajectory
val_time_seq = validate_trajectory
pred_time_seq = predict_trajectory
train_battery_seq = train_battery
val_battery_seq = validate_battery
test = test_trajectory

__all__ = [
    "StandardScaler",
    "MyDataLoader",
    "prepare_data",
    "make_multistep_ground_truth",
    "multi_steps_data",
    "init_state",
    # 轨迹预测
    "train_trajectory", "train_time_seq",
    "validate_trajectory", "val_time_seq",
    "predict_trajectory", "pred_time_seq",
    "test_trajectory", "test",
    # 电池 SOC
    "train_battery", "train_battery_seq",
    "validate_battery", "val_battery_seq",
    "test_battery",
    # 评估指标
    "mse", "mae", "rmse", "r2_score_torch",
    "endpoint_error", "compute_classification_metrics",
    # 学习率
    "lr_warmup", "lr_decay", "lr_down",
]
