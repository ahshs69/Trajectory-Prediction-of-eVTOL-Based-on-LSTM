# -*- coding: utf-8 -*-
"""
可视化工具模块
==============
提供 3D 轨迹动画、损失曲线绘制、轨迹对比图等功能。
依赖 matplotlib 和 numpy。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — 注册 3D 投影

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


# =============================================================================
# 3D 轨迹动画
# =============================================================================

def animation(dataset, num_poses=252, save_dir="results"):
    """绘制单条轨迹的 3D 动画并保存为 GIF。

    Args:
        dataset: 包含 "tx", "ty", "tz" 列的类字典对象
        num_poses: 总帧数
        save_dir: 保存目录
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], markersize=4, color='blue', label='Trajectory')
    ax.set_title("3D Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    x_min, x_max = min(dataset["tx"]) - 1, max(dataset["tx"]) + 1
    y_min, y_max = min(dataset["ty"]) - 1, max(dataset["ty"]) + 1
    z_min, z_max = min(dataset["tz"]) - 1, max(dataset["tz"]) + 1

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line,

    def update(frame):
        line.set_data(dataset["tx"][:frame], dataset["ty"][:frame])
        line.set_3d_properties(dataset["tz"][:frame])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        return line,

    ani = FuncAnimation(fig, update, frames=num_poses, init_func=init, blit=False)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "animation_gt.gif")
    ani.save(save_path, writer=PillowWriter(fps=30))
    plt.show()


def animation_double(idx, real_dataset, pred_dataset, save_dir="results"):
    """绘制真实轨迹与预测轨迹对比的 3D 动画。

    真实轨迹（蓝色）逐帧累积显示，预测轨迹（红色）为固定窗口的静态延伸。

    Args:
        idx: 轨迹编号，用于命名保存文件
        real_dataset: 真实轨迹数据 (numpy array)，列 1:4 为位置
        pred_dataset: 预测轨迹列表，每个元素为 (train_steps+pred_steps, 3) 的 numpy 数组
        save_dir: 保存目录
    """
    real_data = np.array(real_dataset)
    pred_data = pred_dataset
    total_frames = len(real_data)
    print(f"总帧数: {total_frames}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line_real, = ax.plot([], [], [], markersize=4, color='blue', label='Real')
    line_pred, = ax.plot([], [], [], 'r-', linewidth=2, label='Pred')

    ax.set_title("3D Animation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # 基于真实数据范围设定坐标轴
    x_min, x_max = real_data[:, 1].min() - 5, real_data[:, 1].max() + 5
    y_min, y_max = real_data[:, 2].min() - 5, real_data[:, 2].max() + 5
    z_min, z_max = real_data[:, 3].min() - 5, real_data[:, 3].max() + 5

    def init():
        line_real.set_data([], [])
        line_real.set_3d_properties([])
        line_pred.set_data([], [])
        line_pred.set_3d_properties([])
        return line_real, line_pred

    def update(frame):
        # 真实轨迹逐帧累积
        x_r = real_data[:frame, 1]
        y_r = real_data[:frame, 2]
        z_r = real_data[:frame, 3]
        line_real.set_data(x_r, y_r)
        line_real.set_3d_properties(z_r)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # 预测轨迹显示对应帧的预测窗口
        if frame < len(pred_data):
            pred = pred_data[frame]
            line_pred.set_data(pred[:, 0], pred[:, 1])
            line_pred.set_3d_properties(pred[:, 2])

    ani = FuncAnimation(fig, update, frames=total_frames,
                        init_func=init, blit=False)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"animation_gt_{idx}.gif")
    ani.save(save_path, writer=PillowWriter(fps=15))
    plt.show()


# =============================================================================
# 损失曲线
# =============================================================================

def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png"):
    """绘制训练和验证损失曲线。

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 图片保存路径
    """
    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label="Train Loss", color="#3498db", linewidth=2)
    plt.plot(val_losses, label="Val Loss", color="#e74c3c", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Train & Validation Loss Curve", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"损失曲线已保存: {save_path}")


# =============================================================================
# 静态 3D 轨迹绘制
# =============================================================================

def plot_3d_trajectory(data):
    """绘制单条 3D 轨迹（含点标记和序号）。

    Args:
        data: numpy 数组，shape (N, 3)，列依次为 x, y, z
    """
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x, y, z, 'b-', linewidth=2, label='3D Trajectory')
    ax.scatter(x, y, z, c='red', s=50, marker='o', label='Data Points')

    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ax.text(xi, yi, zi, f'{i + 1}', fontsize=10, ha='right', va='bottom')

    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Z Coordinate', fontsize=12)
    ax.set_title('3D Trajectory Plot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.view_init(elev=20, azim=45)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# 兼容旧接口的别名
traj_draw = plot_3d_trajectory


# =============================================================================
# 预测 vs 真实轨迹对比
# =============================================================================

def plot_pred_vs_true(pred_data_list, true_data_list,
                      pred_idx=0, true_idx=0, time_step=0):
    """绘制单个预测窗口与对应真实轨迹的 3D 对比图。

    Args:
        pred_data_list: 预测数据列表（逆标准化后），每个元素 shape (total_steps, 3)
        true_data_list: 真实数据列表（逆标准化后），每个元素 shape (T, D>3)
        pred_idx: 预测数据集中要绘制的轨迹索引
        true_idx: 真实数据集中要绘制的轨迹索引
        time_step: 时间步偏移
    """
    data_pred = pred_data_list[pred_idx][time_step]
    data_true = true_data_list[true_idx][time_step:time_step + 10, 1:4]

    x_pred, y_pred, z_pred = data_pred[:, 0], data_pred[:, 1], data_pred[:, 2]
    x_true, y_true, z_true = data_true[:, 0], data_true[:, 1], data_true[:, 2]

    fig = plt.figure(figsize=(8, 5), dpi=600)
    ax = fig.add_subplot(111, projection='3d')

    # 预测线（蓝色实线）
    ax.plot(x_pred, y_pred, z_pred, 'b-', linewidth=2, alpha=0.7,
            label='Predicted Trajectory')
    ax.scatter(x_pred, y_pred, z_pred, c='blue', s=30, marker='o', alpha=0.6)

    # 真实线（红色虚线）
    ax.plot(x_true, y_true, z_true, 'r--', linewidth=2, alpha=0.7,
            label='True Trajectory')
    ax.scatter(x_true, y_true, z_true, c='red', s=30, marker='^', alpha=0.6)

    # 标注起点和终点
    ax.text(x_pred[0], y_pred[0], z_pred[0], 'Pred Start',
            fontsize=10, color='blue', fontweight='bold')
    ax.text(x_pred[-1], y_pred[-1], z_pred[-1], 'Pred End',
            fontsize=10, color='blue', fontweight='bold')
    ax.text(x_true[0], y_true[0], z_true[0], 'True Start',
            fontsize=10, color='red', fontweight='bold')
    ax.text(x_true[-1], y_true[-1], z_true[-1], 'True End',
            fontsize=10, color='red', fontweight='bold')

    ax.xaxis.set_major_locator(ticker.MaxNLocator(7))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(6))

    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Z Coordinate', fontsize=12)
    ax.legend(loc='upper right')
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# 兼容旧接口的别名
line_plot = plot_pred_vs_true


def plot_trajectory_comparison(data1, data2):
    """并排绘制两条 3D 轨迹的对比图。

    Args:
        data1: 第一条轨迹 (numpy array)，列 1:4 为位置
        data2: 第二条轨迹 (numpy array)，列 1:4 为位置
    """
    x1, y1, z1 = data1[:, 1], data1[:, 2], data1[:, 3]
    x2, y2, z2 = data2[:, 1], data2[:, 2], data2[:, 3]

    fig = plt.figure(figsize=(18, 8), dpi=200)

    # ---- 左图：轨迹 1 ----
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x1, y1, z1, 'b-', linewidth=2.5, alpha=0.8, label='Trajectory')
    ax1.scatter(x1, y1, z1, c='blue', s=40, marker='o', alpha=0.8,
                edgecolors='darkblue', linewidth=0.5)
    ax1.scatter(x1[0], y1[0], z1[0], c='green', s=100, marker='o',
                edgecolors='darkgreen', linewidth=1.5, label='Start')
    ax1.scatter(x1[-1], y1[-1], z1[-1], c='red', s=100, marker='s',
                edgecolors='darkred', linewidth=1.5, label='End')
    ax1.set_xlabel('X', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax1.set_title('Trajectory 1', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.view_init(elev=45, azim=45)

    # ---- 右图：轨迹 2 ----
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(x2, y2, z2, 'r-', linewidth=2.5, alpha=0.8, label='Trajectory')
    ax2.scatter(x2, y2, z2, c='red', s=40, marker='o', alpha=0.8,
                edgecolors='darkred', linewidth=0.5)
    ax2.scatter(x2[0], y2[0], z2[0], c='green', s=100, marker='o',
                edgecolors='darkgreen', linewidth=1.5, label='Start')
    ax2.scatter(x2[-1], y2[-1], z2[-1], c='orange', s=100, marker='s',
                edgecolors='darkorange', linewidth=1.5, label='End')
    ax2.set_xlabel('X', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax2.set_zlabel('Z', fontsize=12, fontweight='bold')
    ax2.set_title('Trajectory 2', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.view_init(elev=45, azim=45)

    plt.tight_layout()
    plt.show()


# 兼容旧接口的别名
traj_plot = plot_trajectory_comparison
