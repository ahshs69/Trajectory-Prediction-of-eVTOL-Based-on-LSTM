# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def animation(dataset, num_poses=252, save_dir="results"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], markersize=4, color='blue', label='Trajectory')
    ax.set_title("3D_Animation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    x_min, x_max = min(dataset["tx"]) - 1, max(dataset["tx"]) + 1
    y_min, y_max = min(dataset["ty"]) - 1, max(dataset["ty"]) + 1
    z_min, z_max = min(dataset["tz"]) - 1, max(dataset["tz"]) + 1

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        return line

    def update(frame):
        line.set_data(dataset["tx"][:frame], dataset["ty"][:frame])
        line.set_3d_properties(dataset["tz"][:frame])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        return line

    ani = FuncAnimation(fig, update, frames=num_poses, init_func=init, blit=False)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "animation_gt.gif")
    ani.save(save_path, writer=PillowWriter(fps=30))
    plt.show()


# def animation_double(idx, dataset1, dataset2, train_steps=20, pred_steps=20,
#                      save_dir="results"):
    # ===========================================================================
    # [Bug] 预测线与真实轨迹"贴在一起"的原因:
    #   pred_data[frame] 每帧取不同预测窗口 → 预测线每帧变化，始终从当前位置出发
    #   导致预测线永远"追赶"真实轨迹的当前位置，而非作为独立延续显示
    #   前 train_steps 帧是 inverse_transform 的 pad_first 零点 → 显示在原点
    # ===========================================================================
    # [修改] 取固定窗口(如第 0 个窗口)的预测作为静态延续:
    #   pred = pred_data[train_steps]  # 跳过零点填充，取第一个有效预测
    #   预测在动画中作为固定红色线条，从输入序列结束位置开始延伸
    # ===========================================================================
def animation_double(idx, dataset1, dataset2, save_dir="results"):
    real_data = np.array(dataset1)
    pred_data = dataset2

    total_frames = len(real_data)
    print(f"Total frames: {total_frames}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    line_real, = ax.plot([], [], [], markersize=4, color='blue', label='Real')
    line_pred, = ax.plot([], [], [], 'r-', linewidth=2, label='Pred')

    ax.set_title("3D_Animation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()

    x_min, x_max = real_data[:, 1].min() - 5, real_data[:, 1].max() + 5
    y_min, y_max = real_data[:, 2].min() - 5, real_data[:, 2].max() + 5
    z_min, z_max = real_data[:, 3].min() - 5, real_data[:, 3].max() + 5
    
    # x_min, x_max = -40, 40
    # y_min, y_max = -40, 40
    # z_min, z_max =  -10, 20

    def init():
        line_real.set_data([], [])
        line_real.set_3d_properties([])
        line_pred.set_data([], [])
        line_pred.set_3d_properties([])
        return line_real, line_pred

    def update(frame):
        x_r = real_data[:frame, 1]
        y_r = real_data[:frame, 2]
        z_r = real_data[:frame, 3]
        line_real.set_data(x_r, y_r)
        line_real.set_3d_properties(z_r)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

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


def plot_loss_curve(train_losses, val_losses, save_path="loss_curve.png"):
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
    print(f"Loss curve saved: {save_path}")
    
    import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def traj_draw(data):
# 你的三维坐标数据
    data

    # 分离x, y, z坐标
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制轨迹线
    ax.plot(x, y, z, 'b-', linewidth=2, label='3D Trajectory')

    # 绘制数据点
    ax.scatter(x, y, z, c='red', s=50, marker='o', label='Data Points')

    # 标注每个点的序号
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        ax.text(xi, yi, zi, f'{i+1}', fontsize=10, ha='right', va='bottom')

    # 设置坐标轴标签
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_zlabel('Z Coordinate', fontsize=12)

    # 设置标题
    ax.set_title('3D Trajectory Plot', fontsize=14, fontweight='bold')

    # 添加图例
    ax.legend()

    # 设置视角（可选，调整为更好的观察角度）
    ax.view_init(elev=20, azim=45)

    # 显示网格
    ax.grid(True)

    # 显示图形
    plt.tight_layout()
    plt.show()
