# -*- coding: utf-8 -*-
"""
无人机直线飞行轨迹生成器
==========================
生成两架无人机的直线飞行轨迹，用于冲突检测仿真。
支持随机起终点、边界约束、可视化对比和 CSV 导出。
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 轨迹生成
# =============================================================================

def generate_linear_trajectory(duration=30.0, freq=10.0,
                               start=None, end=None, bounds=None):
    """生成无人机匀速直线飞行轨迹。

    Args:
        duration: 总飞行时长 (秒)
        freq: 采样频率 (Hz)
        start: 起点坐标 [x, y, z]，None 则在边界内随机生成
        end: 终点坐标 [x, y, z]，None 则在边界内随机生成
        bounds: 边界字典 {'x': (min, max), 'y': (min, max), 'z': (min, max)}

    Returns:
        t:          时间数组 (秒)
        trajectory: 轨迹数组 shape (N, 3)
        start:      起点坐标
        end:        终点坐标
    """
    if bounds is None:
        bounds = {
            'x': (-40, 40),
            'y': (-40, 40),
            'z': (-20, 20),
        }

    if start is None:
        start = np.array([
            np.random.uniform(*bounds['x']),
            np.random.uniform(*bounds['y']),
            np.random.uniform(*bounds['z']),
        ])

    if end is None:
        end = np.array([
            np.random.uniform(*bounds['x']),
            np.random.uniform(*bounds['y']),
            np.random.uniform(*bounds['z']),
        ])

    num_samples = int(duration * freq)
    t = np.linspace(0, duration, num_samples)

    trajectory = np.zeros((num_samples, 3))
    for i in range(3):
        trajectory[:, i] = np.linspace(start[i], end[i], num_samples)

    return t, trajectory, start, end


# =============================================================================
# 可视化
# =============================================================================

def plot_trajectory(t1, traj1, start1, end1,
                    t2, traj2, start2, end2, bounds):
    """并排绘制两条无人机轨迹的多视角对比图。

    包含四个子图：
      1. 3D 轨迹对比
      2. 位置-时间曲线
      3. XY 平面投影
      4. XZ 平面投影
    """
    fig = plt.figure(figsize=(14, 12))

    # ---- 子图 1: 3D 轨迹 ----
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot(traj1[:, 0], traj1[:, 1], traj1[:, 2],
             'b-', linewidth=2, label='UAV1 轨迹')
    ax1.scatter(start1[0], start1[1], start1[2],
                c='blue', s=100, marker='o', label='UAV1 起点')
    ax1.scatter(end1[0], end1[1], end1[2],
                c='cyan', s=100, marker='*', label='UAV1 终点')
    ax1.plot(traj2[:, 0], traj2[:, 1], traj2[:, 2],
             'r-', linewidth=2, label='UAV2 轨迹')
    ax1.scatter(start2[0], start2[1], start2[2],
                c='red', s=100, marker='o', label='UAV2 起点')
    ax1.scatter(end2[0], end2[1], end2[2],
                c='magenta', s=100, marker='*', label='UAV2 终点')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('两架无人机轨迹对比 (3D)')
    ax1.legend(fontsize=7)
    ax1.grid(True)

    # ---- 子图 2: 位置-时间 ----
    ax2 = fig.add_subplot(222)
    ax2.plot(t1, traj1[:, 0], 'r-', linewidth=1.5, label='UAV1 X')
    ax2.plot(t1, traj1[:, 1], 'g-', linewidth=1.5, label='UAV1 Y')
    ax2.plot(t1, traj1[:, 2], 'b-', linewidth=1.5, label='UAV1 Z')
    ax2.plot(t2, traj2[:, 0], 'r--', linewidth=1.5, label='UAV2 X')
    ax2.plot(t2, traj2[:, 1], 'g--', linewidth=1.5, label='UAV2 Y')
    ax2.plot(t2, traj2[:, 2], 'b--', linewidth=1.5, label='UAV2 Z')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('位置 (m)')
    ax2.set_title('位置随时间变化 (实线UAV1, 虚线UAV2)')
    ax2.legend(fontsize=7)
    ax2.grid(True)

    # ---- 子图 3: XY 平面投影 ----
    ax3 = fig.add_subplot(223)
    ax3.plot(traj1[:, 0], traj1[:, 1], 'b-', linewidth=2, label='UAV1')
    ax3.scatter(start1[0], start1[1],
                c='blue', s=80, marker='o', label='UAV1起点')
    ax3.scatter(end1[0], end1[1],
                c='cyan', s=80, marker='*', label='UAV1终点')
    ax3.plot(traj2[:, 0], traj2[:, 1], 'r-', linewidth=2, label='UAV2')
    ax3.scatter(start2[0], start2[1],
                c='red', s=80, marker='o', label='UAV2起点')
    ax3.scatter(end2[0], end2[1],
                c='magenta', s=80, marker='*', label='UAV2终点')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('XY平面投影')
    ax3.set_xlim(bounds['x'])
    ax3.set_ylim(bounds['y'])
    ax3.grid(True)
    ax3.legend(fontsize=7)

    # ---- 子图 4: XZ 平面投影 ----
    ax4 = fig.add_subplot(224)
    ax4.plot(traj1[:, 0], traj1[:, 2], 'b-', linewidth=2, label='UAV1')
    ax4.scatter(start1[0], start1[2],
                c='blue', s=80, marker='o', label='UAV1起点')
    ax4.scatter(end1[0], end1[2],
                c='cyan', s=80, marker='*', label='UAV1终点')
    ax4.plot(traj2[:, 0], traj2[:, 2], 'r-', linewidth=2, label='UAV2')
    ax4.scatter(start2[0], start2[2],
                c='red', s=80, marker='o', label='UAV2起点')
    ax4.scatter(end2[0], end2[2],
                c='magenta', s=80, marker='*', label='UAV2终点')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_title('XZ平面投影')
    ax4.set_xlim(bounds['x'])
    ax4.set_ylim(bounds['z'])
    ax4.grid(True)
    ax4.legend(fontsize=7)

    plt.tight_layout()
    plt.show()


# =============================================================================
# 信息输出
# =============================================================================

def print_trajectory_info(t, trajectory, start, end, bounds):
    """打印单条轨迹的详细信息。

    Args:
        t:          时间数组
        trajectory: 轨迹数组 (N, 3)
        start:      起点坐标
        end:        终点坐标
        bounds:     边界字典
    """
    print("=" * 60)
    print("无人机直线飞行轨迹信息")
    print("=" * 60)
    print(f"总时长: {t[-1]:.1f} 秒")
    print(f"采样频率: {1 / (t[1] - t[0]):.1f} Hz")
    print(f"轨迹点数: {len(t)}")
    print(f"\n边界约束:")
    print(f"  X范围: [{bounds['x'][0]}, {bounds['x'][1]}] m")
    print(f"  Y范围: [{bounds['y'][0]}, {bounds['y'][1]}] m")
    print(f"  Z范围: [{bounds['z'][0]}, {bounds['z'][1]}] m")
    print(f"\n起点坐标: ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f}) m")
    print(f"终点坐标: ({end[0]:.2f}, {end[1]:.2f}, {end[2]:.2f}) m")

    distance = np.linalg.norm(end - start)
    speed = distance / t[-1]
    print(f"\n飞行距离: {distance:.2f} m")
    print(f"平均速度: {speed:.2f} m/s")

    if speed > 20:
        print(f"警告: 平均速度 {speed:.1f} m/s 较高，请确认无人机性能")

    print("\n轨迹前5个点 (时间, x, y, z):")
    for i in range(min(5, len(t))):
        print(f"  t={t[i]:.2f}s: "
              f"({trajectory[i, 0]:.2f}, {trajectory[i, 1]:.2f}, {trajectory[i, 2]:.2f})")

    print("...")
    print("\n轨迹后5个点:")
    for i in range(max(0, len(t) - 5), len(t)):
        print(f"  t={t[i]:.2f}s: "
              f"({trajectory[i, 0]:.2f}, {trajectory[i, 1]:.2f}, {trajectory[i, 2]:.2f})")
    print("=" * 60)


# =============================================================================
# CSV 导出
# =============================================================================

def export_trajectory_to_csv(t, trajectory, filename="drone_trajectory.csv"):
    """将轨迹导出为 CSV 文件（仅含 X, Y, Z 坐标）。

    Args:
        t:          时间数组 (保留用于确定行数)
        trajectory: 轨迹数组 (N, 3)
        filename:   输出文件名
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['X坐标(m)', 'Y坐标(m)', 'Z坐标(m)'])
        for i in range(len(t)):
            writer.writerow([trajectory[i, 0], trajectory[i, 1], trajectory[i, 2]])

    print(f"\n轨迹数据已导出到文件: {filename}")


# =============================================================================
# 工具函数
# =============================================================================

def check_bounds(point, bounds):
    """检查点是否在边界范围内。"""
    return (bounds['x'][0] <= point[0] <= bounds['x'][1] and
            bounds['y'][0] <= point[1] <= bounds['y'][1] and
            bounds['z'][0] <= point[2] <= bounds['z'][1])


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    # 飞行参数
    DURATION_1 = 230        # 无人机 1 飞行时长 (秒)
    DURATION_2 = 230        # 无人机 2 飞行时长 (秒)
    FREQUENCY = 10.0        # 采样频率 (Hz)

    BOUNDS = {
        'x': (-40, 40),
        'y': (-40, 40),
        'z': (-20, 20),
    }

    # 起终点（两机相向飞行，路径交叉）
    start_point1 = np.array([20.0, 20.0, 15.0])
    end_point1 = np.array([-20.0, -20.0, 15.0])
    start_point2 = np.array([-20.0, -20.0, 15.0])
    end_point2 = np.array([20.0, 20.0, 15.0])

    # 生成轨迹
    t1, trajectory1, start1, end1 = generate_linear_trajectory(
        duration=DURATION_1, freq=FREQUENCY,
        start=start_point1, end=end_point1, bounds=BOUNDS,
    )
    t2, trajectory2, start2, end2 = generate_linear_trajectory(
        duration=DURATION_2, freq=FREQUENCY,
        start=start_point2, end=end_point2, bounds=BOUNDS,
    )

    # 打印轨迹信息
    print_trajectory_info(t1, trajectory1, start1, end1, BOUNDS)
    print_trajectory_info(t2, trajectory2, start2, end2, BOUNDS)

    # 可视化对比
    plot_trajectory(t1, trajectory1, start1, end1,
                    t2, trajectory2, start2, end2, BOUNDS)

    # 导出 CSV（仅 X, Y, Z 三列，供 run.py 使用）
    export_trajectory_to_csv(t1, trajectory1, "linear_trajectory1.csv")
    export_trajectory_to_csv(t2, trajectory2, "linear_trajectory2.csv")
