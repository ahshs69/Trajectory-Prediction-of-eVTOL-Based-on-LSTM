# -*- coding: utf-8 -*-
"""
Panda3D 无人机冲突检测仿真
===========================
加载城市与无人机 3D 模型，驱动两架无人机按预设轨迹飞行，
实时进行轨迹预测和冲突检测。

功能：
  - 两架无人机沿 CSV 轨迹独立飞行
  - 使用 Seq2Seq 模型实时预测未来轨迹（10 步）
  - 基于预测轨迹进行时空冲突检测（距离阈值判断）
  - 绘制历史轨迹（绿色/蓝色）和预测轨迹（红色/黄色）
  - 支持多线程并行推理
"""

import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from direct.showbase.ShowBase import ShowBase
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    LineSegs,
    Point3,
    Vec2,
    Vec4,
    VBase3,
    VBase4,
    loadPrcFileData,
)

from mydataloader2 import init_state
from seq2seq import Seq2Seq


# =============================================================================
# Panda3D 窗口配置
# =============================================================================

WINDOW_CONFIG = """
win-size 1920 1080
window-title Panda3D Window
show-frame-rate-meter 1
"""
loadPrcFileData("", WINDOW_CONFIG)

# =============================================================================
# 轨迹数据加载
# =============================================================================

# CSV 列名: X坐标(m), Y坐标(m), Z坐标(m)
TRAJECTORY_CSV_1 = "linear_trajectory1.csv"
TRAJECTORY_CSV_2 = "linear_trajectory2.csv"

datas = pd.read_csv(TRAJECTORY_CSV_1)[
    ['X坐标(m)', 'Y坐标(m)', 'Z坐标(m)']
].to_numpy()
datas2 = pd.read_csv(TRAJECTORY_CSV_2)[
    ['X坐标(m)', 'Y坐标(m)', 'Z坐标(m)']
].to_numpy()

# =============================================================================
# 标准化参数加载
# =============================================================================

inverse_data = np.load('scaler.npz')
pos_mean = inverse_data['pos_mean']
pos_std = inverse_data['pos_std']
vel_mean = inverse_data['vel_mean']
vel_std = inverse_data['vel_std']
inverse_data.close()

# =============================================================================
# 模型参数与加载
# =============================================================================

TRAIN_STEPS = 20
INPUT_SIZE = 6          # 位置(3) + 速度(3)
DE_INPUT_SIZE = 3       # 解码器输入: 位置(3)
HIDDEN_SIZE = 256
LINEAR_SIZE = 128
OUTPUT_SIZE = 3
NUM_LAYERS = 2
PRED_STEPS = 10
RNN_TYPE = "lstm"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Seq2Seq(INPUT_SIZE, DE_INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
              NUM_LAYERS, LINEAR_SIZE, PRED_STEPS, TRAIN_STEPS,
              RNN_TYPE).to(device)
net.load_state_dict(
    torch.load("time_seq_best.opt", map_location=device, weights_only=False)
)
net.eval()


# =============================================================================
# 主应用类
# =============================================================================

class ConflictDetectionApp(ShowBase):
    """Panda3D 冲突检测仿真应用。

    加载城市和无人机模型，驱动两架无人机按轨迹飞行，
    实时进行轨迹预测和冲突检测。
    """

    # 冲突检测参数
    CONFLICT_THRESHOLD = 2.0    # 预测冲突距离阈值 (m)
    COLLISION_THRESHOLD = 0.1   # 实时碰撞距离阈值 (m)
    POS_BUFFER_SIZE = 20        # 位置缓冲区大小
    TIME_STEP = 0.1             # 仿真时间步 (s)

    def __init__(self):
        super().__init__()

        # ---- 场景加载 ----
        self._load_scene()
        # ---- 光照设置 ----
        self._setup_lights()
        # ---- 无人机模型 ----
        self._load_drone_models()
        # ---- 相机设置 ----
        self._setup_camera()
        # ---- 轨迹线可视化 ----
        self._setup_trail_lines()
        # ---- 任务链与推理 ----
        self._setup_tasks()

    # -------------------------------------------------------------------------
    # 场景
    # -------------------------------------------------------------------------

    def _load_scene(self):
        """加载城市模型和天空盒。"""
        self.scene = self.loader.loadModel("city")
        self.scene.reparentTo(self.render)
        self.scene.setScale(3, 3, 3)
        self.scene.setPos(datas[0, 0] + 20, datas[0, 1] + 50, datas[0, 2] - 50)

        self.skybox = self.loader.loadModel("skybox")
        self.skybox.setScale(100, 100, 100)
        self.skybox.setPos(0, 0, -500)
        self.skybox.reparentTo(self.render)

        skycol = VBase3(135 / 255.0, 206 / 255.0, 235 / 255.0)
        self.set_background_color(skycol)

    # -------------------------------------------------------------------------
    # 光照
    # -------------------------------------------------------------------------

    def _setup_lights(self):
        """配置环境光和方向光。"""
        skycol = VBase3(135 / 255.0, 206 / 255.0, 235 / 255.0)

        # 环境光
        alight = AmbientLight("sky")
        alight.set_color(VBase4(skycol * 0.04, 1))
        alight_path = self.render.attachNewNode(alight)
        self.render.set_light(alight_path)

        # 四盏垂直方向补光
        for angle in (0, -90, -180, -270):
            dlight = DirectionalLight('directionalLight')
            dlight.setColor(Vec4(0.3, 0.3, 0.3, 0.3))
            dlight_np = self.render.attachNewNode(dlight)
            dlight_np.setHpr(angle, 0, 0)
            self.render.set_light(dlight_np)

        # 主方向光（模拟太阳，投射阴影）
        dlight = DirectionalLight('directionalLight')
        dlight.setColor(Vec4(1, 1, 1, 1))
        dlight.getLens().setFilmSize(Vec2(50, 50))
        dlight.getLens().setNearFar(-100, 100)
        dlight.setShadowCaster(True, 4096 * 2, 4096 * 2)
        dlight_np = self.render.attachNewNode(dlight)
        dlight_np.setHpr(0, -65, 0)
        self.render.setShaderAuto()
        self.render.setLight(dlight_np)

    # -------------------------------------------------------------------------
    # 无人机模型
    # -------------------------------------------------------------------------

    def _load_drone_models(self):
        """加载两架无人机的 3D 模型及螺旋桨。"""
        # ---- 无人机 1 (quad_model) ----
        self.quad_model = self.loader.loadModel("drone_noprop.glb")
        self.quad_model.reparentTo(self.render)

        # 螺旋桨（带偏移位置）
        propeller_positions_1 = [
            (11.736, 9.8547, 5.3903),
            (13.766, -10.881, 7.6289),
            (-11.754, 9.8547, 5.3903),
            (-13.767, -10.886, 7.6289),
        ]
        self.propellers_1 = []
        for pos in propeller_positions_1:
            prop = self.loader.loadModel("prop4.glb")
            prop.reparentTo(self.quad_model)
            prop.setPos(*pos)
            self.propellers_1.append(prop)

        self.quad_model.setScale(0.01)
        self.quad_model.setPos(datas[0, 0], datas[0, 1], datas[0, 2])

        # ---- 无人机 2 (uav_model) ----
        self.uav_model = self.loader.loadModel("quad")
        self.uav_model.reparentTo(self.render)

        propeller_positions_2 = [
            (-0.26, 0, 0),
            (0.26, 0, 0),
            (0, 0.26, 0),
            (0, -0.26, 0),
        ]
        self.propellers_2 = []
        for pos in propeller_positions_2:
            prop = self.loader.loadModel("prop")
            prop.reparentTo(self.uav_model)
            prop.setPos(*pos)
            self.propellers_2.append(prop)

        self.uav_model.setPos(datas2[0, 0], datas2[0, 1], datas2[0, 2])

    # -------------------------------------------------------------------------
    # 相机
    # -------------------------------------------------------------------------

    def _setup_camera(self):
        """设置相机跟随无人机 1。"""
        self.cam_neutral_pos = Point3(1000, 1000, 1500)
        self.cam.setScale(100)
        self.cam.node().getLens().setFilmSize(36, 24)
        self.cam.node().getLens().setFocalLength(45)
        self.cam.setPos(self.cam_neutral_pos)
        self.cam.reparentTo(self.quad_model)
        self.cam.lookAt(self.quad_model)

    # -------------------------------------------------------------------------
    # 轨迹线可视化
    # -------------------------------------------------------------------------

    def _setup_trail_lines(self):
        """初始化四条轨迹线：两架无人机各自的历史（绿/蓝）和预测（红/黄）轨迹。"""
        # 无人机 1 历史轨迹（绿色）
        self.line_segs = LineSegs()
        self.line_segs.setColor(0, 1, 0, 1)
        self.line_segs.setThickness(5)
        self.trail_node = self.line_segs.create()
        self.trail_np = self.render.attachNewNode(self.trail_node)
        self.last_pos = None

        # 无人机 2 历史轨迹（蓝色）
        self.line_segs2 = LineSegs()
        self.line_segs2.setColor(0, 0, 1, 1)
        self.line_segs2.setThickness(5)
        self.trail_node2 = self.line_segs2.create()
        self.trail_np2 = self.render.attachNewNode(self.trail_node2)
        self.last_pos2 = None

        # 无人机 1 预测轨迹（红色）
        self.line_pred = LineSegs()
        self.line_pred.setColor(1, 0, 0, 1)
        self.line_pred.setThickness(30)
        self.pred_node = self.line_pred.create()
        self.pred_np = self.render.attachNewNode(self.pred_node)

        # 无人机 2 预测轨迹（黄色）
        self.line_pred2 = LineSegs()
        self.line_pred2.setColor(1, 1, 0, 1)
        self.line_pred2.setThickness(30)
        self.pred_node2 = self.line_pred2.create()
        self.pred_np2 = self.render.attachNewNode(self.pred_node2)

    # -------------------------------------------------------------------------
    # 任务调度
    # -------------------------------------------------------------------------

    def _setup_tasks(self):
        """注册所有周期性任务。"""
        # 后台任务链（双线程，用于更新缓冲区和推理）
        self.taskMgr.setupTaskChain("bg_chain", numThreads=2, frameBudget=0.01)

        # 位置缓冲区
        self.pos_buffer = []
        self.pos_buffer2 = []

        # 两架无人机各自独立的 LSTM 隐状态
        self.state1 = init_state(1, HIDDEN_SIZE, device, NUM_LAYERS, RNN_TYPE)
        self.state2 = init_state(1, HIDDEN_SIZE, device, NUM_LAYERS, RNN_TYPE)

        # 线程池用于两机并行推理
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 注册任务
        self.taskMgr.add(self._draw_trail, "drawtrail")
        self.taskMgr.add(self._draw_trail2, "drawtrail2")
        self.taskMgr.doMethodLater(0.1, self._get_pos, "getpos")
        self.taskMgr.add(self._rotate_props, "rotateproptask")

        # 启动飞行动画
        self._start_flight_animations()

    # -------------------------------------------------------------------------
    # 飞行动画
    # -------------------------------------------------------------------------

    def _start_flight_animations(self):
        """为两架无人机创建沿轨迹飞行的位移动画序列。"""
        # 无人机 1
        pos_interval_1 = []
        for pos in datas[1:]:
            pos_interval_1.append(
                self.quad_model.posInterval(self.TIME_STEP,
                                            Point3(pos[0], pos[1], pos[2]))
            )
        seq1 = Sequence(*pos_interval_1)
        seq1.start()

        # 无人机 2
        pos_interval_2 = []
        for pos in datas2[1:]:
            pos_interval_2.append(
                self.uav_model.posInterval(self.TIME_STEP,
                                           Point3(pos[0], pos[1], pos[2]))
            )
        seq2 = Sequence(*pos_interval_2)
        seq2.start()

    # -------------------------------------------------------------------------
    # 位置缓冲区更新
    # -------------------------------------------------------------------------

    def _update_buffer(self, now_pos, pos_buffer):
        """更新位置缓冲区并维护速度信息。

        缓冲区满 20 条时返回归一化后的输入 tensor，否则返回 None。

        Args:
            now_pos: (Point3) 当前位置
            pos_buffer: 位置缓冲区列表

        Returns:
            归一化 tensor (1, 20, 6) 或 None
        """
        dx, dy, dz = now_pos

        if len(pos_buffer) < 1:
            pos_buffer.append([dx, dy, dz, 0, 0, 0])
        else:
            p_prev = pos_buffer[-1][:3]
            new_vel = [
                (dx - p_prev[0]) / self.TIME_STEP,
                (dy - p_prev[1]) / self.TIME_STEP,
                (dz - p_prev[2]) / self.TIME_STEP,
            ]
            pos_buffer.append([dx, dy, dz, new_vel[0], new_vel[1], new_vel[2]])

        # 保持缓冲区大小不超过 20
        if len(pos_buffer) > self.POS_BUFFER_SIZE:
            pos_buffer[:] = pos_buffer[-self.POS_BUFFER_SIZE:]

        if len(pos_buffer) == self.POS_BUFFER_SIZE:
            tensor_data = torch.tensor(pos_buffer)
            # 标准化
            tensor_data[:, :3] = (tensor_data[:, :3] - pos_mean) / pos_std
            tensor_data[:, 3:] = (tensor_data[:, 3:] - vel_mean) / vel_std
            return tensor_data.unsqueeze(0).to(device)
        return None

    # -------------------------------------------------------------------------
    # 模型推理
    # -------------------------------------------------------------------------

    def _model_infer(self, input_tensor, state):
        """纯模型推理（线程安全，不修改实例状态）。

        Args:
            input_tensor: (1, 20, 6) 归一化输入
            state: LSTM 隐状态

        Returns:
            Point3 列表，反标准化后的预测位置
        """
        with torch.no_grad():
            pred, _ = net(input_tensor, state, epoch=0, y=None, mode="val")
            pred = pred.squeeze(0).cpu().detach().numpy()
            pred = (pred * pos_std) + pos_mean
            return self._np_to_panda3d_list(pred)

    # -------------------------------------------------------------------------
    # 核心循环
    # -------------------------------------------------------------------------

    def _get_pos(self, task):
        """每帧执行：获取位置 → 更新缓冲区 → 并行推理 → 可视化更新 → 冲突检测。"""
        now_pos1 = self.quad_model.getPos(self.render)
        now_pos2 = self.uav_model.getPos(self.render)
        self._check_collision(now_pos1, now_pos2)

        # 准备两机的输入 tensor（主线程，快速）
        input1 = self._update_buffer(now_pos1, self.pos_buffer)
        input2 = self._update_buffer(now_pos2, self.pos_buffer2)

        # 并行推理（线程池）
        pred1, pred2 = None, None
        if input1 is not None and input2 is not None:
            future1 = self.executor.submit(self._model_infer, input1, self.state1)
            future2 = self.executor.submit(self._model_infer, input2, self.state2)
            pred1 = future1.result()
            pred2 = future2.result()
        elif input1 is not None:
            pred1 = self._model_infer(input1, self.state1)
        elif input2 is not None:
            pred2 = self._model_infer(input2, self.state2)

        # 可视化更新（主线程）
        if pred1 is not None:
            self._update_pred_trail(pred1)
        if pred2 is not None:
            self._update_pred_trail2(pred2)

        # 冲突检测
        if pred1 is not None and pred2 is not None:
            self._detect_conflict(pred1, pred2)

        return task.cont

    # -------------------------------------------------------------------------
    # 历史轨迹绘制
    # -------------------------------------------------------------------------

    def _draw_trail(self, task):
        """绘制无人机 1 的历史轨迹（绿色）。"""
        drone_pos = self.quad_model.getPos(self.render)
        if self.last_pos is not None:
            self.line_segs.moveTo(self.last_pos)
            self.line_segs.drawTo(drone_pos)
            self.line_segs.create(self.trail_node)
        self.last_pos = drone_pos
        return task.cont

    def _draw_trail2(self, task):
        """绘制无人机 2 的历史轨迹（蓝色）。"""
        uav_pos = self.uav_model.getPos(self.render)
        if self.last_pos2 is not None:
            self.line_segs2.moveTo(self.last_pos2)
            self.line_segs2.drawTo(uav_pos)
            self.line_segs2.create(self.trail_node2)
        self.last_pos2 = uav_pos
        return task.cont

    # -------------------------------------------------------------------------
    # 预测轨迹可视化
    # -------------------------------------------------------------------------

    def _update_pred_trail(self, pred):
        """更新无人机 1 的预测轨迹（红色）。"""
        self.line_pred.reset()
        self.line_pred.setColor(1, 0, 0, 1)
        self.line_pred.setThickness(30)
        self.line_pred.moveTo(pred[0])
        for point in pred[1:]:
            self.line_pred.drawTo(point)
        self.pred_np.removeNode()
        new_node = self.line_pred.create()
        self.pred_np = self.render.attachNewNode(new_node)

    def _update_pred_trail2(self, pred):
        """更新无人机 2 的预测轨迹（黄色）。"""
        self.line_pred2.reset()
        self.line_pred2.setColor(1, 1, 0, 1)
        self.line_pred2.setThickness(30)
        self.line_pred2.moveTo(pred[0])
        for point in pred[1:]:
            self.line_pred2.drawTo(point)
        self.pred_np2.removeNode()
        new_node = self.line_pred2.create()
        self.pred_np2 = self.render.attachNewNode(new_node)

    # -------------------------------------------------------------------------
    # 螺旋桨旋转
    # -------------------------------------------------------------------------

    def _rotate_props(self, task):
        """旋转两架无人机的螺旋桨。"""
        for i, props in enumerate(zip(self.propellers_1, self.propellers_2)):
            p1, p2 = props
            # 对角螺旋桨旋转方向相反（模拟真实四旋翼）
            direction = 1 if i < 2 else -1
            p1.setH(p1.getH() + 15 * direction)
            p2.setH(p2.getH() + 15 * direction)
        return task.cont

    # -------------------------------------------------------------------------
    # 冲突与碰撞检测
    # -------------------------------------------------------------------------

    def _detect_conflict(self, pred1, pred2):
        """基于预测轨迹逐时间步进行冲突检测。

        若未来任意时刻两机预测位置距离 < 阈值，输出告警。

        Args:
            pred1: 无人机 1 预测位置列表 (Point3)
            pred2: 无人机 2 预测位置列表 (Point3)
        """
        now_str = datetime.now().strftime("%H:%M:%S.%f")
        conflicts = []

        for t, (p1, p2) in enumerate(zip(pred1, pred2)):
            dist = (p1 - p2).length()
            if dist < self.CONFLICT_THRESHOLD:
                conflicts.append((t, dist))
                print(f"[冲突告警] 无人机1-2 距离={dist:.1f}m "
                      f"< 安全阈值{self.CONFLICT_THRESHOLD}m, 时间: {now_str}")

        if not conflicts:
            print(f"[安全] 预测轨迹全程距离 > {self.CONFLICT_THRESHOLD}m")

        return conflicts

    def _check_collision(self, pos1, pos2):
        """实时碰撞检测（基于当前位置的距离判断）。

        Args:
            pos1: 无人机 1 当前位置 (Point3)
            pos2: 无人机 2 当前位置 (Point3)
        """
        now_str = datetime.now().strftime("%H:%M:%S.%f")
        dist = (pos1 - pos2).length()
        if dist < self.COLLISION_THRESHOLD:
            print(f"[碰撞发生] 时间: {now_str}")

    # -------------------------------------------------------------------------
    # 坐标转换工具
    # -------------------------------------------------------------------------

    @staticmethod
    def _np_to_panda3d_list(np_matrix):
        """将 numpy (N, 3) 矩阵转换为 Point3 列表。"""
        return [Point3(float(row[0]), float(row[1]), float(row[2]))
                for row in np_matrix]


# =============================================================================
# 启动
# =============================================================================

if __name__ == "__main__":
    app = ConflictDetectionApp()
    app.run()
