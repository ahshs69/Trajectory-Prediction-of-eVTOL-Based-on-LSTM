# Trajectory Prediction of eVTOL Based on LSTM

基于深度学习的 eVTOL（电动垂直起降飞行器）轨迹预测与冲突检测系统。

## 项目概述

本项目实现了三个核心功能模块：

1. **轨迹预测** — 基于 Seq2Seq + Attention 的编码器-解码器架构，对无人机未来轨迹进行多步预测
2. **电池 SOC 估计与工况分类** — 多任务 LSTM 模型同时预测电池荷电状态 (SOC) 和识别飞行机动段类别
3. **冲突检测仿真** — 基于 Panda3D 的 3D 可视化仿真环境，实时轨迹预测与两机冲突检测

## 项目结构

```
.
├── mydataloader2.py              # 核心工具模块：数据标准化、滑动窗口加载、训练/验证/测试、评估指标
├── animation_marking.py          # 可视化工具：3D 动画、损失曲线、轨迹对比图
│
├── battery/                      # 电池 SOC 与工况分类模块
│   ├── main.py                   # 训练主脚本
│   └── SOC_lstm.py               # 多任务 LSTM 模型（SOC 回归 + 机动段分类）
│
├── Trajectory Prediction/        # 轨迹预测模块
│   ├── main.py                   # 训练主脚本
│   └── seq2seq.py                # Seq2Seq + Attention 模型
│
├── Conflict Detection/           # 冲突检测仿真模块
│   ├── run.py                    # Panda3D 仿真主程序
│   ├── seq2seq.py                # Seq2Seq 模型（冲突检测版，含残差连接）
│   ├── mydataloader2.py          # 数据工具（冲突检测版）
│   └── traj_make.py              # 直线飞行轨迹生成器
│
├── dataset/                      # 数据集
│   ├── battery.csv               # 电池原始数据
│   ├── dataset1.pkl              # 预处理后的电池数据集
│   └── posearray.pkl             # 预处理后的轨迹数据集（位于 results/）
│
└── results/                      # 训练输出
    ├── scaler.npz                # 轨迹数据标准化参数
    ├── scaler_battery.npz        # 电池数据标准化参数
    ├── time_seq_best.opt         # 最佳模型权重
    └── time_seq_final.opt        # 最终模型权重
```

## 环境依赖

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- Matplotlib (可视化)
- Panda3D (冲突检测仿真)
- tqdm (进度条)

## 模型架构

### 轨迹预测 (Seq2Seq + Attention)

```
输入: 过去 20 步位置 (x, y, z)
  ↓
Encoder (Bi-LSTM / LSTM / GRU) → 编码历史轨迹
  ↓
Luong Attention → 注意力加权上下文向量
  ↓
Decoder (LSTM) → 自回归生成未来轨迹
  ↓
输出: 未来 10 步位置 (x, y, z)
```

- **Scheduled Sampling**: 训练时 teacher forcing 比例指数衰减 (`0.96^epoch`)，验证时纯自回归
- **残差连接**: 解码器输出 = 位置变化量 + 当前输入（冲突检测版）
- **支持 RNN 类型**: Bi-LSTM、LSTM、RNN、GRU

### 电池 SOC 估计 (Multi-Task LSTM)

```
输入: 过去 20 步 (电压, 电流, 温度)
  ↓
共享 LSTM 骨干
  ↓
├── SOC 回归头 → SOC 预测值 (回归, MSE Loss)
└── 机动段分类头 → 机动段类别 (分类, CrossEntropy Loss)
```

- **同方差不确定性加权**: 自动学习两个任务的最优损失权重
- **分类任务**: 识别 4 类飞行机动段

## 快速开始

### 1. 轨迹预测训练

```bash
cd "Trajectory Prediction"
python main.py
```

### 2. 电池 SOC 训练

```bash
cd battery
python main.py
```

### 3. 冲突检测仿真

```bash
# 首先生成轨迹 CSV 文件
cd "Conflict Detection"
python traj_make.py

# 启动 Panda3D 仿真
python run.py
```

## 评估指标

### 轨迹预测
| 指标 | 说明 |
|------|------|
| MSE | 均方误差 |
| MAE | 平均绝对误差 |
| RMSE | 均方根误差 |
| R² | 决定系数 |
| Endpoint Error | 平均/最后一步欧氏距离误差 |

### 电池 SOC
| 指标 | 说明 |
|------|------|
| MSE / MAE / RMSE / R² | SOC 回归精度 |
| Accuracy / F1 / Precision | 机动段分类精度 |

## 数据来源

轨迹数据集来自 [Synthetic-UAV-Flight-Trajectories](https://huggingface.co/datasets/riotu-lab/Synthetic-UAV-Flight-Trajectories)，包含 5309 段三维轨迹数据，使用分层聚类结合动态时间规整 (DTW) 分为 circular 和 infinity-like 两类。

## 训练特性

- **混合精度训练** (AMP): 减少显存占用，加速训练
- **梯度裁剪**: `max_norm=1.0` 防止梯度爆炸
- **早停 (Early Stopping)**: 验证损失不再下降时自动停止
- **学习率调度**: Warmup + 自适应衰减（ReduceLROnPlateau）或指数衰减
- **随机种子固定**: 确保实验可复现 (`seed=42`)

## License

This project is for academic research purposes.
