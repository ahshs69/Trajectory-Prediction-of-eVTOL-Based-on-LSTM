# -*- coding: utf-8 -*-
"""
SOC 预测与工况分类模型
======================
基于共享 LSTM 骨干的多任务学习模型：
  - SOC 回归头：预测电池荷电状态 (State of Charge)
  - 机动段分类头：识别当前飞行机动段类别

使用同方差不确定性加权自动平衡两个任务的损失。
"""

import torch
from torch import nn


class SOC_LSTM(nn.Module):
    """多任务 LSTM：共享骨干 + SOC 回归头 + 机动段分类头。

    输入 : (B, 20, input_size)   — 过去 20 步
    输出 : soc         (B, 1)          — 未来 1 步 SOC 估计（回归）
           cls_logits  (B, num_classes) — 机动段类别 logits（分类）
           log_sigma_soc  — 可学习的 SOC 任务 log 方差（同方差不确定性）
           log_sigma_cls  — 可学习的分类任务 log 方差
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, dropout=0.2):
        super().__init__()

        # ---- 共享 LSTM 骨干 ----
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        # ---- SOC 回归头 ----
        self.soc_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
        )

        # ---- 机动段分类头 ----
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes),
        )

        # ---- 可学习的任务权重（同方差不确定性）----
        self.log_sigma_soc = nn.Parameter(torch.zeros(1))
        self.log_sigma_cls = nn.Parameter(torch.zeros(1))

    def forward(self, x, state=None):
        """前向传播。

        Args:
            x:     (B, seq_len, input_size) 输入序列
            state: LSTM 初始隐状态，None 则零初始化

        Returns:
            soc:        (B, 1)           SOC 预测值
            cls_logits: (B, num_classes) 机动段分类 logits
            log_sigma_soc:  SOC 任务的 log 方差参数
            log_sigma_cls:  分类任务的 log 方差参数
        """
        # 仅使用电压/电流/温度特征（跳过时间戳和标签列）
        x = x[:, :, 1:4]
        lstm_out, _ = self.lstm(x, state)          # (B, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]            # 取最后一步隐状态

        soc = self.soc_head(last_hidden)             # (B, 1)
        cls_logits = self.cls_head(last_hidden)      # (B, num_classes)

        return soc, cls_logits, self.log_sigma_soc, self.log_sigma_cls


__all__ = ["SOC_LSTM"]
