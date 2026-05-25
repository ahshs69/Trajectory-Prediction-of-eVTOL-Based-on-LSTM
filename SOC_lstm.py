# -*- coding: utf-8 -*-

import torch
from torch import nn


class SOC_LSTM(nn.Module):
    """多任务 LSTM: 共享骨干 + SOC 回归头 + 机动段分类头

    Input : (B, 20, input_size)   — 过去 20 步
    Output: soc   (B, 1)          — 未来 1 步 SOC 估计 (回归)
            cls   (B, num_classes) — 机动段类别 logits (分类)
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, dropout=0.2):
        super().__init__()

        # 共享 LSTM 骨干
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

        # SOC 回归头
        self.soc_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
        )

        # 机动段分类头
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_classes),
        )
        self.log_sigma_soc = nn.Parameter(torch.zeros(1))  # 可学习的 log 方差
        self.log_sigma_cls = nn.Parameter(torch.zeros(1))

    def forward(self, x, state=None):
        """
        Args:
            x:     (B, seq_len, input_size)
            state: LSTM 初始隐状态，None 则零初始化
        Returns:
            soc: (B, 1)           SOC 预测值
            cls: (B, num_classes) 机动段分类 logits
        """
        x = x[:,:,1:4]
        lstm_out, _ = self.lstm(x, state)          # (B, 20, hidden_size)
        last_hidden = lstm_out[:, -1, :]            # 取最后一步的隐状态

        soc = self.soc_head(last_hidden)             # (B, 1)
        cls_logits = self.cls_head(last_hidden) 
        
        log_sigma_soc = torch.clamp(self.log_sigma_soc, min=-10, max=10)
        log_sigma_cls = torch.clamp(self.log_sigma_cls, min=-10, max=10)

        
        return soc, cls_logits,log_sigma_soc,log_sigma_cls


__all__ = ["SOC_LSTM"]