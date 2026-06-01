# -*- coding: utf-8 -*-
"""
Seq2Seq 轨迹预测模型（冲突检测版本）
====================================
基于编码器-解码器架构的轨迹预测模型，专用于实时冲突检测场景。
特点：
  - 双向 LSTM 编码器（将双向输出投影为单向隐状态）
  - Luong 注意力机制
  - 残差连接（预测位置变化量）
  - Scheduled Sampling
"""

import torch
from torch import nn


class Encoder(nn.Module):
    """序列编码器。支持 bi-lstm 和单向 lstm。"""

    def __init__(self, input_size, hidden_size, num_layers, rnn_type="bi-lstm"):
        super().__init__()
        self.rnn_type = rnn_type

        if rnn_type == "bi-lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.2, bidirectional=True)
            self.output_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.h_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.c_proj = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.2)
            self.output_proj = None
            self.h_proj = None
            self.c_proj = None

    def forward(self, x, state):
        if self.rnn_type == "bi-lstm":
            encoder_outputs, (h, c) = self.rnn(x, state)
            encoder_outputs = self.output_proj(encoder_outputs)

            num_layers = h.shape[0] // 2
            h = h.view(num_layers, 2, -1, h.shape[-1])
            h = torch.cat([h[:, 0], h[:, 1]], dim=-1)
            h = torch.tanh(self.h_proj(h))
            c = c.view(num_layers, 2, -1, c.shape[-1])
            c = torch.cat([c[:, 0], c[:, 1]], dim=-1)
            c = torch.tanh(self.c_proj(c))
            return encoder_outputs, (h, c)
        else:
            encoder_outputs, (h, c) = self.rnn(x, state)
            return encoder_outputs, (h, c)


class Decoder(nn.Module):
    """序列解码器（LSTM）。"""

    def __init__(self, de_input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(de_input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)

    def forward(self, x, state):
        y_hat, new_state = self.rnn(x, state)
        return y_hat, new_state


class Seq2Seq(nn.Module):
    """Seq2Seq 轨迹预测模型（冲突检测版）。

    与训练版的区别：
      - 默认使用 bi-lstm 编码器
      - 解码器输出加入残差连接（预测位置变化量 + 当前输入）
    """

    def __init__(self, input_size, de_input_size, hidden_size, output_size,
                 num_layers, linear_size, pred_steps, train_steps,
                 rnn_type="bi-lstm"):
        super().__init__()
        self.output_size = output_size
        self.pred_steps = pred_steps
        self.train_steps = train_steps

        self.encoder = Encoder(input_size, hidden_size, num_layers,
                               rnn_type=rnn_type if rnn_type == "bi-lstm" else None)
        self.decoder = Decoder(de_input_size, hidden_size, num_layers)

        self.dropout = nn.Dropout(0.2)
        self.linear_out = nn.Linear(hidden_size * 2, linear_size)
        self.tanh = nn.Tanh()
        self.layer_norm = nn.LayerNorm(linear_size)
        self.linear = nn.Linear(linear_size, output_size)

    def luong_attention(self, encoder_outputs, decoder_hidden):
        """Luong 点积注意力。

        Args:
            decoder_hidden:   (B, 1, H)
            encoder_outputs:  (B, T, H)

        Returns:
            注意力输出 (B, linear_size)
        """
        attn_scores = torch.bmm(decoder_hidden,
                                encoder_outputs.transpose(1, 2)).squeeze(1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = self.dropout(attn_weights)
        context = torch.bmm(attn_weights.unsqueeze(1),
                            encoder_outputs).squeeze(1)
        output = torch.cat((decoder_hidden.squeeze(1), context), dim=-1)
        output = self.linear_out(output)
        output = self.tanh(output)
        output = self.layer_norm(output)
        return output

    def forward(self, x, state, epoch=0, y=None, mode=None):
        """前向传播。

        Args:
            x:     (B, train_steps, input_size)
            state: 编码器初始隐状态
            epoch: 当前训练轮数
            y:     真值未来轨迹
            mode:  "train" / "val"

        Returns:
            outputs:   (B, pred_steps, output_size)
            new_state: 最终 (h, c) 隐状态
        """
        batch_size = x.shape[0]

        encoder_outputs, (h, c) = self.encoder(x, state)
        outputs = torch.empty(batch_size, self.pred_steps, self.output_size,
                              device=x.device)
        decoder_input = x[:, -1, :3].unsqueeze(1)

        if mode == "val":
            tf_ratio = 0.0

            for t in range(self.pred_steps):
                decoder_hidden, (h, c) = self.decoder(decoder_input, (h, c))
                decoder_output = self.luong_attention(encoder_outputs, decoder_hidden)
                final_output = self.linear(decoder_output) + decoder_input.squeeze(1)
                outputs[:, t, :] = final_output
                if torch.rand(1).item() < tf_ratio:
                    decoder_input = y[:, t, :3].unsqueeze(1)
                else:
                    decoder_input = outputs[:, t, :].unsqueeze(1).detach()

        elif mode == "train":
            tf_ratio = max(0, 0.96 ** epoch)

            for t in range(self.pred_steps):
                decoder_hidden, (h, c) = self.decoder(decoder_input, (h, c))
                decoder_output = self.luong_attention(encoder_outputs, decoder_hidden)
                final_output = self.linear(decoder_output) + decoder_input.squeeze(1)
                outputs[:, t, :] = final_output

                if torch.rand(1).item() < tf_ratio:
                    decoder_input = y[:, t, :3].unsqueeze(1)
                else:
                    decoder_input = outputs[:, t, :].unsqueeze(1).detach()

        h = h.detach()
        c = c.detach()
        return outputs, (h, c)


__all__ = ["Encoder", "Decoder", "Seq2Seq"]
