# -*- coding: utf-8 -*-
"""
Seq2Seq 轨迹预测模型
====================
基于编码器-解码器架构的轨迹预测模型，支持：
  - 双向/单向 LSTM、RNN、GRU 编码器
  - Luong 注意力机制
  - Scheduled Sampling（训练时逐步降低 teacher forcing 比例）
  - 残差连接（预测位置变化量，收敛更快）
"""

import torch
from torch import nn


class Encoder(nn.Module):
    """序列编码器。

    支持 bi-lstm / lstm / rnn / gru 四种类型。
    bi-lstm 模式下将双向输出投影为单向隐状态供解码器使用。
    """

    def __init__(self, input_size, hidden_size, num_layers, rnn_type="bi-lstm"):
        super().__init__()
        self.rnn_type = rnn_type

        if rnn_type == "bi-lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.2, bidirectional=True)
            # 将双向输出投影为单向维度
            self.output_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.h_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.c_proj = nn.Linear(hidden_size * 2, hidden_size)
        elif rnn_type == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=0.2)
            self.output_proj = None
            self.h_proj = None
            self.c_proj = None
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=0.2)
            self.output_proj = None
            self.h_proj = None
            self.c_proj = None
        else:
            # 默认单向 LSTM
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.2)
            self.output_proj = None
            self.h_proj = None
            self.c_proj = None

    def forward(self, x, state):
        if self.rnn_type == "bi-lstm":
            encoder_outputs, (h, c) = self.rnn(x, state)
            # 投影编码器输出: (B, T, 2*H) → (B, T, H)
            encoder_outputs = self.output_proj(encoder_outputs)
            # 投影解码器初始状态
            num_layers = h.shape[0] // 2
            h = h.view(num_layers, 2, -1, h.shape[-1])       # (L, 2, B, H)
            h = torch.cat([h[:, 0], h[:, 1]], dim=-1)         # (L, B, 2*H)
            h = torch.tanh(self.h_proj(h))                     # (L, B, H)
            c = c.view(num_layers, 2, -1, c.shape[-1])
            c = torch.cat([c[:, 0], c[:, 1]], dim=-1)
            c = torch.tanh(self.c_proj(c))
            return encoder_outputs, (h, c)
        else:
            encoder_outputs, new_state = self.rnn(x, state)
            return encoder_outputs, new_state


class Decoder(nn.Module):
    """序列解码器。

    支持 lstm / rnn / gru 三种类型。
    """

    def __init__(self, de_input_size, hidden_size, num_layers, rnn_type="lstm"):
        super().__init__()
        if rnn_type == "rnn":
            self.rnn = nn.RNN(de_input_size, hidden_size, num_layers,
                              batch_first=True, dropout=0.2)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(de_input_size, hidden_size, num_layers,
                              batch_first=True, dropout=0.2)
        else:
            self.rnn = nn.LSTM(de_input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.2)

    def forward(self, x, state):
        y_hat, new_state = self.rnn(x, state)
        return y_hat, new_state


class Seq2Seq(nn.Module):
    """Seq2Seq 轨迹预测模型。

    编码器处理历史轨迹，解码器自回归生成未来轨迹。
    注意力机制为 Luong 风格（点积注意力 + 上下文拼接）。
    """

    def __init__(self, input_size, de_input_size, hidden_size, output_size,
                 num_layers, linear_size, pred_steps, train_steps,
                 rnn_type="bi-lstm"):
        super().__init__()
        self.output_size = output_size
        self.pred_steps = pred_steps
        self.train_steps = train_steps

        self.encoder = Encoder(input_size, hidden_size, num_layers, rnn_type)
        self.decoder = Decoder(de_input_size, hidden_size, num_layers, rnn_type)
        self.dropout = nn.Dropout(0.2)
        self.linear_out = nn.Linear(hidden_size * 2, linear_size)
        self.tanh = nn.Tanh()
        self.layer_norm = nn.LayerNorm(linear_size)
        self.linear = nn.Linear(linear_size, output_size)

    def luong_attention(self, encoder_outputs, decoder_hidden):
        """Luong 点积注意力。

        Args:
            decoder_hidden:   (B, 1, H) 解码器当前隐状态
            encoder_outputs:  (B, T, H) 编码器全部输出

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
            x:     (B, train_steps, input_size) 输入历史轨迹
            state: 编码器初始隐状态
            epoch: 当前训练轮数（用于 scheduled sampling 衰减）
            y:     (B, pred_steps, 3) 真值未来轨迹（训练时用于 teacher forcing）
            mode:  "train" / "val" — 决定 teacher forcing 比例

        Returns:
            outputs:   (B, pred_steps, output_size) 预测的未来轨迹
            new_state: 最终解码器隐状态
        """
        batch_size = x.shape[0]

        encoder_outputs, new_state = self.encoder(x, state)
        outputs = torch.empty(batch_size, self.pred_steps, self.output_size,
                              device=x.device)
        # 解码器初始输入为输入序列最后一步的位置
        decoder_input = x[:, -1, :3].unsqueeze(1)

        if mode == "val":
            tf_ratio = 0.0  # 纯自回归

            for t in range(self.pred_steps):
                decoder_hidden, new_state = self.decoder(decoder_input, new_state)
                decoder_output = self.luong_attention(encoder_outputs, decoder_hidden)
                final_output = self.linear(decoder_output)
                outputs[:, t, :] = final_output
                # teacher forcing（验证时关闭）
                if torch.rand(1).item() < tf_ratio:
                    decoder_input = y[:, t, :3].unsqueeze(1)
                else:
                    decoder_input = outputs[:, t, :].unsqueeze(1).detach()

        elif mode == "train":
            # scheduled sampling: tf_ratio 从 1.0 指数衰减到 0
            tf_ratio = max(0, 0.96 ** epoch)

            for t in range(self.pred_steps):
                decoder_hidden, new_state = self.decoder(decoder_input, new_state)
                decoder_output = self.luong_attention(encoder_outputs, decoder_hidden)
                final_output = self.linear(decoder_output)
                outputs[:, t, :] = final_output

                if torch.rand(1).item() < tf_ratio:
                    decoder_input = y[:, t, :3].unsqueeze(1)
                else:
                    decoder_input = outputs[:, t, :].unsqueeze(1).detach()

            new_state[0].detach_()
            new_state[1].detach_()

        return outputs, new_state


__all__ = ["Encoder", "Decoder", "Seq2Seq"]
