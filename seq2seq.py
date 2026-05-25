# -*- coding: utf-8 -*-

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, type="bi-lstm"):
        super().__init__()
        self.type = type
        if type == "bi-lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.2, bidirectional=True)
            self.output_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.h_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.c_proj = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=0.2)
            # [优化] 单向时不需要投影层，注册为 None 避免被 optimizer 追踪
            self.output_proj = None
            self.h_proj = None
            self.c_proj = None

    def forward(self, x, state):
        if self.type == "bi-lstm":
            # state: (h, c)，各维度为 (num_layers*D, B, hidden_size)
            encoder_outputs, (h, c) = self.rnn(x, state)

            # 投影 encoder outputs: (B, T, 2*H) -> (B, T, H)
            encoder_outputs = self.output_proj(encoder_outputs)

            # 投影 decoder 初始状态
            # h/c: (num_layers*2, B, H) -> reshape -> (num_layers, B, 2*H) -> Linear -> (num_layers, B, H)
            num_layers = h.shape[0] // 2
            h = h.view(num_layers, 2, -1, h.shape[-1])   # (L, 2, B, H)
            h = torch.cat([h[:, 0], h[:, 1]], dim=-1)     # (L, B, 2*H)
            h = torch.tanh(self.h_proj(h))                 # (L, B, H)

            c = c.view(num_layers, 2, -1, c.shape[-1])
            c = torch.cat([c[:, 0], c[:, 1]], dim=-1)
            c = torch.tanh(self.c_proj(c))
            
        else:
            encoder_outputs, (h, c) = self.rnn(x, state)

        return encoder_outputs, h, c


class Decoder(nn.Module):
    def __init__(self, de_input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(de_input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)

    def forward(self, x, state):
        y_hat, new_state = self.rnn(x, state)
        return y_hat, new_state


class Seq2Seq(nn.Module):

    def __init__(self, input_size, de_input_size,hidden_size, output_size, num_layers,
                 linear_size, pred_steps, train_steps,type = "bi-lstm"):
        super().__init__()
        self.output_size = output_size
        self.pred_steps = pred_steps
        self.train_steps = train_steps
        if type == "bi-lstm":
            self.encoder = Encoder(input_size, hidden_size, num_layers,type=type)
        else:
            self.encoder = Encoder(input_size, hidden_size, num_layers,type=None)
        self.decoder = Decoder(de_input_size, hidden_size, num_layers)

        self.dropout = nn.Dropout(0.2)
        self.linear_out = nn.Linear(hidden_size * 2, linear_size)
        self.tanh = nn.Tanh()
        self.layer_norm = nn.LayerNorm(linear_size)
        self.linear = nn.Linear(linear_size, output_size)

    def Luongscore(self, encoder_outputs, decoder_hidden):
        # decoder_hidden: (B, 1, H), encoder_outputs: (B, T, H)
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

    def forward(self, x, state, epoch=0, y=None, type=None):
        # ===========================================================================
        # [Bug] train_time_seq 调用时传 epoch=0，导致 tf_ratio 恒为 0.95
        #   应改为: net(x, state, epoch=epoch, y=y, type="train") 传入实际 epoch
        # ===========================================================================
        # [优化] train/val 两个分支的循环体几乎完全相同，建议合并为一个循环
        #   仅 tf_ratio 和 decoder_input 来源不同，可用条件判断消除重复代码
        # ===========================================================================
        # [优化] tf_ratio 下限 0.15 导致训练后期仍有 15% 强制教学
        #   模型从未完全自主预测，val 时突然全自回归 => 位置误差大
        #   建议: max(0.0, 0.95**epoch) 让后期完全自回归
        # ===========================================================================
        batch_size = x.shape[0]

        encoder_outputs, h, c = self.encoder(x, state)
        outputs = torch.empty(batch_size, self.pred_steps, self.output_size,
                              device=x.device)
        decoder_input = x[:, -1, :3].unsqueeze(1)

        if type == "val":
            tf_ratio = 0.0

            for t in range(self.pred_steps):
                decoder_hidden, (h, c) = self.decoder(decoder_input, (h, c))
                decoder_output = self.Luongscore(encoder_outputs, decoder_hidden)
                final_output = self.linear(decoder_output)
                outputs[:, t, :] = final_output
                if torch.rand(1).item() < tf_ratio:
                    decoder_input = y[:, t, :3].unsqueeze(1)
                else:
                    decoder_input = outputs[:, t, :].unsqueeze(1).detach()

        elif type == "train":
            tf_ratio = max(0.1, 0.95 ** epoch)

            for t in range(self.pred_steps):
                decoder_hidden, (h, c) = self.decoder(decoder_input, (h, c))
                decoder_output = self.Luongscore(encoder_outputs, decoder_hidden)
                final_output = self.linear(decoder_output)
                outputs[:, t, :] = final_output

                if torch.rand(1).item() < tf_ratio:
                    decoder_input = y[:, t, :3].unsqueeze(1)
                else:
                    decoder_input = outputs[:, t, :].unsqueeze(1).detach()

        h = h.detach()
        c = c.detach()
        return outputs, (h, c)
