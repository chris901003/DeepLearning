import torch
from torch import nn


class RemainTimeRegression(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, remain_time_classes):
        super(RemainTimeRegression, self).__init__()
        # 這裡的104表示，剩餘量只會有從[0, 100]的範圍之後再加上起始以及結束以及padding，所以這部分會是固定的
        self.remain_embed = nn.Embedding(104, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.cls = nn.Linear(hidden_size, remain_time_classes)

    def forward(self, x):
        x = self.remain_embed(x)
        r_out, (h_state, c_state) = self.lstm(x)
        outs = list()
        for time_step in range(r_out.size(1)):
            outs.append(self.cls(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state, c_state
