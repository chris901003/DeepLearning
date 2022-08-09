# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        """ 已看過，雙向LSTM初始化部分
        Args:
            nIN: 輸入的channel深度
            nHidden: 隱藏層channel深度
            nOut: 輸出的channel深度
        """
        super().__init__()

        # 構建雙向LSTM實例化對象，nHidden就會是輸出時後的channel深度，如果有設定成雙向的LSTM就會是2倍深度
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        # 將輸出的channel調整到分類類別數，因為是雙向的所有channel深度會是兩倍的nHidden
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
