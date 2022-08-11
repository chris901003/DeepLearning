# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.builder import build_activation_layer


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention Module. This code is adopted from
    https://github.com/jadore801120/attention-is-all-you-need-pytorch.

    Args:
        temperature (float): The scale factor for softmax input.
        attn_dropout (float): Dropout layer on attn_output_weights.
    """

    def __init__(self, temperature, attn_dropout=0.1):
        """ 已看過，處理qkv計算的部分
        Args:
            temperature: qkv公式當中的分母部分
            attn_dropout: dropout概率
        """
        # 繼承自nn.Module，對繼承對象進行初始化
        super().__init__()
        # 保存
        self.temperature = temperature
        # 構建dropout層結構
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """ 已看過，注意力機制的forward
        Args:
            q = query資料，tensor shape [batch_size, heads, seq_len, channel]
            k = key資料，tensor shape [batch_size, heads, seq_len, channel]
            v = value資料，tensor shape [batch_size, heads, seq_len, channel]
            mask = 如果有需要忽略的部分就會是0，否則就會是1
        """

        # matmul就是進行矩陣乘法，self.temperature會是head的數量，這裡做的就是q*k
        # attn shape = tensor [batch_size, heads, seq_len, seq_len]
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # 如果有傳入mask就會到這裡，將mask當中為0的部分用-inf進行替代
            attn = attn.masked_fill(mask == 0, float('-inf'))

        # 將attn通過softmax並且進行dropout
        attn = self.dropout(F.softmax(attn, dim=-1))
        # 最後再乘上value，output shape = [batch_size, heads, seq_len, channel]
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module.

    Args:
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
    """

    def __init__(self,
                 n_head=8,
                 d_model=512,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False):
        """ 已看過，自注意機制的初始化部分
        Args:
            n_head: 多頭注意力機制當中的頭數
            d_model: 輸入的channel深度
            d_k: 每個注意裡頭key的channel深度
            d_v: 每個注意裡頭value的channel深度
            dropout: dropout概率
            qkv_bias: qkv是否需要添加bias
        """
        # 繼承自nn.Module，對繼承對象進行初始化
        super().__init__()
        # 保存傳入的參數
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # 總key與value的深度
        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v

        # 構建獲取qkv的全連接層結構
        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)

        # 構建qkv計算的部分
        self.attention = ScaledDotProductAttention(d_k**0.5, dropout)

        # 最終透過全連接層調整channel到d_model
        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)
        # dropout層結構
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """ 已看過，多頭自注意力機制當中的forward函數
        Args:
            q: query資料，tensor shape [batch_size, seq_len, channel]
            k: key資料，tensor shape [batch_size, seq_len, channel]
            v: value資料，tensor shape [batch_size, seq_len, channel]
            mask: 哪些部分是需要掩蓋住的，在需要掩蓋的地方會是0，其他部分會是1，tensor shape [batch_size, seq_len]
        """
        # 獲取q的shape資訊
        batch_size, len_q, _ = q.size()
        # 獲取k的shape資訊
        _, len_k, _ = k.size()

        # 將qkv通過全連接層，之後再調整通道排列
        # [batch_size, seq_len, channel] -> [batch_size, seq_len, n_head, channel_pre_head]
        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)

        # 調整通道排列 [batch_size, seq_len, n_head, channel_pre_head] -> [batch_size, n_head, seq_len, channel_pre_head]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            # 如果有傳入mask就會到這裡
            if mask.dim() == 3:
                # 如果傳入的mask是3通道的就會在第一個維度添加一個維度
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                # 如果傳入的mask是2通道就會添加兩個維度
                mask = mask.unsqueeze(1).unsqueeze(1)

        # 進行attention向前傳遞，attn_out shape = [batch_size, n_head, seq_len, channel_pre_head]
        attn_out, _ = self.attention(q, k, v, mask=mask)

        # 進行一系列通道調整，最後shape = [batch_size, seq_len, channel]
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, len_q, self.dim_v)

        # 通過全連接層調整通道
        attn_out = self.fc(attn_out)
        # 最後會有dropout層
        attn_out = self.proj_drop(attn_out)

        # 回傳結果，attn_out shape = [batch_size, seq_len, channel]
        return attn_out


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward module.

    Args:
        d_in (int): The dimension of the input for feedforward
            network model.
        d_hid (int): The dimension of the feedforward
            network model.
        dropout (float): Dropout layer on feedforward output.
        act_cfg (dict): Activation cfg for feedforward module.
    """

    def __init__(self, d_in, d_hid, dropout=0.1, act_cfg=dict(type='Relu')):
        """ 已看過，FFN初始化
        Args:
            d_in: 輸入的channel深度
            d_hid: 中間層的channel深度
            dropout: dropout概率
            act_cfg: 激活函數設定
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        # 進行擴維
        self.w_1 = nn.Linear(d_in, d_hid)
        # 再將維度降維
        self.w_2 = nn.Linear(d_hid, d_in)
        # 構建激活函數層
        self.act = build_activation_layer(act_cfg)
        # 構建dropout層
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 已看過，就是一般的FFN結構
        # x = tensor shape [batch_size, seq_len, channel]

        # 透過一層全連接層進行channel加深
        x = self.w_1(x)
        # 經過激活函數
        x = self.act(x)
        # 透過一層全連接層進行channel調整回原來深度
        x = self.w_2(x)
        # 經過dropout層
        x = self.dropout(x)

        # 最後回傳x，shape不會產生變化
        return x


class PositionalEncoding(nn.Module):
    """Fixed positional encoding with sine and cosine functions."""

    def __init__(self, d_hid=512, n_position=200, dropout=0):
        """ 已看過，獲取固定的位置編碼，這裡是用sin與cos構建的位置編碼
        Args:
            d_hid: channel深度
            n_position: 文字序列長度
            dropout: dropout概率
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        # 構建dropout層結構
        self.dropout = nn.Dropout(p=dropout)

        # Not a parameter
        # Position table of shape (1, n_position, d_hid)
        # 構建位置編碼
        self.register_buffer(
            'position_table',
            self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        # 已看過，構建透過sin與cos構建的位置編碼
        denominator = torch.Tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = denominator.view(1, -1)
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        # sinusoid_table shape = [1, n_position, d_hid]
        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor of shape (batch_size, pos_len, d_hid, ...)
        """
        # 已看過，將傳入的特徵圖添加上位置編碼
        # x shape = [batch_size, seq_len, channel]

        # 獲取當前訓練設備
        self.device = x.device
        # 獲取需要長度的序列位置編碼，channel會在初始化時就已經固定了，這裡會將反向傳播給關閉，因為這裡用的是sin與cos構成的位置編碼
        x = x + self.position_table[:, :x.size(1)].clone().detach()
        # 最後通過dropout層
        return self.dropout(x)
