# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmocr.models.common import MultiHeadAttention


class SatrnEncoderLayer(BaseModule):
    """"""

    def __init__(self,
                 d_model=512,
                 d_inner=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 init_cfg=None):
        """ 已看過，構建SATRN的encoder層結構，這裡就會是自注意力結構
        Args:
            d_model: 輸入的channel深度
            d_inner: 在FFN中間層的channel深度
            n_head: 多頭注意力機制當中的頭數
            d_k: 每個注意裡頭key的channel深度
            d_v: 每個注意裡頭value的channel深度
            dropout: dropout率
            qkv_bias: qkv是否需要bias
            init_cfg: 初始化方式
        """
        # 繼承自BaseModule，將繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)
        # 構建LN標準化層結構實例對象
        self.norm1 = nn.LayerNorm(d_model)
        # 使用pytorch官方的自注意力模塊
        self.attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        # 第二個標準化層結構
        self.norm2 = nn.LayerNorm(d_model)
        # 這裡是FFN的改進版本，實質也是做FFN的事情
        self.feed_forward = LocalityAwareFeedforward(
            d_model, d_inner, dropout=dropout)

    def forward(self, x, h, w, mask=None):
        """ 已看過，SATRN的encoder的forward函數
        Args:
            x = 輸入的特徵圖資料，tensor shape [batch_size, height * width, channel=512]
            h = 原始特徵圖的高度
            w = 原始特徵圖的寬度
            mask = 如果有地方是填充上去的就會是0，否則就會是1
        """
        # 獲取特徵圖的shape資料
        n, hw, c = x.size()
        # 保留殘差邊的資料
        residual = x
        # 將特徵圖通過LN標準化層
        x = self.norm1(x)
        # 進行自注意力機制，同時將結果與殘差邊進行add操作，shape不會產生改變
        x = residual + self.attn(x, x, x, mask)
        # 更新殘差結構需要保存的資料
        residual = x
        # 通過LN標準化層
        x = self.norm2(x)
        # 調整通道，最終shape = [batch_size, channel, height, width]
        x = x.transpose(1, 2).contiguous().view(n, c, h, w)
        # 通過FFN層結構，這裡採用的是透過卷積進行FFN，作者認為這樣比使用全連結正確率更高同時參數量較小
        x = self.feed_forward(x)
        # 再重新調整通道，shape = [batch_size, channel, height * width = seq_len]
        x = x.view(n, c, hw).transpose(1, 2)
        # 進行殘差邊相加
        x = residual + x
        # 回傳一次注意力機制後的結果
        return x


class LocalityAwareFeedforward(BaseModule):
    """Locality-aware feedforward layer in SATRN, see `SATRN.

    <https://arxiv.org/abs/1910.04396>`_
    """

    def __init__(self,
                 d_in,
                 d_hid,
                 dropout=0.1,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Constant', layer='BatchNorm2d', val=1, bias=0)
                 ]):
        """ 已看過，SATRN當中的FFN層結構，這裡會改用conv而不是linear進行FFN結構
        Args:
            d_in: 輸入的channel深度
            d_hid: 中間層的channel深度
            dropout: dropout概率，這裡沒有用到
            init_cfg: 初始化方式
        """
        # 繼承自BaseModule，對繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)
        # 通過conv1將channel維度進行升維，這裡特徵圖高寬不變
        self.conv1 = ConvModule(
            d_in,
            d_hid,
            kernel_size=1,
            padding=0,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

        # 中間使用DW卷積，可以減少參數量
        self.depthwise_conv = ConvModule(
            d_hid,
            d_hid,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=d_hid,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

        # 最後將channel深度調整到與輸入相同
        self.conv2 = ConvModule(
            d_hid,
            d_in,
            kernel_size=1,
            padding=0,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))

    def forward(self, x):
        """ 已看過，FFN部分的forward函數
        Args:
            x = 特徵圖，tensor shape [batch_size, channel, height, width]
        """
        # 這裡透過conv的方式進行FFN層結構，而不是傳統的fc進行FFN層結構
        # 第一次使用的是正常的卷積進行升維
        x = self.conv1(x)
        # 中間使用的是DW卷積
        x = self.depthwise_conv(x)
        # 最後用普通的卷積進行降維
        x = self.conv2(x)

        return x


class Adaptive2DPositionalEncoding(BaseModule):
    """Implement Adaptive 2D positional encoder for SATRN, see
      `SATRN <https://arxiv.org/abs/1910.04396>`_
      Modified from https://github.com/Media-Smart/vedastr
      Licensed under the Apache License, Version 2.0 (the "License");
    Args:
        d_hid (int): Dimensions of hidden layer.
        n_height (int): Max height of the 2D feature output.
        n_width (int): Max width of the 2D feature output.
        dropout (int): Size of hidden layers of the model.
    """

    def __init__(self,
                 d_hid=512,
                 n_height=100,
                 n_width=100,
                 dropout=0.1,
                 init_cfg=[dict(type='Xavier', layer='Conv2d')]):
        """ 已看過，自適應構建2D位置編碼，專門給SATRN使用
        Args:
            d_hid: 隱藏層的channel深度
            n_height: 2D特徵圖的最大高度
            n_width: 2D特徵圖的最大寬度
            dropout: dropout概率
            init_cfg: 初始化設定
        """
        # 繼承自BaseModule，將繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)

        # 透過_get_sinusoid_encoding_table獲取高度的位置編碼
        # h_position_encoder shape = [n_height, d_hid]
        h_position_encoder = self._get_sinusoid_encoding_table(n_height, d_hid)
        # [n_height, d_hid] -> [d_hid, n_height]
        h_position_encoder = h_position_encoder.transpose(0, 1)
        # [d_hid, n_height] -> [1, d_hid, n_height, 1]
        h_position_encoder = h_position_encoder.view(1, d_hid, n_height, 1)

        # 透過_get_sinusoid_encoding_table獲取寬度的位置編碼
        # w_position_encoder shape = [n_width, d_hid]
        w_position_encoder = self._get_sinusoid_encoding_table(n_width, d_hid)
        # [n_width, d_hid] -> [d_hid, n_width]
        w_position_encoder = w_position_encoder.transpose(0, 1)
        # [d_hid, n_width] -> [1, d_hid, 1, n_width]
        w_position_encoder = w_position_encoder.view(1, d_hid, 1, n_width)

        # 將高寬的位置編碼放到register_buffer當中
        self.register_buffer('h_position_encoder', h_position_encoder)
        self.register_buffer('w_position_encoder', w_position_encoder)

        # 構建高寬的縮放比例
        self.h_scale = self.scale_factor_generate(d_hid)
        self.w_scale = self.scale_factor_generate(d_hid)
        # 構建pool層結構
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 構建dropout層
        self.dropout = nn.Dropout(p=dropout)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        # 已看過，獲取正弦曲線與餘弦的位置編碼表
        # n_position = 總共的長度
        # d_hid = channel深度

        # 獲取分母部分的資料，這裡可以對照公式，denominator = tensor shape [d_hid]
        denominator = torch.Tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        # 調整通道，denominator shape = [1, d_hid]
        denominator = denominator.view(1, -1)
        # 獲取分子的部分，這裡會有[0, n_position)，pos_tensor shape = [n_position, 1]
        pos_tensor = torch.arange(n_position).unsqueeze(-1).float()
        # 將pos_tensor與denominator相乘就會是分子與分母的結合
        sinusoid_table = pos_tensor * denominator
        # 將index為偶數的部分通過sin與index為奇數的部分通過cos就可以獲得最終的位置編碼
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])

        # 回傳位置編碼表，sinusoid_table shape = [n_position, d_hid]
        return sinusoid_table

    def scale_factor_generate(self, d_hid):
        # 已看過，構建縮放因子
        # 通過兩個卷積層中間會有relu激活層，最後會通過Sigmoid進行激活
        scale_factor = nn.Sequential(
            nn.Conv2d(d_hid, d_hid, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(d_hid, d_hid, kernel_size=1), nn.Sigmoid())

        return scale_factor

    def forward(self, x):
        # 已看過，給傳入的特徵圖加上位置編碼
        # x = 特徵圖，tensor shape [batch_size, channel, height, width]

        # 獲取x的資訊
        b, c, h, w = x.size()

        # 將x通過一個平均池化層將高寬都變成1，tensor shape [batch_size, channel, height=1, width=1]
        avg_pool = self.pool(x)

        # 獲取高度方面的位置編碼，這裡會取出需要的高度的量
        # h_scale與w_scale會是將avg_pool的資料通過兩個卷積層以及激活函數，其中最後一個激活函數是Sigmoid
        h_pos_encoding = \
            self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        # 獲取寬度方面的位置編碼，這裡會取出需要的寬度的量
        w_pos_encoding = \
            self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]

        # h_pos_encoding shape = [batch_size, channel, height, width=1]
        # w_pos_encoding shape = [batch_size, channel, height=1, width]
        # 使用add方式融合
        out = x + h_pos_encoding + w_pos_encoding

        # 最後通過dropout層
        out = self.dropout(out)

        # 最後輸出
        return out
