# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        # 如果channel深度沒辦法被頭數整除就會有問題
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        # 每一個注意力頭的channel深度為多少
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        # 當每個頭的channel深度不是2的倍數的時候會有警告，如果是2的倍數會對設備比較友善，可以有提升速度的效果
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a "
                          "power of 2 " "which is more efficient in our CUDA implementation.")

        # 用於cuda實現
        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # 採樣點的座標偏移調整，每個query在每個注意力頭和每個特徵層都需要採樣n_points個
        # 由於(x, y)座標都有對應的偏移量，所以需要乘以2
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        # 每個query對應的所有採樣點的注意力權重
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        # 透過線性變換獲得value
        self.value_proj = nn.Linear(d_model, d_model)
        # 最後還需要透過一個線性變換得到最後輸出
        self.output_proj = nn.Linear(d_model, d_model)

        # 初始化，這裡挺重要的
        self._reset_parameters()

    def _reset_parameters(self):
        # 將sampling_offsets的值先全部設定成0
        constant_(self.sampling_offsets.weight.data, 0.)
        # 下面是在初始化偏移量預測的偏置(bias)，使得初始偏移位置猶如不同大小的方形卷積組合
        # thetas shape [8] (0, pi/4, 2*pi/4, 3*pi/4, ..., 7*pi/4)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        # thetas shape [8, 2]，透過cos以及sin組成
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        # grid_init / grid_init.abs().max(-1, keepdim=True)[0] = 這步計算得到8個頭對應的座標偏移
        # (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)
        # 然後再將這些偏移repeat到所有層當中
        # grid_init shape [8, 4, 4, 2] (n_head, n_levels, n_points, (x, y))
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)\
            .repeat(1, self.n_levels, self.n_points, 1)
        # 同一特徵層中不同採樣點的座標偏移量肯定不能夠一樣，因此這裡做了處理
        # 對於第i個採樣點，在8個頭部和所有特徵層中，其座標偏移為:
        # (i, 0), (i, i), (0, i), (-i, i), (-i, 0), (-i, -i), (0, -i), (i, -i)
        # 從圖形上來看，形成的位置偏移相當於是3x3、5x5、7x7、9x9正方形卷積核(去除中心點)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        # 這裡取消梯度將設定好的bias給sampling_offsets的bias
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # 將剩下的東西進行初始化
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        '''  Multi-Scale Deformable Attention 主要做以下事情
        (1). 將輸入input_flatten(對於encoder就是由backbone輸出的特徵層變換而來，對於decoder就是encoder輸出的memory)
             透過變換矩陣得到value，同時將padding的部分填充為0
        (2). 將query(對於encoder就是特徵圖本身加上position和scale-level embedding，
             對於decoder就是self-attention的輸出加上position embedding的結果)，
             2-stage時這個position embedding是由encoder預測的top-k proposal boxes進行position embedding而來，
             1-stage時是預設的query embedding經過兩個全連接層得到採樣點對應座標偏移和注意力權重(注意力權重會歸一化)
        (3). 根據參考點(reference points:對於decoder來說，2-stage時是encoder預測的top-k proposal boxes，1-stage時是由預設的
             query embedding經過全連接層得到。兩種情況下最終都經過sigmoid進行歸一化，對於encoder來說，就是各特徵點在所有特徵層對應
             的歸一化中心座標)，座標和預測座標偏移得到採樣點座標
        (4). 由採樣點座標在value中差值採樣出對應特徵向量，然後施加注意力權重，最後將這個結果經過一個全連接層輸出得到最後結果
        '''
        # ---------------------------------------------------------------
        # 以下是從backbone出來的
        # query shape [batch_size, total_pixel, channel]
        # reference_points shape (1-stage = [batch_size, total_pixel, lvl, 2],
        #                         2-stage = [batch_size, total_pixel, lvl, 4])
        # input_flatten shape [batch_size, total_pixel, channel]
        # input_spatial_shapes shape [lvl, 2]
        # input_level_start_index shape [lvl]
        # input_padding_mask shape [batch_size, total_pixel]
        # ---------------------------------------------------------------

        # 獲取輸入相關資料
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        # 這個值需要是所有特徵層特徵點的數量
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # 將輸入的值透過一個全連接層後變成注意力機制中的value部分
        # value shape [batch_size, total_pixel, channel=256]
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            # 將原圖padding部分用0填充，透過masked_fill會將input_padding_mask為True的地方用0取代
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        # value shape [batch_size, total_pixel, channel=256] -> [batch_size, total_pixel, heads=8, channel=32]
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # sampling_offsets shape [batch_size, Len_q=(total_pixel或num_queries), heads=8, levels=4, points=4, 2]
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        # attention_weights shape [batch_size, Len_q=(total_pixel或num_queries), heads=8, levels*points=16]
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # 對最後一個維度進行softmax表示每一個點的權重，最後權重總和會是1
        # attention_weights shape [batch_size, Len_q=(total_pixel或num_queries), heads=8, levels=4, points=4]
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        # 使用1-stage與2-stage會有不同的情況，在這裡會分成兩種情況討論
        # sampling_locations shape [batch_size, Len_q, n_heads, n_levels, n_points, 2]
        if reference_points.shape[-1] == 2:
            # offset_normalizer shape [lvl=4, 2]，2表示的是(w, h)
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # [bs, Len_q, 1, levels, 1, 2] + [bs, Len_q, heads, levels, points, 2] / [1, 1, 1, levels, 1, 2]
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            # 2-stage會走這裡
            # 這裡分別取出的是(x, y)，下面的是取出(w, h)
            # 最後變成sampling_locations，可以跟上面的1-stage稍微對比一下
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            # 其他情況就直接報錯
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        # 根據採樣點位置拿出對應的value，並且施加預測出來的注意力權重(和value進行weight sum)
        # 注：實質是調用cuda版本的需要進行編譯，這裡如果使用window會有很多問題，建議使用linux版本
        # output shape [batch_size, Len_in, channel=256]
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
            self.im2col_step)
        # output shape [batch_size, Len_in, channel=256]
        output = self.output_proj(output)
        return output
