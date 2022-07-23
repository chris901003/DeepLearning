# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import Upsample
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SETRMLAHead(BaseDecodeHead):
    """Multi level feature aggretation head of SETR.

    MLA head of `SETR  <https://arxiv.org/pdf/2012.15840.pdf>`_.

    Args:
        mlahead_channels (int): Channels of conv-conv-4x of multi-level feature
            aggregation. Default: 128.
        up_scale (int): The scale factor of interpolate. Default:4.
    """

    def __init__(self, mla_channels=128, up_scale=4, **kwargs):
        """ 已看過，會將多層特徵層進行融合，最後只會輸出一個特徵層，這裡是專門給STER用的
        Args:
            mla_channels: 會將輸入圖像的channel調整到的channel深度
            up_scale: 上採樣倍率
            kwargs: 有其他許多參數這裡就不去列舉，需要就用Debug模式看一下
        """

        # 這裡繼承於BaseDecodeHead，先對繼承對象進行初始化
        super(SETRMLAHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.mla_channels = mla_channels

        num_inputs = len(self.in_channels)

        # Refer to self.cls_seg settings of BaseDecodeHead
        assert self.channels == num_inputs * mla_channels

        # 構建一系列上採樣層
        self.up_convs = nn.ModuleList()
        for i in range(num_inputs):
            self.up_convs.append(
                nn.Sequential(
                    # 會先將輸入圖像的通道數變成mla_channels
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=mla_channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    # 通過一層3*3卷積層
                    ConvModule(
                        in_channels=mla_channels,
                        out_channels=mla_channels,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    # 進行4被上採樣，這裡用的是雙線性差值方式
                    Upsample(
                        scale_factor=up_scale,
                        mode='bilinear',
                        align_corners=self.align_corners)))

    def forward(self, inputs):
        """ 已看過，這裡會將多層特徵層進行融合
        Args:
            inputs: tuple[tensor]，tensor shape [batch_size, channel, height, width]，tuple長度就是backbone輸出特徵圖的數量
        """
        # 這裡我們選擇的融合方式會讓inputs結果一樣不會改變
        inputs = self._transform_inputs(inputs)
        # 中途紀錄的地方
        outs = []
        # 遍歷所有的特徵圖
        for x, up_conv in zip(inputs, self.up_convs):
            # 透過up_conv會同時調整channel與高寬，這裡會縮減channel深度會將高寬進行4倍上採樣
            outs.append(up_conv(x))
        # 將所有特徵圖通過拼接連在一起
        out = torch.cat(outs, dim=1)
        # 透過cls_seg將channel深度調整到num_classes
        out = self.cls_seg(out)
        # out shape = [batch_size, channel=num_classes, height, width]
        return out
