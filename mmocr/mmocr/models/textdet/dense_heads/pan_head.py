# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmocr.models.builder import HEADS
from mmocr.utils import check_argument
from .head_mixin import HeadMixin


@HEADS.register_module()
class PANHead(HeadMixin, BaseModule):
    """The class for PANet head.

    Args:
        in_channels (list[int]): A list of 4 numbers of input channels.
        out_channels (int): Number of output channels.
        downsample_ratio (float): Downsample ratio.
        loss (dict): Configuration dictionary for loss type. Supported loss
            types are "PANLoss" and "PSELoss".
        postprocessor (dict): Config of postprocessor for PANet.
        train_cfg, test_cfg (dict): Depreciated.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample_ratio=0.25,
                 loss=dict(type='PANLoss'),
                 postprocessor=dict(
                     type='PANPostprocessor', text_repr_type='poly'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     mean=0,
                     std=0.01,
                     override=dict(name='out_conv')),
                 **kwargs):
        """ 已看過，PANHead的預測頭
        Args:
            in_channels: 輸入的channel深度，會是list且list長度就會是輸入的特徵圖數量
            out_channels: 輸出的channel深度
            downsample_ratio: 下採樣倍率
            loss: 損失函數的設定
            postprocessor: 後處理方式
            train_cfg: train的設定
            test_cfg: test的設定
            init_cfg: 初始化方式
        """

        # 一些已經被淘汰掉的key
        old_keys = ['text_repr_type', 'decoding_type']
        for key in old_keys:
            # 遍歷kwargs當中有沒有已經淘汰的key，如果有的話會將其放到postprocessor當中
            if kwargs.get(key, None):
                postprocessor[key] = kwargs.get(key)
                warnings.warn(
                    f'{key} is deprecated, please specify '
                    'it in postprocessor config dict. See '
                    'https://github.com/open-mmlab/mmocr/pull/640'
                    ' for details.', UserWarning)

        # 繼承自BaseModule，將繼承對象初始化
        BaseModule.__init__(self, init_cfg=init_cfg)
        # 繼承自HeadMixin，將繼承對象初始化
        HeadMixin.__init__(self, loss, postprocessor)

        # 檢查in_channels需要是list且裏面的值需要是int
        assert check_argument.is_type_list(in_channels, int)
        # 檢查out_channels需要是int
        assert isinstance(out_channels, int)

        # 下採樣率需要在[0, 1]
        assert 0 <= downsample_ratio <= 1

        # 保存傳入的參數
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_ratio = downsample_ratio
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 構建一個conv實例對象，將channel調整到out_channels
        self.out_conv = nn.Conv2d(
            in_channels=np.sum(np.array(in_channels)),
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, inputs):
        r"""
        Args:
            inputs (list[Tensor] | Tensor): Each tensor has the shape of
                :math:`(N, C_i, W, H)`, where :math:`\sum_iC_i=C_{in}` and
                :math:`C_{in}` is ``input_channels``.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, W, H)` where
            :math:`C_{out}` is ``output_channels``.
        """
        # 已看過，PANHead的forward函數
        # inputs shape = [batch_size, channel, height, width]

        if isinstance(inputs, tuple):
            # 如果輸入的inputs是tuple就直接從第一個維度進行concat
            outputs = torch.cat(inputs, dim=1)
        else:
            outputs = inputs
        # 透過out_conv進行特徵提取以及調整channel深度，這裡深度會是7因為會分成7次的擴大
        outputs = self.out_conv(outputs)

        return outputs
