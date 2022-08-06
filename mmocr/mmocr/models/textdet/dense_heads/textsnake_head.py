# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
from mmcv.runner import BaseModule

from mmocr.models.builder import HEADS
from .head_mixin import HeadMixin


@HEADS.register_module()
class TextSnakeHead(HeadMixin, BaseModule):
    """The class for TextSnake head: TextSnake: A Flexible Representation for
    Detecting Text of Arbitrary Shapes.

    TextSnake: `A Flexible Representation for Detecting Text of Arbitrary
    Shapes <https://arxiv.org/abs/1807.01544>`_.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downsample_ratio (float): Downsample ratio.
        loss (dict): Configuration dictionary for loss type.
        postprocessor (dict): Config of postprocessor for TextSnake.
        train_cfg, test_cfg: Depreciated.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 out_channels=5,
                 downsample_ratio=1.0,
                 loss=dict(type='TextSnakeLoss'),
                 postprocessor=dict(
                     type='TextSnakePostprocessor', text_repr_type='poly'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     override=dict(name='out_conv'),
                     mean=0,
                     std=0.01),
                 **kwargs):
        """ 已看過，TextSnake的預測頭，將特徵圖變成最後的預測圖
        Args:
            in_channels: 輸入特徵圖channel深度
            out_channels: 輸出channel深度，也就是最後需要的channel深度
            downsample_ratio: 下採樣倍率
            loss: 損失計算設定
            postprocessor: 後處理方式
            train_cfg: train時其他額外的設定
            test_cfg: test時其他額外的設定
            init_cfg: 初始化方式
        """
        # 檢查一些已經遺棄的參數是否有在kwargs當中
        old_keys = ['text_repr_type', 'decoding_type']
        for key in old_keys:
            if kwargs.get(key, None):
                postprocessor[key] = kwargs.get(key)
                warnings.warn(
                    f'{key} is deprecated, please specify '
                    'it in postprocessor config dict. See '
                    'https://github.com/open-mmlab/mmocr/pull/640 '
                    'for details.', UserWarning)
        # 繼承自BaseModule，對繼承對象進行初始化
        BaseModule.__init__(self, init_cfg=init_cfg)
        # 繼承自HeadMixin，對繼承對象進行初始化
        HeadMixin.__init__(self, loss, postprocessor)

        # 檢查in_channels需要是int
        assert isinstance(in_channels, int)
        # 保存傳入的參數
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_ratio = downsample_ratio
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 構建一個卷積實例化對象，會將channel調整到最終輸出的channel深度
        self.out_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): Shape :math:`(N, C_{in}, H, W)`, where
                :math:`C_{in}` is ``in_channels``. :math:`H` and :math:`W`
                should be the same as the input of backbone.

        Returns:
            Tensor: A tensor of shape :math:`(N, 5, H, W)`.
        """
        # 已看過，會將channel深度調整到最後需要的深度
        outputs = self.out_conv(inputs)
        return outputs
