# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmocr.models.builder import BACKBONES


@BACKBONES.register_module()
class ShallowCNN(BaseModule):
    """Implement Shallow CNN block for SATRN.

    SATRN: `On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention
    <https://arxiv.org/pdf/1910.04396.pdf>`_.

    Args:
        base_channels (int): Number of channels of input image tensor
            :math:`D_i`.
        hidden_dim (int): Size of hidden layers of the model :math:`D_m`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 input_channels=1,
                 hidden_dim=512,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(type='Uniform', layer='BatchNorm2d')
                 ]):
        """ 已看過，SATRN使用的淺層CNN層結構
        Args:
            input_channels: 輸入圖像的channel深度
            hidden_dim: 隱藏的channel深度
            init_cfg: 初始化設定
        """
        # 繼承自BaseModule，將繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)
        # 檢查input_channels與hidden_dim是否都為int格式
        assert isinstance(input_channels, int)
        assert isinstance(hidden_dim, int)

        # 構建卷積以及標準化以及激活層結構
        self.conv1 = ConvModule(
            input_channels,
            # 通過conv1後channel深度會是hidden_dim的一半
            hidden_dim // 2,
            kernel_size=3,
            stride=1,
            # 透過padding通過conv1後的高寬不會改變
            padding=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        # 構建第二個卷積以及標準化以及激活層結構
        self.conv2 = ConvModule(
            hidden_dim // 2,
            # 將channel深度調整到hidden_dim深度
            hidden_dim,
            kernel_size=3,
            stride=1,
            # 透過padding通過conv2後圖像大小不會改變
            padding=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        # 最後通過最大池化2倍下採樣，通過此層後高寬會是原來的一半
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input image feature :math:`(N, D_i, H, W)`.

        Returns:
            Tensor: A tensor of shape :math:`(N, D_m, H/4, W/4)`.
        """

        # 已看過，SATRN的淺層CNN的forward函數部分
        # 通過conv1將channel加深
        x = self.conv1(x)
        # 通過pool後高寬會下採樣2倍
        x = self.pool(x)

        # 再通過conv2將channel加深
        x = self.conv2(x)
        # 通過pool後高寬會下採樣2倍
        x = self.pool(x)

        # 最後回傳高寬下採樣4倍的特徵圖，tensor shape [batch_size, channel=512, height, width]
        return x
