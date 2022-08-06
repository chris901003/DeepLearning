# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from torch import nn

from mmocr.models.builder import NECKS


class UpBlock(BaseModule):
    """Upsample block for DRRG and TextSnake."""

    def __init__(self, in_channels, out_channels, init_cfg=None):
        """ 已看過，主要是給DRRG以及TextSnake使用的上採樣模塊
        Args:
            in_channels: 輸入特徵圖channel深度
            out_channels: 輸出特徵圖channel深度
            init_cfg: 初始化參數
        """
        # 繼承自BaseModule，對繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)

        # 檢查傳入的in_channels以及out_channels是否為int格式
        assert isinstance(in_channels, int)
        assert isinstance(out_channels, int)

        # 透過1*1卷積提取特徵
        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 透過3*3卷積擴大channel維度
        self.conv3x3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 最後透過轉至卷積將特徵圖進行2倍上採樣
        self.deconv = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # 已看過，UpBlock的forward函數，主要是將兩個concat後的特徵圖提取特徵後調整channel深度
        x = F.relu(self.conv1x1(x))
        x = F.relu(self.conv3x3(x))
        # 通過deconv後會上採樣2倍
        x = self.deconv(x)
        return x


@NECKS.register_module()
class FPN_UNet(BaseModule):
    """The class for implementing DRRG and TextSnake U-Net-like FPN.

    DRRG: `Deep Relational Reasoning Graph Network for Arbitrary Shape
    Text Detection <https://arxiv.org/abs/2003.07493>`_.

    TextSnake: `A Flexible Representation for Detecting Text of Arbitrary
    Shapes <https://arxiv.org/abs/1807.01544>`_.

    Args:
        in_channels (list[int]): Number of input channels at each scale. The
            length of the list should be 4.
        out_channels (int): The number of output channels.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 init_cfg=dict(
                     type='Xavier',
                     layer=['Conv2d', 'ConvTranspose2d'],
                     distribution='uniform')):
        """ 已看過，主要是給DRRG以及TextSnake使用的neck模塊，採用UNet的特徵融合方式，將不同下採樣倍率的特徵圖進行融合
        Args:
            in_channels: 輸入特徵圖channel深度
            out_channels: 輸出特徵圖channel深度
            init_cfg: 初始化方式
        """
        # 繼承自BaseModule，將繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)

        # 檢查輸入的特徵圖象是否為4張
        assert len(in_channels) == 4
        # 檢查out_channels需要是int格式
        assert isinstance(out_channels, int)

        # 獲取通過每個模塊時channel深度，會是[32, 32, 64, 128, 256]
        blocks_out_channels = [out_channels] + [
            min(out_channels * 2**i, 256) for i in range(4)
        ]
        # 獲取每個模塊輸入的channel深度，會是[32, 320, 640, 1280, 2048]
        blocks_in_channels = [blocks_out_channels[1]] + [
            in_channels[i] + blocks_out_channels[i + 2] for i in range(3)
        ] + [in_channels[3]]

        # 構建轉至卷積實例對對象，經過該卷積後會上採樣2倍
        # Ex:[batch_size, in_channel, height, width] -> [batch_size, out_channel, height * 2, width * 2]
        self.up4 = nn.ConvTranspose2d(
            blocks_in_channels[4],
            blocks_out_channels[4],
            kernel_size=4,
            stride=2,
            padding=1)
        # 構建一系列UpBlock實例對象
        self.up_block3 = UpBlock(blocks_in_channels[3], blocks_out_channels[3])
        self.up_block2 = UpBlock(blocks_in_channels[2], blocks_out_channels[2])
        self.up_block1 = UpBlock(blocks_in_channels[1], blocks_out_channels[1])
        self.up_block0 = UpBlock(blocks_in_channels[0], blocks_out_channels[0])

    def forward(self, x):
        """
        Args:
            x (list[Tensor] | tuple[Tensor]): A list of four tensors of shape
                :math:`(N, C_i, H_i, W_i)`, representing C2, C3, C4, C5
                features respectively. :math:`C_i` should matches the number in
                ``in_channels``.

        Returns:
            Tensor: Shape :math:`(N, C, H, W)` where :math:`H=4H_0` and
            :math:`W=4W_0`.
        """
        # 已看過，主要是TextSnake的neck模塊，將backbone輸出的特徵圖進行通道調整以及融合
        # x = backbone輸出的特徵圖，tuple(tensor) tensor shape [batch_size, channel, height, width]
        # index越大的特徵圖大小越小channel越深同時抽象程度越高

        # 將x當中4個特徵圖提取出來
        c2, c3, c4, c5 = x

        # 將高維特徵圖通過轉至卷積進行2倍上採樣後在經過激活函數
        x = F.relu(self.up4(c5))

        # 將x與c4進行拼接
        x = torch.cat([x, c4], dim=1)
        # 通過up_block進行通道調整並且通過激活函數，up_block會對特徵圖進行上採樣
        x = F.relu(self.up_block3(x))

        # 下面就是一系列的上採樣拼接融合激活等等

        x = torch.cat([x, c3], dim=1)
        x = F.relu(self.up_block2(x))

        x = torch.cat([x, c2], dim=1)
        x = F.relu(self.up_block1(x))

        x = self.up_block0(x)
        # the output should be of the same height and width as backbone input
        # x shape = [batch_size, channel, height, width]，高寬最後會縮放到與傳入網路時相同的高寬
        return x
