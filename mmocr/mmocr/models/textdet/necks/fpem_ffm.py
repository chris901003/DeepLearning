# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from torch import nn

from mmocr.models.builder import NECKS


class FPEM(BaseModule):
    """FPN-like feature fusion module in PANet.

    Args:
        in_channels (int): Number of input channels.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self, in_channels=128, init_cfg=None):
        """ 已看過，像是FPN一樣的東西，可以對特徵圖進行融合
        Args:
            in_channels: 輸入的特徵圖channel深度
            init_cfg: 初始化方式
        """
        super().__init__(init_cfg=init_cfg)
        # 構建SeparableConv2d實例對象，最後一個參數是步距
        self.up_add1 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add2 = SeparableConv2d(in_channels, in_channels, 1)
        self.up_add3 = SeparableConv2d(in_channels, in_channels, 1)
        self.down_add1 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add2 = SeparableConv2d(in_channels, in_channels, 2)
        self.down_add3 = SeparableConv2d(in_channels, in_channels, 2)

    def forward(self, c2, c3, c4, c5):
        """
        Args:
            c2, c3, c4, c5 (Tensor): Each has the shape of
                :math:`(N, C_i, H_i, W_i)`.

        Returns:
            list[Tensor]: A list of 4 tensors of the same shape as input.
        """
        # 已看過，FPEM的forward函數
        # c2, c3, c4, c5 = 特徵圖，tensor shape [batch_size, channel, height, width]，c5會是最抽象的特徵提取
        # upsample，進行上採樣，_upsample_add函數在正下方
        c4 = self.up_add1(self._upsample_add(c5, c4))  # c4 shape
        c3 = self.up_add2(self._upsample_add(c4, c3))
        c2 = self.up_add3(self._upsample_add(c3, c2))

        # downsample，再進行下採樣，會先將高維特徵圖進行上採樣後進行融合最後透過步距為2的卷積下採樣
        c3 = self.down_add1(self._upsample_add(c3, c2))
        c4 = self.down_add2(self._upsample_add(c4, c3))
        c5 = self.down_add3(self._upsample_add(c5, c4))  # c4 / 2
        return c2, c3, c4, c5

    def _upsample_add(self, x, y):
        # 已看過，進行上採樣後再相加
        # 將x的特徵圖上採樣到與y相同大小後再進行相加
        return F.interpolate(x, size=y.size()[2:]) + y


class SeparableConv2d(BaseModule):

    def __init__(self, in_channels, out_channels, stride=1, init_cfg=None):
        """ 已看過，初始化可分離卷積
        Args:
            in_channels: 輸入的channel深度
            out_channels: 輸出的channel深度
            stride: 步距
            init_cfg: 初始化方式
        """
        # 繼承自BaseModule
        super().__init__(init_cfg=init_cfg)

        # 這裡會使用dw卷積，所以會傳入groups參數
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            groups=in_channels)
        # 普通卷積
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        # 標準化層結構以及激活函數
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@NECKS.register_module()
class FPEM_FFM(BaseModule):
    """This code is from https://github.com/WenmuZhou/PAN.pytorch.

    Args:
        in_channels (list[int]): A list of 4 numbers of input channels.
        conv_out (int): Number of output channels.
        fpem_repeat (int): Number of FPEM layers before FFM operations.
        align_corners (bool): The interpolation behaviour in FFM operation,
            used in :func:`torch.nn.functional.interpolate`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 conv_out=128,
                 fpem_repeat=2,
                 align_corners=False,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        """ 已看過，PAN的neck模塊
        Args:
            in_channels: 輸入的channel深度，會是list表示會從backbone輸出多少張特徵圖
            conv_out: 輸出的channel深度
            fpem_repeat: FPEM的層數在進行FFM之前的層結構
            align_corners: 差值相關資訊
            init_cfg: 初始化方式
        """
        # 繼承自BaseModule，將繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)
        # reduce layers，將每個特徵圖的channel深度調整到與輸出channel深度相同
        # 構建卷積層用來調整channel深度，用來調整第一張特徵圖
        self.reduce_conv_c2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[0],
                out_channels=conv_out,
                kernel_size=1), nn.BatchNorm2d(conv_out), nn.ReLU())
        # 構建卷積層用來調整channel深度，用來調整第二張特徵圖
        self.reduce_conv_c3 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[1],
                out_channels=conv_out,
                kernel_size=1), nn.BatchNorm2d(conv_out), nn.ReLU())
        # 構建卷積層用來調整channel深度，用來調整第三張特徵圖
        self.reduce_conv_c4 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[2],
                out_channels=conv_out,
                kernel_size=1), nn.BatchNorm2d(conv_out), nn.ReLU())
        # 構建卷積層用來調整channel深度，用來調整第四張特徵圖
        self.reduce_conv_c5 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels[3],
                out_channels=conv_out,
                kernel_size=1), nn.BatchNorm2d(conv_out), nn.ReLU())
        # 保存align_corners資料
        self.align_corners = align_corners
        # 構建fpems模塊
        self.fpems = ModuleList()
        # 遍歷次數
        for _ in range(fpem_repeat):
            # 構建FPEM實例對象
            self.fpems.append(FPEM(conv_out))

    def forward(self, x):
        """
        Args:
            x (list[Tensor]): A list of four tensors of shape
                :math:`(N, C_i, H_i, W_i)`, representing C2, C3, C4, C5
                features respectively. :math:`C_i` should matches the number in
                ``in_channels``.

        Returns:
            list[Tensor]: Four tensors of shape
            :math:`(N, C_{out}, H_0, W_0)` where :math:`C_{out}` is
            ``conv_out``.
        """
        # 已看過，PAN的neck部分
        # x = 從backbone的輸出，會是tuple(tensor)格式，tensor shape [batch_size, channel, height, width]
        # 將x的東西進行提取
        c2, c3, c4, c5 = x
        # reduce channel
        # 透過reduce_conv將channel深度調整到指定的輸出channel深度
        c2 = self.reduce_conv_c2(c2)
        c3 = self.reduce_conv_c3(c3)
        c4 = self.reduce_conv_c4(c4)
        c5 = self.reduce_conv_c5(c5)

        # FPEM，通過FPEM層結構，主要是可以將特徵進行融合
        for i, fpem in enumerate(self.fpems):
            # 將特徵圖放入進行正向傳播，輸出的shape不會產生變化，不過已經有經過特徵融合
            c2, c3, c4, c5 = fpem(c2, c3, c4, c5)
            if i == 0:
                # 如果是第一次的fpem就會到這裡，將結果存放
                c2_ffm = c2
                c3_ffm = c3
                c4_ffm = c4
                c5_ffm = c5
            else:
                # 第二次就會到這裡，將第二次結果與第一次結果進行相加
                c2_ffm = c2_ffm + c2
                c3_ffm = c3_ffm + c3
                c4_ffm = c4_ffm + c4
                c5_ffm = c5_ffm + c5

        # FFM
        # 透過雙線性差值將特徵圖大小統一調整到最大特徵圖的高寬
        c5 = F.interpolate(
            c5_ffm,
            c2_ffm.size()[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        c4 = F.interpolate(
            c4_ffm,
            c2_ffm.size()[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        c3 = F.interpolate(
            c3_ffm,
            c2_ffm.size()[-2:],
            mode='bilinear',
            align_corners=self.align_corners)
        # 最後將所有特徵圖用list包裝起來
        outs = [c2_ffm, c3, c4, c5]
        # 最後外面list換成tuple後回傳
        return tuple(outs)
