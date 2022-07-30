# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, ModuleList, auto_fp16

from mmocr.models.builder import NECKS


@NECKS.register_module()
class FPNF(BaseModule):
    """FPN-like fusion module in Shape Robust Text Detection with Progressive
    Scale Expansion Network.

    Args:
        in_channels (list[int]): A list of number of input channels.
        out_channels (int): The number of output channels.
        fusion_type (str): Type of the final feature fusion layer. Available
            options are "concat" and "add".
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 fusion_type='concat',
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        """ 已看過，(PSENet的neck部分進行標注)
        Args:
            in_channels: 從backbone輸出特徵圖的channel深度，這裡用的是resnet作為backbone且每層都進行輸出
            out_channels: 輸出的channel深度
            fusion_type: 合併方式，這裡使用的是concat
            init_cfg: 初始化方式
        """

        # 繼承於BaseModule，對繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)
        # conv_cfg先設定成None，這樣預設就會直接使用Conv2d
        conv_cfg = None
        # 標準化設定
        norm_cfg = dict(type='BN')
        # 激活函數設定
        act_cfg = dict(type='ReLU')

        # 保存傳入的變數
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 一些保存層結構的地方
        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        # 獲取從backbone輸出的特徵圖數量
        self.backbone_end_level = len(in_channels)
        # 遍歷傳入的特徵圖數量
        for i in range(self.backbone_end_level):
            # 構建卷積加上標準化加上激活函數層
            # 將特徵圖的channel深度調整到輸出的channel深度
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            # 將實例化對象進行保存
            self.lateral_convs.append(l_conv)

            if i < self.backbone_end_level - 1:
                # 除了最一一層特徵圖其他都會進來
                fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                # 存放到fpn_convs當中
                self.fpn_convs.append(fpn_conv)

        # 獲取多特徵圖融合方式
        self.fusion_type = fusion_type

        if self.fusion_type == 'concat':
            # 如果是透過拼接那麼channel就會是256*4
            feature_channels = 1024
        elif self.fusion_type == 'add':
            # 如果是透過相加channel就會是256
            feature_channels = 256
        else:
            # 其他狀況就沒有實作對象，會直接報錯
            raise NotImplementedError

        # 將融合後的結果再通過一次卷積結構，將多個特徵層進行特徵融合同時調整channel深度
        self.output_convs = ConvModule(
            feature_channels,
            out_channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

    @auto_fp16()
    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): Each tensor has the shape of
                :math:`(N, C_i, H_i, W_i)`. It usually expects 4 tensors
                (C2-C5 features) from ResNet.

        Returns:
            Tensor: A tensor of shape :math:`(N, C_{out}, H_0, W_0)` where
            :math:`C_{out}` is ``out_channels``.
        """
        # 已看過，FPNF的forward函數
        # inputs = 從backbone的輸出，tuple[tensor]，tensor shape [batch_size, channel, height, width]

        # 檢查輸入的inputs數量要與self.in_channels當中相同
        assert len(inputs) == len(self.in_channels)

        # build laterals，透過lateral_convs將channel深度調整到out_channel，所有特徵圖的channel深度就會變成相同
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path，獲取總共有多少張特徵圖
        used_backbone_levels = len(laterals)
        # for range (start, stop, step)
        for i in range(used_backbone_levels - 1, 0, -1):
            # step 1: upsample to level i-1 size and add level i-1
            # 獲取較低層次特徵圖的高寬
            prev_shape = laterals[i - 1].shape[2:]
            # 先將當前特徵圖透過差值算法將高寬擴大，變成與低層次特徵圖相同高寬後進行相加融合
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
            # step 2: smooth level i-1
            # 使用conv將融合後的特徵圖
            laterals[i - 1] = self.fpn_convs[i - 1](laterals[i - 1])

        # upsample and cont，最後laterals[0]就會是融合了所有特徵圖的內容
        # 獲取最低層次特徵圖的高寬
        bottom_shape = laterals[0].shape[2:]
        for i in range(1, used_backbone_levels):
            # 使用差值方式將所有較高層次的特徵圖擴大到與最低層次相同
            laterals[i] = F.interpolate(
                laterals[i], size=bottom_shape, mode='nearest')

        if self.fusion_type == 'concat':
            # 如果選擇的方式是拼接就會在第一維度上面將所有特徵圖拼接
            out = torch.cat(laterals, 1)
        elif self.fusion_type == 'add':
            # 如果是相加就直接加在一起
            out = laterals[0]
            for i in range(1, used_backbone_levels):
                out += laterals[i]
        else:
            # 其他就直接報錯
            raise NotImplementedError
        # 最後通過conv同時融合特徵圖以及調整通道，因為如果是用concat會將channel深度擴大
        out = self.output_convs(out)

        # 最後回傳結果，tensor shape [batch_size, channel=out_channel, height, width]
        return out
