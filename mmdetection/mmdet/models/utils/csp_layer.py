# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule


class DarknetBottleneck(BaseModule):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The kernel size of the convolution. Default: 0.5
        add_identity (bool): Whether to add identity to the out.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        """ 使用在Darknet當中的瓶頸結構初始化函數
        Args:
            in_channels: 輸入的channel深度
            out_channels: 輸出的channel深度
            expansion: 影層的channel深度倍率
            add_identity: 是否需要添加捷徑
            use_depthwise: 是否使用dw卷積
            conv_cfg: 卷積層設定
            norm_cfg: 標準化層設定
            act_cfg: 激活函數層設定
            init_cfg: 初始化設定
        """
        # 繼承自BaseModule，將繼承對象進行初始化
        super().__init__(init_cfg)
        # 獲取影層的channel深度
        hidden_channels = int(out_channels * expansion)
        # 如果有指定使用dw卷積就會使用dw卷積模塊，否則就會使用普通卷積模塊
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        # 構建第一層卷積，這裡就只會使用普通卷積模塊
        self.conv1 = ConvModule(
            # 輸入的channel深度
            in_channels,
            # 影層channel深度
            hidden_channels,
            # 卷積核為1x1
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # 這裡就會根據指定的卷積方式使用不同卷積標準化激活模塊
        self.conv2 = conv(
            # 輸入channel就會是影層channel深度
            hidden_channels,
            # 輸出channel會是指定輸出channel深度
            out_channels,
            # 使用3x3卷積核
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # 保存是否使用identity，不過條件是輸入以及輸出的channel深度需要相同才可以使用
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        # CSPLayer當中的瓶頸結構的forward函數
        # x shape = [batch_size, channel, height, width]

        # 先保存一份x作為捷徑分支上需要使用到的
        identity = x
        # 通過第一層卷積，這裡卷積核大部分會是1x1的
        out = self.conv1(x)
        # 通過第二層卷積，這裡卷積核大部分會是3x3的
        out = self.conv2(out)

        if self.add_identity:
            # 如果有需要添加上捷徑分支就會到這裡
            return out + identity
        else:
            # 否則就會到這裡直接輸出
            return out


class CSPLayer(BaseModule):
    """Cross Stage Partial Layer.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Default: 0.5
        num_blocks (int): Number of blocks. Default: 1
        add_identity (bool): Whether to add identity in blocks.
            Default: True
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio=0.5,
                 num_blocks=1,
                 add_identity=True,
                 use_depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        """ 初始化CSP層結構，這裡會有產生主線以及支線
        Args:
            in_channels: 輸入的channel深度
            out_channels: 輸出的channel深度
            expand_ratio: 分成主線以及支線時的channel深度縮放倍率
            num_blocks: 主線當中層結構堆疊數量
            add_identity: 是否需要添加上捷徑
            use_depthwise: 是否需要使用dw卷積
            conv_cfg: 卷積設定
            norm_cfg: 標準化層設定
            act_cfg: 激活函數層設定
            init_cfg: 初始化方式
        """
        # 繼承自BaseModule，將繼承對象進行初始化
        super().__init__(init_cfg)
        # 獲取分叉後的channel深度
        mid_channels = int(out_channels * expand_ratio)
        # 主幹上的卷積層結構
        self.main_conv = ConvModule(
            # 輸入channel深度
            in_channels,
            # 這裡輸出的channel深度也會是縮放後的深度
            mid_channels,
            # 使用1x1的卷積核
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # 支線的卷積層結構
        self.short_conv = ConvModule(
            # 輸入的channel深度
            in_channels,
            # 輸出的channel深度也會是縮放後的深度
            mid_channels,
            # 使用1x1卷積核
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # 將主線以及支線拼接後的卷積結構
        self.final_conv = ConvModule(
            # channel深度會是兩倍的縮放channel深度，因為是使用拼接方式進行融合
            2 * mid_channels,
            # 輸出channel就會是指定的channel深度
            out_channels,
            # 使用1x1卷積核
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # 這裡會是在主線上進行堆截的層結構，使用nn.Sequential進行包裝，總共會產生num_blocks層
        self.blocks = nn.Sequential(*[
            # 使用的是Darknet的Bottleneck結構
            DarknetBottleneck(
                # 主幹當中的channel深度都會是縮放後的channel深度
                mid_channels,
                mid_channels,
                # 卷積核大小
                1.0,
                # 傳入是否使用捷徑
                add_identity,
                # 是否使用dw卷積
                use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)
        ])

    def forward(self, x):
        # 進行CSPLayer的forward函數
        # x shape = [batch_size, channel, height, width]

        # 這裡會先獲取支線上的卷積結果，x_short shape [batch_size, channel / 2, height, width]
        x_short = self.short_conv(x)

        # 進行主幹上的卷積，x_main shape [batch_size, channel / 2, height, width]
        x_main = self.main_conv(x)
        # 通過主幹上多層堆疊的block
        x_main = self.blocks(x_main)

        # 最後將主幹與支線的特徵圖進行拼接
        # x_final shape = [batch_size, channel, height, width]
        x_final = torch.cat((x_main, x_short), dim=1)
        # 最後結果再通過一層卷積模塊
        return self.final_conv(x_final)
