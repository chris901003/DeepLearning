# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import CSPLayer


class Focus(nn.Module):
    """Focus width and height information into channel space.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the convolution. Default: 1
        stride (int): The stride of the convolution. Default: 1
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish')):
        """ 注意力模塊
        Args:
            in_channels: 輸入的channel深度
            out_channels: 輸出的channel深度
            kernel_size: 卷積核大小
            stride: 步距
            conv_cfg: 卷積層設定
            norm_cfg: 標準化層設定
            act_cfg: 激活函數層設定
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        # 構建卷積標準化激活函數模塊
        self.conv = ConvModule(
            # 這裡進行正向傳遞時輸入的channel會放大成四倍
            in_channels * 4,
            # 輸出channel不變
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        # Focus的forward函數部分，這裡會先對圖像進行注意力機制
        # shape of x [batch_size, channel, height, width]
        # 以下的每個都是shape [batch_size, channel, height / 2, width / 2]
        # 獲取左上角部分資訊
        patch_top_left = x[..., ::2, ::2]
        # 獲取右上角資訊
        patch_top_right = x[..., ::2, 1::2]
        # 獲取左下角資訊
        patch_bot_left = x[..., 1::2, ::2]
        # 獲取右下角資訊
        patch_bot_right = x[..., 1::2, 1::2]
        # 使用concat在第二個維度上進行拼接，x shape [batch_size, channel * 4, height / 2, width / 2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        # 通過卷積進行，也就是通過focus層後圖像高寬會減半
        return self.conv(x)


class SPPBottleneck(BaseModule):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        """ 構建SPP層結構，初始化函數
        Args:
            in_channels: 輸入channel深度
            out_channels: 輸出channel深度
            kernel_sizes: 使用的卷積核大小
            conv_cfg: 卷積核設定
            norm_cfg: 標準化層設定
            act_cfg: 激活函數層設定
            init_cfg: 初始化設定
        """
        # 繼承自BaseModule，對繼承對象進行初始化
        super().__init__(init_cfg)
        # 獲取中間層channel深度
        mid_channels = in_channels // 2
        # 進入時使用的卷積層
        self.conv1 = ConvModule(
            # 將channel調整到影層的channel深度
            in_channels,
            mid_channels,
            # 使用1x1卷積核大小
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # 中間會有3種不同池化下採樣方式
        self.poolings = nn.ModuleList([
            # 這裡會遍歷不同池化核大小
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        # 獲取最終卷積時的channel深度，這裡會將池化後的結果用concat進行融合
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        # 最後輸出時通過的卷積核
        self.conv2 = ConvModule(
            # 從池化後的結果concat後的channel深度
            conv2_channels,
            # 將channel深度調整到指定輸出channel深度
            out_channels,
            # 使用1x1卷積核
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


@BACKBONES.register_module()
class CSPDarknet(BaseModule):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Default: P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Default: 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False.
        arch_ovewrite(list): Overwrite default arch settings. Default: None.
        spp_kernal_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Default: (5, 9, 13).
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    Example:
        >>> from mmdet.models import CSPDarknet
        >>> import torch
        >>> self = CSPDarknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings = {
        'P5': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 1024, 3, False, True]],
        'P6': [[64, 128, 3, True, False], [128, 256, 9, True, False],
               [256, 512, 9, True, False], [512, 768, 3, True, False],
               [768, 1024, 3, False, True]]
    }

    def __init__(self,
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 arch_ovewrite=None,
                 spp_kernal_sizes=(5, 9, 13),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        """ CSPDarknet特徵提取模型，專門給yoloV5以及yolox使用
        Args:
            arch: 選擇CSPDarknet的架構，這裡可以選擇P5或是P6
            deepen_factor: 模型當中堆疊模塊的深度，這裡會是基礎深度乘上的倍率，也就是可以透過此參數調整模型大小
            widen_factor: 模型當中channel的深度，這裡會是基礎channel深度乘上的倍率，也是用來控制模型大小的參數
            out_indices: 在backbone當中的哪些模塊結果需要輸出出去
            frozen_stages: 選擇哪些層需要進行凍結，如果設定成-1就全部都會進行訓練
            use_depthwise: 是否使用dw卷積，如果使用dw卷積可以降低參數量，但是會多少引響到正確率
            arch_ovewrite: 如果有需要更改預設的模型架構就會寫在這裡
            spp_kernal_sizes: spp層結構的卷積核大小
            conv_cfg: 卷積層設定
            act_cfg: 激活函數設定
            norm_cfg: 標準化層設定
            norm_eval: 是否需要凍結標準化的均值以及方差
            init_cfg: 初始化設定方式
        """
        # 繼承自BaseModule，將繼承對象進行初始化
        super().__init__(init_cfg)
        # 根據指定的模型架構獲取對應的配置
        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            # 如果要使用自己的架構就會到這裡進行覆蓋
            arch_setting = arch_ovewrite
        # 輸出的層數需要符合
        assert set(out_indices).issubset(i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            # 檢查指定的凍結層是否合法
            raise ValueError('frozen_stages must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        # 將傳入參數進行保存
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        # 如果有指定需要使用dw卷積就會指定成dw卷積模塊，否則就會使用普通的卷積模塊
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # 這裡會構建注意力模塊
        self.stem = Focus(
            # 輸入channel深度
            3,
            # 輸出的深度會是下個模塊輸入channel的深度
            int(arch_setting[0][0] * widen_factor),
            # 使用3x3的卷積核
            kernel_size=3,
            # 使用指定的卷積核
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        # 構建層結構名稱，方便在正向傳遞時調用
        self.layers = ['stem']

        # 構建剩下的層結構
        for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            # 獲取輸入的channel深度，這裡的深度會是基礎深度乘上縮放倍率
            in_channels = int(in_channels * widen_factor)
            # 獲取輸出的channel深度，這裡的深度會是基礎深度乘上縮放倍率
            out_channels = int(out_channels * widen_factor)
            # 獲取會堆疊多少模塊，這裡也會有堆疊數量的倍率，最少會堆疊一塊
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            # 保存一個dark_block當中的每個模塊
            stage = []
            # 使用指定的卷積模塊，這裡會有普通卷積或是dw卷積兩種
            conv_layer = conv(
                # 輸入的channel深度
                in_channels,
                # 輸出的channel深度
                out_channels,
                # 卷積核大小
                3,
                # 步距
                stride=2,
                # 填充大小
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            # 將實例化後的結果保存到stage當中
            stage.append(conv_layer)
            if use_spp:
                # 如果有需要使用spp就會到這裡，構建SPPBottleneck實例化對象
                spp = SPPBottleneck(
                    # 輸入channel深度
                    out_channels,
                    # 輸出channel深度
                    out_channels,
                    # 傳入在spp當中會使用到的卷積核大小
                    kernel_sizes=spp_kernal_sizes,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                # 將spp實例化對象保存到stage當中
                stage.append(spp)
            # 構建CSP層結構
            csp_layer = CSPLayer(
                # 輸入channel深度與輸出channel深度相同
                out_channels,
                out_channels,
                # 傳入總共需要重複多少次
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            # 添加到stage當中
            stage.append(csp_layer)
            # 使用add_module方式添加到模型當中
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            # 這裡需要保存模型名稱，之後才可以在正向傳遞時進行呼叫
            self.layers.append(f'stage{i + 1}')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(CSPDarknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        # CSPDarknet的forward函數，對輸入的圖像提取特徵
        # 保存指定層結構輸出結果
        outs = []
        # 遍歷每層層結構
        for i, layer_name in enumerate(self.layers):
            # 使用名稱獲取對應的層結構實例對象
            layer = getattr(self, layer_name)
            # 進行正向傳遞
            x = layer(x)
            if i in self.out_indices:
                # 如果該層是需要進行回傳的就保存到outs當中
                outs.append(x)
        # 回傳結果
        return tuple(outs)
