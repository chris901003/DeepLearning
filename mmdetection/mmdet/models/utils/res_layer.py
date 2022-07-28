# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, Sequential
from torch import nn as nn


class ResLayer(Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 **kwargs):
        """ 已看過，構建resnet多層中的每一層結構，裏面包含堆疊多層卷積層
        Args:
            block: 使用的class，在比較深層的resnet時我們會用Bottleneck
            inplanes: 輸入的channel深度
            planes: 輸出的channel深度
            num_blocks: 總共需要堆疊幾層block
            stride: 第一層卷積的步距
            avg_down: 是否要用AvgPooling代替卷積的下採樣
            conv_cfg: 卷積層的設定
            norm_cfg: 標準化層的設定
            downsample_first: 是否在第一個卷積層就進行下採樣，預設為True
            kwargs: 其他資訊，如果需要了解可以用Debug模式下去看
        """

        # 這裡是繼承於Sequential
        # 將block保存下來
        self.block = block

        # 預設先將下採樣設定成None
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            # 如果步距不是1或是輸入的channel與輸出channel不相同
            # 就會需要透過downsample對殘差結構進行調整
            downsample = []
            # 當前的卷積步距
            conv_stride = stride
            if avg_down:
                # 如果有設定透過AvgPooling進行下採樣
                # 因為會透過AvgPooling下採樣，所以將步距調整到1
                conv_stride = 1
                downsample.append(
                    # 添加上AvgPooling層
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                # 添加卷積層
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                # 構建標準化層，這裡我們只需要實例對象就可以
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            # 將downsample包裝成Sequential
            downsample = nn.Sequential(*downsample)

        # 主幹部分
        layers = []
        if downsample_first:
            # 如果是要先進行下採樣就會是這裡
            layers.append(
                # 添加block層，這裡的步距就是指定的步距
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
            # 更新下輪的輸入channel深度
            inplanes = planes * block.expansion
            # 構建剩下的層結構
            for _ in range(1, num_blocks):
                layers.append(
                    # 這裡也是構建block模塊，不過不具都會是1，因為輸入以及輸出的channel同時高寬不會改變，所以就沒有傳入downsample
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))

        else:  # downsample_first=False is for HourglassModule
            # 與上面不同的地方只有會進行下採樣的層結構是在最後
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        **kwargs))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))
        # 這裡繼承於Sequential，所以我們將構建好的layers放到繼承對象當中進行初始化
        super(ResLayer, self).__init__(*layers)


class SimplifiedBasicBlock(BaseModule):
    """Simplified version of original basic residual block. This is used in
    `SCNet <https://arxiv.org/abs/2012.10150>`_.

    - Norm layer is now optional
    - Last ReLU in forward function is removed
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_fg=None):
        super(SimplifiedBasicBlock, self).__init__(init_fg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        assert not with_cp, 'Not implemented yet.'
        self.with_norm = norm_cfg is not None
        with_bias = True if norm_cfg is None else False
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=with_bias)
        if self.with_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, planes, postfix=1)
            self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=with_bias)
        if self.with_norm:
            self.norm2_name, norm2 = build_norm_layer(
                norm_cfg, planes, postfix=2)
            self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name) if self.with_norm else None

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name) if self.with_norm else None

    def forward(self, x):
        """Forward function."""

        identity = x

        out = self.conv1(x)
        if self.with_norm:
            out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_norm:
            out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out
