# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .conv_module import ConvModule


class DepthwiseSeparableConvModule(nn.Module):
    """Depthwise separable convolution module.

    See https://arxiv.org/pdf/1704.04861.pdf for details.

    This module can replace a ConvModule with the conv block replaced by two
    conv block: depthwise conv block and pointwise conv block. The depthwise
    conv block contains depthwise-conv/norm/activation layers. The pointwise
    conv block contains pointwise-conv/norm/activation layers. It should be
    noted that there will be norm/activation layer in the depthwise conv block
    if `norm_cfg` and `act_cfg` are specified.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``. Default: 1.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``. Default: 0.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``. Default: 1.
        norm_cfg (dict): Default norm config for both depthwise ConvModule and
            pointwise ConvModule. Default: None.
        act_cfg (dict): Default activation config for both depthwise ConvModule
            and pointwise ConvModule. Default: dict(type='ReLU').
        dw_norm_cfg (dict): Norm config of depthwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        dw_act_cfg (dict): Activation config of depthwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        pw_norm_cfg (dict): Norm config of pointwise ConvModule. If it is
            'default', it will be the same as `norm_cfg`. Default: 'default'.
        pw_act_cfg (dict): Activation config of pointwise ConvModule. If it is
            'default', it will be the same as `act_cfg`. Default: 'default'.
        kwargs (optional): Other shared arguments for depthwise and pointwise
            ConvModule. See ConvModule for ref.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Dict = dict(type='ReLU'),
                 dw_norm_cfg: Union[Dict, str] = 'default',
                 dw_act_cfg: Union[Dict, str] = 'default',
                 pw_norm_cfg: Union[Dict, str] = 'default',
                 pw_act_cfg: Union[Dict, str] = 'default',
                 **kwargs):
        """ DW卷積初始化，也就是深度可分卷積的初始化函數
        Args:
            in_channels: 輸入的channel深度
            out_channels: 輸出的channel深度
            kernel_size: 卷積核大小
            stride: 步距
            padding: 填充大小
            dilation: 膨脹係數
            norm_cfg: 標準化層設定資料
            act_cfg: 激活函數設定
            dw_norm_cfg: dw當中標準化設定
            dw_act_cfg: dw當中激活函數設定
            pw_norm_cfg: pw當中標準化設定
            pw_act_cfg: pw當中激活函數設定
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        # 在dw卷積當中不可以指定在卷積時的groups設定，因為這裡就會直接分成in_channel組
        assert 'groups' not in kwargs, 'groups should not be specified'

        # if norm/activation config of depthwise/pointwise ConvModule is not
        # specified, use default config.
        # 如果沒有特殊設定標準化層或是激活函數層，就會直接使用傳入的norm_cfg與act_cfg
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg  # type: ignore # noqa E501
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg  # type: ignore # noqa E501
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg

        # depthwise convolution
        # 首先進行depth-wise卷積
        self.depthwise_conv = ConvModule(
            # 將傳入的channel深度
            in_channels,
            # 輸出的channel深度不會發生變化
            in_channels,
            # 卷積核大小
            kernel_size,
            stride=stride,
            padding=padding,
            # 膨脹係數
            dilation=dilation,
            # 這裡的組數就會是in_channel，這樣就是dw卷積
            groups=in_channels,
            norm_cfg=dw_norm_cfg,  # type: ignore
            act_cfg=dw_act_cfg,  # type: ignore
            **kwargs)

        # 構建point-wise卷積，需要將dw出來的結果通過point-wise卷積
        self.pointwise_conv = ConvModule(
            in_channels,
            # 輸出的channel深度會發生變化
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,  # type: ignore
            act_cfg=pw_act_cfg,  # type: ignore
            **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
