# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmcv.utils import _BatchNorm, _InstanceNorm
from ..utils import constant_init, kaiming_init
from .activation import build_activation_layer
from .conv import build_conv_layer
from .norm import build_norm_layer
from .padding import build_padding_layer
from .registry import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: Union[bool, str] = 'auto',
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 inplace: bool = True,
                 with_spectral_norm: bool = False,
                 padding_mode: str = 'zeros',
                 order: tuple = ('conv', 'norm', 'act')):
        """
        :param in_channels: 輸入的channel
        :param out_channels: 輸出的channel
        :param kernel_size: 卷積核大小
        :param stride: 步距
        :param padding: 填充
        :param dilation: 膨脹係數
        :param groups: 組卷積
        :param bias: 偏置
        :param conv_cfg: 卷積的config
        :param norm_cfg: 標準化層的config
        :param act_cfg: 激活函數的config
        :param inplace: 是否直接改值
        :param with_spectral_norm:
        :param padding_mode: 填充的值
        :param order: 卷積、標準化、激活函數的順序
        """
        # 已看過

        # 繼承自torch.nn.Module
        super().__init__()
        # 檢查一些型態是否正確
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        # 下面兩種是官方給的padding模式
        official_padding_mode = ['zeros', 'circular']
        # 將一些值保存下來
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        # with_explicit_padding = padding模式是否為官方給的方式
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        # order一定是由3個東西組成的
        assert isinstance(self.order, tuple) and len(self.order) == 3
        # 且order內只會出現[conv, norm, act]三種東西
        assert set(order) == {'conv', 'norm', 'act'}

        # 只有在有傳入norm_cfg才會有標準化層
        self.with_norm = norm_cfg is not None
        # 只有在有傳入act_cfg才會有激活層
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        # 如果卷積層在標準化層前面，那麼卷積層當中的偏置就可以去除
        if bias == 'auto':
            # 如果將bias設定成auto就會在這裡檢查是否要啟用bias
            bias = not self.with_norm
        # 紀錄是否啟用bias
        self.with_bias = bias

        if self.with_explicit_padding:
            # 這裡MMCV有多實現了一種padding方法(reflect)會在這裡透過build_padding_layer實現
            # 這裡我們先暫時不看，大多數情況padding都只會用0填充
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)

        # reset padding to 0 for conv module
        # 如果有使用特殊的padding方法，我們就會把conv_padding設定成0，否則就是傳入的padding值
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer，構建卷積層
        # self.conv = 卷積實例化對象(假設我們需要的是Conv2d，這裡就會是torch.nn.Conv2d的實例對象)
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        # 紀錄一些卷積參數，對於之後的操作可以有更多資料可以獲取
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            # 這部分主要在生成對抗網路中才會用到，詳細可以上網查，通常這裡我們默認會是False
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers，構建標準化層
        if self.with_norm:
            # norm layer is after conv layer
            # 根據標準化與卷積的順序，標準化要輸入的channel會有不同
            if order.index('norm') > order.index('conv'):
                # 先卷積的話channel就會是out_channels
                norm_channels = out_channels
            else:
                # 先標準化的話channel就會是in_channels
                norm_channels = in_channels
            # 透過build_norm_layer獲取標準化層
            # self.norm_name = 標準化層名稱
            # norm = 標準化層實例對象
            self.norm_name, norm = build_norm_layer(
                norm_cfg, norm_channels)  # type: ignore
            # 其實這步操作跟self.norm_name = norm基本上是等價的意思，只是中途可以更換self.norm_name對應上的實例化對象
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                # 檢查一些東西，正常不會有問題
                if isinstance(norm, (_BatchNorm, _InstanceNorm)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')
        else:
            # 如果沒有用標準化層就直接設定成None就可以了
            self.norm_name = None  # type: ignore

        # build activation layer，構建激活函數層
        if self.with_activation:
            # 先拷貝一份設定檔到act_cfg_
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            # 如果設定的激活函數不在以下幾個就將增加inplace設定值，一下幾個沒有inplace功能
            if act_cfg_['type'] not in [
                    'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            # 透過build_activation_layer構建激活函數層
            # self.activate = 激活函數實例對象
            self.activate = build_activation_layer(act_cfg_)

        # Use msra init by default，調用初始化函數，進行初始化
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.

        # 已看過
        # 進行初始化權重的函數
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self,
                x: torch.Tensor,
                activate: bool = True,
                norm: bool = True) -> torch.Tensor:
        # 已看過
        # 這裡同時會做卷積以及標準化以及激活三個步驟
        # 透過遍歷order決定哪些步驟先進行
        for layer in self.order:
            if layer == 'conv':
                # 當前要做卷積
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                # 當前要進行標準化
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                # 當前要進行激活
                x = self.activate(x)
        return x
