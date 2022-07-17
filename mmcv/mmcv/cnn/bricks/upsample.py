# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import xavier_init
from .registry import UPSAMPLE_LAYERS

UPSAMPLE_LAYERS.register_module('nearest', module=nn.Upsample)
UPSAMPLE_LAYERS.register_module('bilinear', module=nn.Upsample)


@UPSAMPLE_LAYERS.register_module(name='pixel_shuffle')
class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer.

    This module packs `F.pixel_shuffle()` and a nn.Conv2d module together to
    achieve a simple upsampling with pixel shuffle.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of the conv layer to expand the
            channels.
    """

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int,
                 upsample_kernel: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        self.init_weights()

    def init_weights(self):
        xavier_init(self.upsample_conv, distribution='uniform')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


def build_upsample_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build upsample layer.

    Args:
        cfg (dict): The upsample layer config, which should contain:

            - type (str): Layer type.
            - scale_factor (int): Upsample ratio, which is not applicable to
              deconv.
            - layer args: Args needed to instantiate a upsample layer.
        args (argument list): Arguments passed to the ``__init__``
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the
            ``__init__`` method of the corresponding conv layer.

    Returns:
        nn.Module: Created upsample layer.
    """
    # 已看過

    # cfg = 上採樣設定檔
    # kwargs = 一些參數，因為會先透過上採樣後在與其他層進行融合，所以這裡需要將channel以及特徵圖大小設定到與之後融合的特徵圖相同

    if not isinstance(cfg, dict):
        # 檢查cfg必須為dict格式，如果不是這裡會報錯
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        # 如果cfg當中沒有type就會報錯
        raise KeyError(
            f'the cfg dict must contain the key "type", but got {cfg}')
    # 複製cfg到cfg_當中
    cfg_ = cfg.copy()

    # 取出要上採樣的方式
    layer_type = cfg_.pop('type')
    if layer_type not in UPSAMPLE_LAYERS:
        # 如果上採樣的方式不再UPSAMPLE_LAYERS當中就會在這裡報錯
        raise KeyError(f'Unrecognized upsample type {layer_type}')
    else:
        # 獲取upsample class
        upsample = UPSAMPLE_LAYERS.get(layer_type)

    if upsample is nn.Upsample:
        cfg_['mode'] = layer_type
    # layer = upsample實例對象，將參數傳入class當中進行初始化
    layer = upsample(*args, **kwargs, **cfg_)
    return layer
