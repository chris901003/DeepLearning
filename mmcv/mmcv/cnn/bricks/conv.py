# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

from torch import nn

from .registry import CONV_LAYERS

CONV_LAYERS.register_module('Conv1d', module=nn.Conv1d)
CONV_LAYERS.register_module('Conv2d', module=nn.Conv2d)
CONV_LAYERS.register_module('Conv3d', module=nn.Conv3d)
CONV_LAYERS.register_module('Conv', module=nn.Conv2d)


def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    # 已看過
    # cfg = 指定要用什麼卷積層(dict格式)
    # args = (in_channels, out_channels, kernel_size)
    # kwargs = (stride, padding, dilation, groups, bias) [在這裡的是dict格式，因為在初始化Conv時這些的順序會亂放，所以需要用dict]

    # cfg = 卷積的config，也就是要用的卷積(Conv2d, Conv3d, ...)
    if cfg is None:
        # 如果沒有傳入cfg，這裡我們就會默認使用Conv2d來進行卷積
        cfg_ = dict(type='Conv2d')
    else:
        # 如果cfg不是None，會在這裡檢查cfg是否為dict格式
        if not isinstance(cfg, dict):
            # 不是dict格式就會在這裡報錯
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            # 如果cfg當中沒有type這裡就會報錯，因為沒有type就不知道要用哪種卷積層
            raise KeyError('the cfg dict must contain the key "type"')
        # 將cfg拷貝到cfg_
        cfg_ = cfg.copy()

    # 將cfg_當中的type拿出來
    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        # 如果該卷積層沒有在CONV_LAYERS裏面的話這裡就會報錯，表示沒有支援
        raise KeyError(f'Unrecognized layer type {layer_type}')
    else:
        # 獲取卷積層class，這裡拿到的就會是torch.nn.Conv的class
        conv_layer = CONV_LAYERS.get(layer_type)

    # 將設定值傳入到conv_layer進行實例化
    layer = conv_layer(*args, **kwargs, **cfg_)

    # 將最後實例化完的卷積層回傳(假設cfg指定是Conv2d，這裡回傳就會是torch.nn.Conv2d的實例化對象)
    return layer
