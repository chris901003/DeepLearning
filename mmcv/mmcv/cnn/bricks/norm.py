# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from typing import Dict, Tuple, Union

import torch.nn as nn

from mmcv.utils import is_tuple_of
from mmcv.utils.parrots_wrapper import SyncBatchNorm, _BatchNorm, _InstanceNorm
from .registry import NORM_LAYERS

NORM_LAYERS.register_module('BN', module=nn.BatchNorm2d)
NORM_LAYERS.register_module('BN1d', module=nn.BatchNorm1d)
NORM_LAYERS.register_module('BN2d', module=nn.BatchNorm2d)
NORM_LAYERS.register_module('BN3d', module=nn.BatchNorm3d)
NORM_LAYERS.register_module('SyncBN', module=SyncBatchNorm)
NORM_LAYERS.register_module('GN', module=nn.GroupNorm)
NORM_LAYERS.register_module('LN', module=nn.LayerNorm)
NORM_LAYERS.register_module('IN', module=nn.InstanceNorm2d)
NORM_LAYERS.register_module('IN1d', module=nn.InstanceNorm1d)
NORM_LAYERS.register_module('IN2d', module=nn.InstanceNorm2d)
NORM_LAYERS.register_module('IN3d', module=nn.InstanceNorm3d)


def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    # 已看過
    # 依據上面第一條內容這個函數主要是用來，從類名推斷縮寫

    if not inspect.isclass(class_type):
        # 如果傳入的不是class類型在這裡就會報錯
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        # 如果已經有命名就依據裏面的命名回傳
        return class_type._abbr_
    # 下面就是根據裏面用的標準化方式給定名稱
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'


def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    # 已看過

    # cfg = 標準化層的設定檔，裏面有指定的標準化層以及是否可以學習
    # num_features = 進入標準化層的channel深度
    # postfix = 在命名當中後面加上特殊標示，沒有特別重要

    if not isinstance(cfg, dict):
        # 檢查cfg是否為dict格式，如果不是就會報錯
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        # 檢查cfg當中有沒有type，沒有就會報錯
        raise KeyError('the cfg dict must contain the key "type"')
    # 將cfg拷貝到cfg_當中
    cfg_ = cfg.copy()

    # 取出要使用的標準化層
    layer_type = cfg_.pop('type')
    if layer_type not in NORM_LAYERS:
        # 如果layer_type沒有在NORM_LAYERS當中這裡就會報錯
        raise KeyError(f'Unrecognized norm type {layer_type}')

    # 獲取標準化層的類(假設我們需要的是BN，返回的就會是torch.nn.BatchNorm2d的class)，**這裡都不會是實例化對象
    norm_layer = NORM_LAYERS.get(layer_type)
    # 將norm_layer放入判斷該class要給的名稱(假設傳入的是torch.nn.BatchNorm2d的class，abbr=BN)
    abbr = infer_abbr(norm_layer)

    # 用來在名稱後面加上一些東西，沒有特別重要
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    # 檢查是否要將可學習打開，如果沒有特別設定默認就會是開啟學習
    requires_grad = cfg_.pop('requires_grad', True)
    # 在cfg_當中加上eps的設定值，在進行標準化操作時的分佈修正項，預設就會是1e-5
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        # 如果不是GN就會往這裡
        # layer = 標準化層實例化對象，**cfg_裏面應該就只會有一些設定的參數
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            # 如果使用的是SyncBn這裡會需要進行微調
            layer._specify_ddp_gpu_num(1)
    else:
        # GN會往這裡
        assert 'num_groups' in cfg_
        # 因為在pytorch當中num_channels不是在第一個，所以這裡要特別設定
        layer = norm_layer(num_channels=num_features, **cfg_)

    # 設定是否啟用學習
    for param in layer.parameters():
        param.requires_grad = requires_grad

    # 回傳標準化層名稱以及標準化層實例對象
    return name, layer


def is_norm(layer: nn.Module,
            exclude: Union[type, tuple, None] = None) -> bool:
    """Check if a layer is a normalization layer.

    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type | tuple[type]): Types to be excluded.

    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not is_tuple_of(exclude, type):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)
