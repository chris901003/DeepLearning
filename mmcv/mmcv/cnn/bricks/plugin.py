# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import platform
from typing import Dict, Tuple, Union

import torch.nn as nn

from .registry import PLUGIN_LAYERS

if platform.system() == 'Windows':
    import regex as re  # type: ignore
else:
    import re  # type: ignore


def infer_abbr(class_type: type) -> str:
    """Infer abbreviation from the class name.

    This method will infer the abbreviation to map class types to
    abbreviations.

    Rule 1: If the class has the property "abbr", return the property.
    Rule 2: Otherwise, the abbreviation falls back to snake case of class
    name, e.g. the abbreviation of ``FancyBlock`` will be ``fancy_block``.

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    # 已看過，從類名推斷縮寫

    def camel2snack(word):
        """Convert camel case word into snack case.

        Modified from `inflection lib
        <https://inflection.readthedocs.io/en/latest/#inflection.underscore>`_.

        Example::

            >>> camel2snack("FancyBlock")
            'fancy_block'
        """

        word = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', word)
        word = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', word)
        word = word.replace('-', '_')
        return word.lower()

    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_  # type: ignore
    else:
        return camel2snack(class_type.__name__)


def build_plugin_layer(cfg: Dict,
                       postfix: Union[int, str] = '',
                       **kwargs) -> Tuple[str, nn.Module]:
    """Build plugin layer.

    Args:
        cfg (dict): cfg should contain:

            - type (str): identify plugin layer type.
            - layer args: args needed to instantiate a plugin layer.
        postfix (int, str): appended into norm abbreviation to
            create named layer. Default: ''.

    Returns:
        tuple[str, nn.Module]: The first one is the concatenation of
        abbreviation and postfix. The second is the created plugin layer.
    """
    # 已看過，構建插入層結構的實例對象
    # cfg = 層結構的config資料
    # postfix = 命名用的
    # kwargs = 實例化層結構時的設定

    if not isinstance(cfg, dict):
        # 如果傳入的cfg不是dict格式就會直接報錯
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        # 如果cfg當中沒有type就直接報錯，沒有type沒辦法知道要構建哪個實例對象
        raise KeyError('the cfg dict must contain the key "type"')
    # 拷貝一份config資料
    cfg_ = cfg.copy()

    # 將type資訊拿出來
    layer_type = cfg_.pop('type')
    if layer_type not in PLUGIN_LAYERS:
        # 如果指定層結構不在PLUGIN_LAYERS註冊器當中就會報錯，表示不支援
        raise KeyError(f'Unrecognized plugin type {layer_type}')

    # 獲取指定的class
    plugin_layer = PLUGIN_LAYERS.get(layer_type)
    # 從class獲取縮寫名稱
    abbr = infer_abbr(plugin_layer)

    # 檢查postfix是否合法
    assert isinstance(postfix, (int, str))
    # 創建此實例對象的名稱
    name = abbr + str(postfix)

    # 進行實例化
    layer = plugin_layer(**kwargs, **cfg_)

    # 將名稱以及實例化對象進行回傳
    return name, layer
