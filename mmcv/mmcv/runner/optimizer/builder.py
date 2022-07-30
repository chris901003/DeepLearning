# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
from typing import Dict, List

import torch

from ...utils import Registry, build_from_cfg

OPTIMIZERS = Registry('optimizer')
OPTIMIZER_BUILDERS = Registry('optimizer builder')


def register_torch_optimizers() -> List:
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            OPTIMIZERS.register_module()(_optim)
            torch_optimizers.append(module_name)
    return torch_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()


def build_optimizer_constructor(cfg: Dict):
    return build_from_cfg(cfg, OPTIMIZER_BUILDERS)


def build_optimizer(model, cfg: Dict):
    """ 已看過，構建優化器
    Args:
        model: 模型本身
        cfg: 優化器設定相關內容
    """

    # 將優化器設定內容拷貝一份
    optimizer_cfg = copy.deepcopy(cfg)
    # 檢查config當中使否有指定的constructor如果沒有就會是DefaultOptimizerConstructor
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    # 看有沒有特別設定哪些層結構的學習率
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    # 構建構建優化器實例對象
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    # 將模型放入，構建優化器實例對象
    optimizer = optim_constructor(model)
    # 回傳優化器實例對象
    return optimizer
