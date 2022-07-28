# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmcv.runner.optimizer import OPTIMIZER_BUILDERS as MMCV_OPTIMIZER_BUILDERS
from mmcv.utils import Registry, build_from_cfg

OPTIMIZER_BUILDERS = Registry(
    'optimizer builder', parent=MMCV_OPTIMIZER_BUILDERS)


def build_optimizer_constructor(cfg):
    constructor_type = cfg.get('type')
    if constructor_type in OPTIMIZER_BUILDERS:
        return build_from_cfg(cfg, OPTIMIZER_BUILDERS)
    elif constructor_type in MMCV_OPTIMIZER_BUILDERS:
        return build_from_cfg(cfg, MMCV_OPTIMIZER_BUILDERS)
    else:
        raise KeyError(f'{constructor_type} is not registered '
                       'in the optimizer builder registry.')


def build_optimizer(model, cfg):
    """ 已看過，構建優化器
    Args:
        model: 模型本身
        cfg: config文件
    """
    # 拷貝一份config文件
    optimizer_cfg = copy.deepcopy(cfg)
    # 獲取構建的類型，沒有特別指定就是DefaultOptimizerConstructor
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    # 獲取有沒有特別指定的學習率
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    # 優化器構建
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    # 將模型放入獲取優化器實例對象
    optimizer = optim_constructor(model)
    return optimizer
