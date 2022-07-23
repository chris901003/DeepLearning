# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmcv.runner.optimizer import OPTIMIZER_BUILDERS as MMCV_OPTIMIZER_BUILDERS
from mmcv.utils import Registry, build_from_cfg

OPTIMIZER_BUILDERS = Registry(
    'optimizer builder', parent=MMCV_OPTIMIZER_BUILDERS)


def build_optimizer_constructor(cfg):
    # 已看過
    # 這裡cfg當中的內容可以直接看下面的build_optimizer呼叫時傳入的資料

    constructor_type = cfg.get('type')
    # 根據不同的type會透過不同的註冊表進行實例化優化器
    if constructor_type in OPTIMIZER_BUILDERS:
        # 如果是使用其他的會到這裡
        return build_from_cfg(cfg, OPTIMIZER_BUILDERS)
    elif constructor_type in MMCV_OPTIMIZER_BUILDERS:
        # 如果是使用DefaultOptimizerConstructor的會到這裡
        return build_from_cfg(cfg, MMCV_OPTIMIZER_BUILDERS)
    else:
        # 如果沒有匹配的就會到這裡報錯
        raise KeyError(f'{constructor_type} is not registered '
                       'in the optimizer builder registry.')


def build_optimizer(model, cfg):
    # 已看過
    # model = 網路模型
    # cfg = 優化器的配置，包含使用哪種優化器，以及優化器內的超參數

    # 拷貝一份cfg到optimizer_cfg當中
    optimizer_cfg = copy.deepcopy(cfg)
    # 如果沒有指定constructor就會使用默認的DefaultOptimizerConstructor
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    # 如果沒有設定就會是None，如果有放通常會是對於某個部分會有特別的權重
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    # optim_constructor = 一個實例對象，已經保存我們要創建的優化器
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    # 將模型放入就可以直接獲得最終需要的優化器實例對象
    # 如果是pytorch官方有實現的，這裡就會直接是官方的優化器實例對象
    optimizer = optim_constructor(model)
    return optimizer
