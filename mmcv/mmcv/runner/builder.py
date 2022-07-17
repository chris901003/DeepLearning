# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional

from ..utils import Registry

RUNNERS = Registry('runner')
RUNNER_BUILDERS = Registry('runner builder')


def build_runner_constructor(cfg: dict):
    # 已看過
    # 將配置文件傳入，透過註冊器構建實例化對象
    return RUNNER_BUILDERS.build(cfg)


def build_runner(cfg: dict, default_args: Optional[dict] = None):
    # 已看過
    runner_cfg = copy.deepcopy(cfg)
    constructor_type = runner_cfg.pop('constructor',
                                      'DefaultRunnerConstructor')
    # 透過build_runner_constructor構建runner前置作業
    # 將構建的類型以及構建的設定檔傳入，實例化runner_constructor實例對象
    runner_constructor = build_runner_constructor(
        dict(
            type=constructor_type,
            runner_cfg=runner_cfg,
            default_args=default_args))
    # 使用runner_constructor的__call__函數實例化最終需要的runner實例對象
    runner = runner_constructor()
    return runner
