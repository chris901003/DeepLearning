# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from .builder import RUNNER_BUILDERS, RUNNERS


@RUNNER_BUILDERS.register_module()
class DefaultRunnerConstructor:
    """Default constructor for runners.

    Custom existing `Runner` like `EpocBasedRunner` though `RunnerConstructor`.
    For example, We can inject some new properties and functions for `Runner`.

    Example:
        >>> from mmcv.runner import RUNNER_BUILDERS, build_runner
        >>> # Define a new RunnerReconstructor
        >>> @RUNNER_BUILDERS.register_module()
        >>> class MyRunnerConstructor:
        ...     def __init__(self, runner_cfg, default_args=None):
        ...         if not isinstance(runner_cfg, dict):
        ...             raise TypeError('runner_cfg should be a dict',
        ...                             f'but got {type(runner_cfg)}')
        ...         self.runner_cfg = runner_cfg
        ...         self.default_args = default_args
        ...
        ...     def __call__(self):
        ...         runner = RUNNERS.build(self.runner_cfg,
        ...                                default_args=self.default_args)
        ...         # Add new properties for existing runner
        ...         runner.my_name = 'my_runner'
        ...         runner.my_function = lambda self: print(self.my_name)
        ...         ...
        >>> # build your runner
        >>> runner_cfg = dict(type='EpochBasedRunner', max_epochs=40,
        ...                   constructor='MyRunnerConstructor')
        >>> runner = build_runner(runner_cfg)
    """

    def __init__(self, runner_cfg: dict, default_args: Optional[dict] = None):
        """
        :param runner_cfg: runner相關的設定參數，保括使用哪種runner
        :param default_args: 裏面包含許多資料，有完整模型，優化器，資料集檔案位置，logger以及meta
        """
        # 已看過

        # 檢查runner_cfg型態是否為dict
        if not isinstance(runner_cfg, dict):
            raise TypeError('runner_cfg should be a dict',
                            f'but got {type(runner_cfg)}')
        # 保存傳入的變量
        self.runner_cfg = runner_cfg
        self.default_args = default_args

    def __call__(self):
        return RUNNERS.build(self.runner_cfg, default_args=self.default_args)
