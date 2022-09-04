# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import build_optimizer


def build_optimizers(model, cfgs):
    """Build multiple optimizers from configs.

    If `cfgs` contains several dicts for optimizers, then a dict for each
    constructed optimizers will be returned.
    If `cfgs` only contains one optimizer config, the constructed optimizer
    itself will be returned.

    For example,

    1) Multiple optimizer configs:

    .. code-block:: python

        optimizer_cfg = dict(
            model1=dict(type='SGD', lr=lr),
            model2=dict(type='SGD', lr=lr))

    The return dict is
    ``dict('model1': torch.optim.Optimizer, 'model2': torch.optim.Optimizer)``

    2) Single optimizer config:

    .. code-block:: python

        optimizer_cfg = dict(type='SGD', lr=lr)

    The return is ``torch.optim.Optimizer``.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        cfgs (dict): The config dict of the optimizer.

    Returns:
        dict[:obj:`torch.optim.Optimizer`] | :obj:`torch.optim.Optimizer`:
            The initialized optimizers.
    """
    # 構建優化器存放的dict
    optimizers = {}
    if hasattr(model, 'module'):
        model = model.module
    # determine whether 'cfgs' has several dicts for optimizers
    # 先將is_dict_of_dict設定成True
    is_dict_of_dict = True
    for key, cfg in cfgs.items():
        # 遍歷cfgs當中的資料
        if not isinstance(cfg, dict):
            # 如果當中不是dict就將is_dict_of_dict設定成False
            is_dict_of_dict = False
    if is_dict_of_dict:
        # 如果是dict_of_dict就用遞歸進行構建，這裡最後回傳傳的會是dict構成的優化器，key會與傳入的時候相同
        # 只是value就會是優化器實例對象
        # 在GAN當中通常會有兩個優化器，一個是給生成器的另一個是給鑑別器的
        for key, cfg in cfgs.items():
            cfg_ = cfg.copy()
            module = getattr(model, key)
            optimizers[key] = build_optimizer(module, cfg_)
        return optimizers

    # 透過build_optimizer進行構建
    return build_optimizer(model, cfgs)
