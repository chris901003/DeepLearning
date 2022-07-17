# Copyright (c) OpenMMLab. All rights reserved.
from ..runner import Sequential
from ..utils import Registry, build_from_cfg


def build_model_from_cfg(cfg, registry, default_args=None):
    """Build a PyTorch model from config dict(s). Different from
    ``build_from_cfg``, if cfg is a list, a ``nn.Sequential`` will be built.

    Args:
        cfg (dict, list[dict]): The config of modules, is is either a config
            dict or a list of config dicts. If cfg is a list, a
            the built modules will be wrapped with ``nn.Sequential``.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.

    Returns:
        nn.Module: A built nn module.
    """
    # 已看過
    # cfg = config文件當中model的部分
    # registry = 註冊器的實例對象
    # default_args = train_cfg以及test_cfg，新版都會是None

    if isinstance(cfg, list):
        # 如果傳入的cfg是list形式，我們會將list中的內容依序拿出，並且最後組成一個Sequential
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return Sequential(*modules)
    else:
        # 如果傳入的是dict就可以直接使用build_from_cfg
        return build_from_cfg(cfg, registry, default_args)


MODELS = Registry('model', build_func=build_model_from_cfg)
