# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg

BBOX_ASSIGNERS = Registry('bbox_assigner')
BBOX_SAMPLERS = Registry('bbox_sampler')
BBOX_CODERS = Registry('bbox_coder')


def build_assigner(cfg, **default_args):
    """Builder of box assigner."""
    # 已看過，構建預測匡與真實匡的匹配方式
    return build_from_cfg(cfg, BBOX_ASSIGNERS, default_args)


def build_sampler(cfg, **default_args):
    """Builder of box sampler."""
    # 已看過，構建sampler實例化對象
    return build_from_cfg(cfg, BBOX_SAMPLERS, default_args)


def build_bbox_coder(cfg, **default_args):
    """Builder of box coder."""
    return build_from_cfg(cfg, BBOX_CODERS, default_args)
