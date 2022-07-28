# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    # 已看過，構建模型特徵提取骨幹部分，BACKBONES註冊器裏面有大量的特徵提取骨幹
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    # 已看過，構建預測頭
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    # 已看過，構建損失函數
    return LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    # 已看過，主要進行構建偵測模型
    # 檢查一些相關內容，有些已經要被淘汰的部分就會在這裡跳出警告
    # 主要是train_cfg與test_cfg已經不會獨立寫，這裡都會放到cfg當中，所以傳入的train_cfg與test_cfg都應該要是None
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    # 透過DETECTORS註冊器進行實例化模型
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
