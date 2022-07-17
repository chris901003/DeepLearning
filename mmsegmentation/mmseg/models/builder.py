# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

# 構建MODELS註冊器
MODELS = Registry('models', parent=MMCV_MODELS)
ATTENTION = Registry('attention', parent=MMCV_ATTENTION)

BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
SEGMENTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    # 已看過
    # 構建backbone，cfg內容就是構建backbone需要的東西，type就會是要構建backbone的類型
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    # 已看過
    # 構建預測頭，cfg內容就會是構建分類頭需要的配置，type就是要使用的分類頭類型
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    # 已看過
    # 構建損失計算，cfg內容就是構建損失的詳細內容，type就是要用哪種損失計算
    return LOSSES.build(cfg)


def build_segmentor(cfg, train_cfg=None, test_cfg=None):
    """
    :param cfg: config當中model部分
    :param train_cfg: 這裡會先是None(暫時不確定作用)，依據下方警告好像在新版本已經要移除了
    :param test_cfg: 這裡會先是None(暫時不確定作用)，依據下方警告好像在新版本已經要移除了
    :return:
    """
    """Build segmentor."""
    # 已看過

    # 以下都只是在檢查一些東西，最後return才是真正要創建model
    if train_cfg is not None or test_cfg is not None:
        # 如果有傳入train_cfg或是test_cfg會警告這種方式即將要被淘汰
        # 要將這些內容寫入到model當中
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    # 如果有train_cfg這裡會直接報錯
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    # 如果有test_cfg這裡會直接報錯
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '

    # 這裡才是真正要創建
    # SEGMENTORS在上面有定義為MODELS，這裡會調用SEGMENTORS的build函數去實例化對象
    return SEGMENTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
