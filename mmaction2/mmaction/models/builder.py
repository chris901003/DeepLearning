# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
RECOGNIZERS = MODELS
LOSSES = MODELS
LOCALIZERS = MODELS

try:
    from mmdet.models.builder import DETECTORS, build_detector
except (ImportError, ModuleNotFoundError):
    # Define an empty registry and building func, so that can import
    DETECTORS = MODELS

    def build_detector(cfg, train_cfg, test_cfg):
        warnings.warn(
            'Failed to import `DETECTORS`, `build_detector` from '
            '`mmdet.models.builder`. You will be unable to register or build '
            'a spatio-temporal detection model. ')


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_recognizer(cfg, train_cfg=None, test_cfg=None):
    """Build recognizer."""
    # 已看過，構建辨識器
    # cfg = 構建的配置文件
    # train_cfg = 在train時候對圖像的額外處理方式
    # test_cfg = 在test時候對圖像的額外處理方式
    if train_cfg is not None or test_cfg is not None:
        # 如果有使用train_cfg或是test_cfg就會跳出警告，這兩個已經要被淘汰了
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model. Details see this '
            'PR: https://github.com/open-mmlab/mmaction2/pull/629',
            UserWarning)
    assert cfg.get(
        'train_cfg'
    ) is None or train_cfg is None, 'train_cfg specified in both outer field and model field'  # noqa: E501
    assert cfg.get(
        'test_cfg'
    ) is None or test_cfg is None, 'test_cfg specified in both outer field and model field '  # noqa: E501
    # 構建模型
    return RECOGNIZERS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_localizer(cfg):
    """Build localizer."""
    return LOCALIZERS.build(cfg)


def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    # 已看過，構建模型實例化對象函數
    # 將config資料拷貝一份
    args = cfg.copy()
    # 獲取使用哪個模型
    obj_type = args.pop('type')
    if obj_type in LOCALIZERS:
        # 如果是屬於Localizer的類型就會到這裡，使用localizer的註冊器進行模型實例化
        return build_localizer(cfg)
    if obj_type in RECOGNIZERS:
        # 如果是屬於Recognizer的類型就會到這裡，使用recognizer的註冊器進行模型實例化
        return build_recognizer(cfg, train_cfg, test_cfg)
    if obj_type in DETECTORS:
        # 如果是屬於Detector的類型就會到這裡，使用detector的註冊器進行模型實例化
        if train_cfg is not None or test_cfg is not None:
            warnings.warn(
                'train_cfg and test_cfg is deprecated, '
                'please specify them in model. Details see this '
                'PR: https://github.com/open-mmlab/mmaction2/pull/629',
                UserWarning)
        return build_detector(cfg, train_cfg, test_cfg)
    # 不在以上幾種註冊器當中就會到這裡
    model_in_mmdet = ['FastRCNN']
    if obj_type in model_in_mmdet:
        # 如果是時空任務就需要安裝mmdet才可以
        raise ImportError(
            'Please install mmdet for spatial temporal detection tasks.')
    # 其他就會到這裡報錯
    raise ValueError(f'{obj_type} is not registered in '
                     'LOCALIZERS, RECOGNIZERS or DETECTORS')


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)
