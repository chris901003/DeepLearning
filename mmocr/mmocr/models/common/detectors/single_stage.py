# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models.detectors import \
    SingleStageDetector as MMDET_SingleStageDetector

from mmocr.models.builder import (DETECTORS, build_backbone, build_head,
                                  build_neck)


@DETECTORS.register_module()
class SingleStageDetector(MMDET_SingleStageDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        """ 已看過，單步驟檢測頭
        Args:
            backbone: backbone的設定資料
            neck: 將backbone的輸出進行加工，不一定會有
            bbox_head: 將提取出來的特徵進行預測，也就是預測頭
            train_cfg: train相關的設定
            test_cfg: test相關的設定
            pretrained: 預訓練權重相關資料
            init_cfg: 初始化設定方式
        """
        # 繼承自MMDET_SingleStageDetector，對繼承對象進行初始化
        super(MMDET_SingleStageDetector, self).__init__(init_cfg=init_cfg)
        if pretrained:
            # 如果有設定pretrained會跳出警告，新版本需要將pretrained寫到backbone當中
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            # 將預訓練權重資料放到backbone當中
            backbone.pretrained = pretrained
        # 構建backbone實例對象
        self.backbone = build_backbone(backbone)
        if neck is not None:
            # 構建neck實例對象
            self.neck = build_neck(neck)
        # 將train_cfg與test_cfg放到bbox_head的設定檔中
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        # 保存train_cfg與test_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
