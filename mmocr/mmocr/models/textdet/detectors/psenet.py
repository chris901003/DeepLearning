# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import DETECTORS
from .single_stage_text_detector import SingleStageTextDetector
from .text_detector_mixin import TextDetectorMixin


@DETECTORS.register_module()
class PSENet(TextDetectorMixin, SingleStageTextDetector):
    """The class for implementing PSENet text detector: Shape Robust Text
    Detection with Progressive Scale Expansion Network.

    [https://arxiv.org/abs/1806.02559].
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 show_score=False,
                 init_cfg=None):
        """ 已看過，PSENet的初始化函數
        Args:
            backbone: backbone的設定資料
            neck: 將backbone的輸出進行加工，不一定會有
            bbox_head: 將提取出來的特徵進行預測，也就是預測頭
            train_cfg: train相關的設定
            test_cfg: test相關的設定
            pretrained: 預訓練權重相關資料
            show_score: 展示訓練分數
            init_cfg: 初始化設定方式
        """

        # 繼承自SingleStageTextDetector，對繼承對象進行初始化
        SingleStageTextDetector.__init__(self, backbone, neck, bbox_head,
                                         train_cfg, test_cfg, pretrained,
                                         init_cfg)
        # 繼承自TextDetectorMixin，對繼承對象進行初始化
        TextDetectorMixin.__init__(self, show_score)
