# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import DETECTORS
from .single_stage_text_detector import SingleStageTextDetector
from .text_detector_mixin import TextDetectorMixin


@DETECTORS.register_module()
class PANet(TextDetectorMixin, SingleStageTextDetector):
    """The class for implementing PANet text detector:

    Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel
    Aggregation Network [https://arxiv.org/abs/1908.05900].
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
        """ 已看過，PANet的初始化部分
        Args:
            backbone: backbone設定內容
            neck: neck設定內容
            bbox_head: 預測頭的設定
            train_cfg: train時圖像資料的處理
            test_cfg: test時圖像的資料處理
            pretrained: 預訓練權重資料
            show_score: 是否將過程中的分數顯示
            init_cfg: 初始化設定
        """
        # 主要是在進行預測
        SingleStageTextDetector.__init__(self, backbone, neck, bbox_head,
                                         train_cfg, test_cfg, pretrained,
                                         init_cfg)
        # 主要是對預測進行後處理
        TextDetectorMixin.__init__(self, show_score)
