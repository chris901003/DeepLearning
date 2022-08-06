# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import DETECTORS
from .single_stage_text_detector import SingleStageTextDetector
from .text_detector_mixin import TextDetectorMixin


@DETECTORS.register_module()
class TextSnake(TextDetectorMixin, SingleStageTextDetector):
    """The class for implementing TextSnake text detector: TextSnake: A
    Flexible Representation for Detecting Text of Arbitrary Shapes.

    [https://arxiv.org/abs/1807.01544]
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
        """ 已看過，TextSnake的初始化函數
        Args:
            backbone: backbone設定參數
            neck: 處理從backbone輸出的特徵圖
            bbox_head: 解碼頭設定
            train_cfg: 在train模式下的額外處理方式
            test_cfg: 在test模式下的額外處理方式
            pretrained: 預訓練權重資料
            show_score: 在將結果標注到圖像時是否顯示預測的置信度
            init_cfg: 初始化方式
        """
        # 主要是在預測圖像的
        SingleStageTextDetector.__init__(self, backbone, neck, bbox_head,
                                         train_cfg, test_cfg, pretrained,
                                         init_cfg)
        # 主要是將後處理後的標註座標放到圖像上面進行匡選
        TextDetectorMixin.__init__(self, show_score)
