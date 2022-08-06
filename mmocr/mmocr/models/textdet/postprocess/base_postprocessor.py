# Copyright (c) OpenMMLab. All rights reserved.


class BasePostprocessor:

    def __init__(self, text_repr_type='poly'):
        """ 已看過，在OCR當中的後處理最基礎類
        Args:
            text_repr_type: 最終標註匡的表達方式，如果是poly就是多邊形，如果是quad就是矩形
        """
        # 檢查輸入的text_repr_type是否合法
        assert text_repr_type in ['poly', 'quad'
                                  ], f'Invalid text repr type {text_repr_type}'

        # 保存下來
        self.text_repr_type = text_repr_type

    def is_valid_instance(self, area, confidence, area_thresh,
                          confidence_thresh):
        # 已看過，檢查面積是否有大於最小面積以及平均置信度是否大於設定閾值

        # 如果沒有超過閾值就會返回False，否則就會是True
        return bool(area >= area_thresh and confidence > confidence_thresh)
