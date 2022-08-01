# Copyright (c) OpenMMLab. All rights reserved.


class BasePostprocessor:

    def __init__(self, text_repr_type='poly'):
        assert text_repr_type in ['poly', 'quad'
                                  ], f'Invalid text repr type {text_repr_type}'

        self.text_repr_type = text_repr_type

    def is_valid_instance(self, area, confidence, area_thresh,
                          confidence_thresh):
        # 已看過，檢查面積是否有大於最小面積以及平均置信度是否大於設定閾值

        # 如果沒有超過閾值就會返回False，否則就會是True
        return bool(area >= area_thresh and confidence > confidence_thresh)
