# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Sequence

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES
from .augmentations import PytorchVideoTrans, TorchvisionTrans


@PIPELINES.register_module()
class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]):
            Either config dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        """ 已看過，構建資料處理的流水線
        Args:
            transforms: list[dict]，list長度就會是處理的層數，dict該層的處理方式
        """
        # 檢查transforms需要是Sequence格式
        assert isinstance(transforms, Sequence)
        self.transforms = []
        # 遍歷整個流水線的模塊
        for transform in transforms:
            if isinstance(transform, dict):
                # 如果transform是dict就會到這裡(比較多情況是會到這裡)
                if transform['type'].startswith('torchvision.'):
                    # 如果開頭是torchvision.就會到這裡
                    trans_type = transform.pop('type')[12:]
                    # 透過TorchvisionTrans進行實例化
                    transform = TorchvisionTrans(trans_type, **transform)
                elif transform['type'].startswith('pytorchvideo.'):
                    # 如果開頭是pytorchvideo.就會到這裡
                    trans_type = transform.pop('type')[13:]
                    # 透過PytorchVideoTrans進行實例化
                    transform = PytorchVideoTrans(trans_type, **transform)
                else:
                    # 其他就會到這裡使用mmaction2裏面的類進行實例化
                    transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                # 如果不是dict則該transform就會需要是一個函數或是可以呼叫的類
                # 這裡就直接添加到transforms當中
                self.transforms.append(transform)
            else:
                # 其他型態就直接報錯
                raise TypeError(f'transform must be callable or a dict, '
                                f'but got {type(transform)}')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """
        # 已看過，這裡會遍歷整個流水線處理資料

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
