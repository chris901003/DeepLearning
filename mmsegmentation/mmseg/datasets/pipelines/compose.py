# Copyright (c) OpenMMLab. All rights reserved.
import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose(object):
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        # 已看過
        # transforms = 一系列對資料集的轉換方式，list(dict())裏面的每一個代表一種處理的步驟
        assert isinstance(transforms, collections.abc.Sequence)
        # 保存轉換方式的實例對象的空間
        self.transforms = []
        # 遍歷所有轉換方式
        for transform in transforms:
            if isinstance(transform, dict):
                # 如果transform是dict格式就會直接使用build_from_cfg進行實例化
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                # 如果是可以直接使用的函數，這裡就直接放到transforms裏面
                self.transforms.append(transform)
            else:
                # 其他狀況就直接報錯
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
