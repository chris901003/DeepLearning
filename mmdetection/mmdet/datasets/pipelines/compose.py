# Copyright (c) OpenMMLab. All rights reserved.
import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        # 已看過，圖像一系列處理的流
        assert isinstance(transforms, collections.abc.Sequence)
        # 保存實例化對象
        self.transforms = []
        # 遍歷轉換方式
        for transform in transforms:
            if isinstance(transform, dict):
                # 透過註冊器進行實例化
                transform = build_from_cfg(transform, PIPELINES)
                # 將實例化對象保存下來
                self.transforms.append(transform)
            elif callable(transform):
                # 如果傳入的是可以直接call就保存
                self.transforms.append(transform)
            else:
                # 其他就跳出報錯
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
            str_ = t.__repr__()
            if 'Compose(' in str_:
                str_ = str_.replace('\n', '\n    ')
            format_string += '\n'
            format_string += f'    {str_}'
        format_string += '\n)'
        return format_string
