# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv

from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug:
    """Test-time augmentation with multiple scales and flipping.

    An example configuration is as followed:

    .. code-block::

        img_scale=[(1333, 400), (1333, 800)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]

    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:

    .. code-block::

        dict(
            img=[...],
            img_shape=[...],
            scale=[(1333, 400), (1333, 400), (1333, 800), (1333, 800)]
            flip=[False, True, False, True]
            ...
        )

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple] | None): Images scales for resizing.
        scale_factor (float | list[float] | None): Scale factors for resizing.
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal", "vertical" and "diagonal". If
            flip_direction is a list, multiple flip augmentations will be
            applied. It has no effect when flip == False. Default:
            "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale=None,
                 scale_factor=None,
                 flip=False,
                 flip_direction='horizontal'):
        """ 已看過，測試時用的圖像讀取，可以食實現縮放以及翻轉
        Args:
            transforms: 圖像變換方式
            img_scale: 圖像縮放大小
            scale_factor: 縮放比例
            flip: 是否進行翻轉
            flip_direction: 翻轉方向
        """

        # 將變換方式用Compose進行實例化
        self.transforms = Compose(transforms)
        # img_scale與scale_factor只能選擇一個進行設定
        assert (img_scale is None) ^ (scale_factor is None), (
            'Must have but only one variable can be set')
        if img_scale is not None:
            # 如果是設定img_scale就會到這裡
            # 將img_scale用list進行包裝
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
            # 設定成scale模式
            self.scale_key = 'scale'
            assert mmcv.is_list_of(self.img_scale, tuple)
        else:
            # 如果是輸入scale_factor就會到這裡
            # 將scale_factor用list包裝
            self.img_scale = scale_factor if isinstance(
                scale_factor, list) else [scale_factor]
            # 設定成sale_factor模式
            self.scale_key = 'scale_factor'

        # 紀錄反轉以及翻轉模式
        self.flip = flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """
        # 已看過，call函數

        # 會將變換結果放到aug_data當中
        aug_data = []
        # 構建翻轉的參數，預先會有一個False與None
        flip_args = [(False, None)]
        if self.flip:
            # 如果有設定flip就會到這裡
            # 在flip設定當中添加上True以及翻轉方向
            flip_args += [(True, direction)
                          for direction in self.flip_direction]
        # 遍歷指定的圖像大小
        for scale in self.img_scale:
            # 遍歷不同的翻轉方式，同時也會有不翻轉的情況
            for flip, direction in flip_args:
                # 拷貝一份results
                _results = results.copy()
                # 將scale模式傳入
                _results[self.scale_key] = scale
                # 當前是否進行翻轉
                _results['flip'] = flip
                # 當前翻轉方向
                _results['flip_direction'] = direction
                # 進行變換
                data = self.transforms(_results)
                # 將結果保存到aug_data當中
                aug_data.append(data)
        # list of dict to dict of list，獲取aug_data當中的key值
        aug_data_dict = {key: [] for key in aug_data[0]}
        # 將aug_data當中的內容添加上去
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
