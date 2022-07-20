# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv

from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug(object):
    """Test-time augmentation with multiple scales and flipping.

    An example configuration is as followed:

    .. code-block::

        img_scale=(2048, 1024),
        img_ratios=[0.5, 1.0],
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
            scale=[(1024, 512), (1024, 512), (2048, 1024), (2048, 1024)]
            flip=[False, True, False, True]
            ...
        )

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (None | tuple | list[tuple]): Images scales for resizing.
        img_ratios (float | list[float]): Image ratios for resizing
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal" and "vertical". If flip_direction is list,
            multiple flip augmentations will be applied.
            It has no effect when flip == False. Default: "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 img_ratios=None,
                 flip=False,
                 flip_direction='horizontal'):
        """ 已看過，主要是在驗證集上面會使用到的圖像轉換流程
        Args:
            transforms: list[ConfigDict]，裏面包含了一系列的轉換
            img_scale: 在進行resize時指定的大小
            img_ratios: 在resize時的圖像高寬比，預設為None
            flip: 是否進行翻轉
            flip_direction: 翻轉的方向，這裡可以選擇3種不同的翻轉方式
        """
        if flip:
            # 當我們有使用翻轉時需要檢查，是否先經過翻轉後才進行填充
            trans_index = {
                key['type']: index
                for index, key in enumerate(transforms)
            }
            if 'RandomFlip' in trans_index and 'Pad' in trans_index:
                assert trans_index['RandomFlip'] < trans_index['Pad'], \
                    'Pad must be executed after RandomFlip when flip is True'
        # 這裡還是使用到Compose模塊將一系列的transforms放入
        self.transforms = Compose(transforms)
        if img_ratios is not None:
            # 如果有傳入img_ratios我們就會調整型態
            img_ratios = img_ratios if isinstance(img_ratios,
                                                  list) else [img_ratios]
            # 檢查img_ratios格式
            assert mmcv.is_list_of(img_ratios, float)
        if img_scale is None:
            # 沒有設定img_scale
            # mode 1: given img_scale=None and a range of image ratio
            self.img_scale = None
            assert mmcv.is_list_of(img_ratios, float)
        elif isinstance(img_scale, tuple) and mmcv.is_list_of(
                img_ratios, float):
            # 如果有設定img_scale且同時有設定img_ratios就會進來
            assert len(img_scale) == 2
            # mode 2: given a scale and a range of image ratio
            # 透過img_scale與img_ratios組成最後的img_scale
            self.img_scale = [(int(img_scale[0] * ratio),
                               int(img_scale[1] * ratio))
                              for ratio in img_ratios]
        else:
            # mode 3: given multiple scales
            # 只有傳入img_scale就直接變換格式之後保存
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
        # img_scale要不就是tuple要不就要是None
        assert mmcv.is_list_of(self.img_scale, tuple) or self.img_scale is None
        # 紀錄一些參數
        self.flip = flip
        self.img_ratios = img_ratios
        # 將flip_direction轉成list格式
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        # flip_direction內部需要是str格式
        assert mmcv.is_list_of(self.flip_direction, str)
        # 一些警告表示如果沒有開啟flip那麼flip_direction裏面也不會有作用
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

        aug_data = []
        if self.img_scale is None and mmcv.is_list_of(self.img_ratios, float):
            h, w = results['img'].shape[:2]
            img_scale = [(int(w * ratio), int(h * ratio))
                         for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    _results = results.copy()
                    _results['scale'] = scale
                    _results['flip'] = flip
                    _results['flip_direction'] = direction
                    data = self.transforms(_results)
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        return repr_str
