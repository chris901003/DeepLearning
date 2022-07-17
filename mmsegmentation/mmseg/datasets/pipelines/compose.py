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
        # 已看過，圖像處理流的call函數，透過這裡將整個pipeline跑過一次，這裡每次只會處理一張照片
        # data = 一張圖像的資料，裡面會有要從哪裡讀取照片等等的資料

        for t in self.transforms:
            # 去調用對應轉換方式的__call__函數，將data傳入同時傳出的部分會將data進行更新
            data = t(data)
            if data is None:
                # 如果data是空的話就直接回傳None
                return None
        # 最後回傳data，data中的詳細內容可以用Debug模式下去看
        # data = dict{
        #   'img_metas': DataContainer [包含許多訓練圖像的資料],
        #   'img': DataContainer [訓練圖像的tensor以及相關資訊],
        #   'gt_semantic_seg': DataContainer [標註圖像的tensor以及相關資訊]
        # }
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
