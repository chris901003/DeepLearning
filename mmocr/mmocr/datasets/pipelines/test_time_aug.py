# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.compose import Compose


@PIPELINES.register_module()
class MultiRotateAugOCR:
    """Test-time augmentation with multiple rotations in the case that
    img_height > img_width.

    An example configuration is as follows:

    .. code-block::

        rotate_degrees=[0, 90, 270],
        transforms=[
            dict(
                type='ResizeOCR',
                height=32,
                min_width=32,
                max_width=160,
                keep_aspect_ratio=True),
            dict(type='ToTensorOCR'),
            dict(type='NormalizeOCR', **img_norm_cfg),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_shape', 'img_shape', 'valid_ratio'
                ]),
        ]

    After MultiRotateAugOCR with above configuration, the results are wrapped
    into lists of the same length as follows:

    .. code-block::

        dict(
            img=[...],
            img_shape=[...]
            ...
        )

    Args:
        transforms (list[dict]): Transformation applied for each augmentation.
        rotate_degrees (list[int] | None): Degrees of anti-clockwise rotation.
        force_rotate (bool): If True, rotate image by 'rotate_degrees'
            while ignore image aspect ratio.
    """

    def __init__(self, transforms, rotate_degrees=None, force_rotate=False):
        """ 已看過，測試時會用到的圖像轉換流
        Args:
            transforms: 變換方式設定資料
            rotate_degrees: 順時針旋轉的角度，list[int]
            force_rotate: 如果設定為True在旋轉的時候就有可能會改變原始圖像的高寬比，False就會保留原始圖像高寬比
        """
        # 將transforms進行實例化
        self.transforms = Compose(transforms)
        # 保存force_rotate設定
        self.force_rotate = force_rotate
        if rotate_degrees is not None:
            # 如果有傳入旋轉角度就會到這裡，如果傳入的rotate_degrees不是list包裝，就會用list進行包裝
            self.rotate_degrees = rotate_degrees if isinstance(
                rotate_degrees, list) else [rotate_degrees]
            # 檢查當中的值是否為int
            assert mmcv.is_list_of(self.rotate_degrees, int)
            for degree in self.rotate_degrees:
                # 設定的旋轉角度需要在[0, 360]之間，同時需要是90的倍數
                assert 0 <= degree < 360
                assert degree % 90 == 0
            if 0 not in self.rotate_degrees:
                # 只少需要設定旋轉0度
                self.rotate_degrees.append(0)
        else:
            # 如果沒有給定旋轉的角度就設定一個0度
            self.rotate_degrees = [0]

    def __call__(self, results):
        """Call function to apply test time augment transformation to results.

        Args:
            results (dict): Result dict contains the data to be transformed.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """
        # 已看過，進行測試模式當中的圖像處理流

        # 將當前圖像shape資料取出來，img_shape = (height, width, channel)
        img_shape = results['img_shape']
        # 獲取當前圖像的高寬
        ori_height, ori_width = img_shape[:2]
        if not self.force_rotate and ori_height <= ori_width:
            # 如果不希望圖像高寬比產生變化且原始高度小於原始寬度就會到這裡，將rotate_degrees設定成[0]
            rotate_degrees = [0]
        else:
            # 否則rotate_degrees就會拿初始化時傳入的旋轉角度方式
            rotate_degrees = self.rotate_degrees
        # 暫時存放資料的地方
        aug_data = []
        # 遍歷旋轉圖像的方式
        for degree in set(rotate_degrees):
            # 將results拷貝一份
            _results = results.copy()
            # 非0的旋轉角度就根據degree進行圖像的旋轉
            if degree == 0:
                pass
            elif degree == 90:
                _results['img'] = np.rot90(_results['img'], 1)
            elif degree == 180:
                _results['img'] = np.rot90(_results['img'], 2)
            elif degree == 270:
                _results['img'] = np.rot90(_results['img'], 3)
            # 最後將旋轉好的圖像放到圖像變換流當中
            data = self.transforms(_results)
            # 將data保存
            aug_data.append(data)
        # list of dict to dict of list
        # 遍歷aug_data當中有哪些key值
        aug_data_dict = {key: [] for key in aug_data[0]}
        # 遍歷aug_data當中的資料
        for data in aug_data:
            # 將相同的key的資料放到一起
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'rotate_degrees={self.rotate_degrees})'
        return repr_str
