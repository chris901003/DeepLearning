# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        """
        :param to_float32: 是否需要將圖像轉換成float32，如果為False就會是uint8的型態
        :param color_type: 預設為color，目前不確定作用
        :param file_client_args:
        :param imdecode_backend: 用哪個模組開啟圖像，這裡預設用cv2
        """
        # 已看過
        # 該class主要是用來從資料夾當中讀取照片

        # 將傳入的值進行保存
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        # 已看過，從results中的資訊將圖像載入進來
        # results = 這次要載入的圖像的詳細資訊

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            # 如果有提供訓練圖像的前面路徑位置就會將前面的路徑位置添加上去
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            # 沒有前面路徑就直接使用相對路徑的方式
            filename = results['img_info']['filename']
        # 將訓練圖像的檔案位置傳入，使用二進制方式讀取出來
        img_bytes = self.file_client.get(filename)
        # 將讀取的資料以及color模式以及要使用哪種方式開啟傳入
        # img = 讀取完成的圖像，ndarray [width, height, channel]，dtype=uint8
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            # 如果有需要轉換成float32就直接轉換
            img = img.astype(np.float32)

        # 將原始訓練圖像檔案路徑保存
        results['filename'] = filename
        # 原始訓練圖像檔案名稱
        results['ori_filename'] = results['img_info']['filename']
        # 圖像保存
        results['img'] = img
        # 保存當前訓練圖片的大小，高寬以及通道數
        results['img_shape'] = img.shape
        # 保存原始圖片的大小，高寬以及通道數
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        # 初始化設定pad_shape，這裡用原始圖像大小
        results['pad_shape'] = img.shape
        # 縮放比例設定成1
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            # 將均值設定成0
            mean=np.zeros(num_channels, dtype=np.float32),
            # 方差設定成1
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        """
        :param reduce_zero_label: 是否要將所有的label的值減去一，主要是用在數據集0表示背景，預設為False
        :param file_client_args: 跟著預設就行了，這裡作用不太清楚
        :param imdecode_backend: 預設使用pillow進行圖像載入
        """
        # 已看過
        # 該class主要目的是將標註好的圖像載入進來

        # 將參數保留下來
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """
        # 已看過，主要是在讀取標註訊息的資料

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            # 如果有給seg_prefix就將前面的路徑放到filename當中
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            # 如果沒有傳入前面的路徑就直接使用相對路徑
            filename = results['ann_info']['seg_map']
        # img_byte = 標註圖像的二進制表達方式
        img_bytes = self.file_client.get(filename)
        # 將圖像讀取出來
        # get_semantic_seg = ndarray [width, height]，後面透過squeeze將channel維度壓縮掉
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion，將0變成忽略的index
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
