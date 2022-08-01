# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import lmdb
import mmcv
import numpy as np
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.loading import LoadAnnotations, LoadImageFromFile


@PIPELINES.register_module()
class LoadTextAnnotations(LoadAnnotations):
    """Load annotations for text detection.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        use_img_shape (bool): Use the shape of loaded image from
            previous pipeline ``LoadImageFromFile`` to generate mask.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 use_img_shape=False):
        """ 已看過，讀取偵測文字的標註訊息
        Args:
            with_bbox: 是否需要加載bbox的資訊
            with_label: 是否需要加載label的資訊
            with_mask: 是否需要加載mask的資訊
            with_seg: 是否需要加載seg的資訊
            poly2mask: 是否需要將poly格式轉成mask格式
            use_img_shape: 是否需要使用LoadImageFromFile的圖像大小生成mask
        """
        # 繼承自LoadAnnotations，對繼承對象初始化
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask)

        # 保存參數
        self.use_img_shape = use_img_shape

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """
        # 已看過，將輸入的polygons轉成ndarray格式同時檢查長度是否為偶數同時有沒有超過三個點

        polygons = [np.array(p).astype(np.float32) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        # 已看過，獲取masks資料
        # 將標註訊息拿出來
        ann_info = results['ann_info']
        # 獲取圖像高寬
        h, w = results['img_info']['height'], results['img_info']['width']
        if self.use_img_shape:
            # 是否需要用ori_shape替代img_info中的height與width
            if results.get('ori_shape', None):
                h, w = results['ori_shape'][:2]
                results['img_info']['height'] = h
                results['img_info']['width'] = w
            else:
                warnings.warn('"ori_shape" not in results, use the shape '
                              'in "img_info" instead.')
        # 獲取masks資料，gt_masks=list[list[list]]，第一層list會是該圖像中有多少個標註，第二個list會是1，第三個list會是偶數長度會是一對一對的x與y
        gt_masks = ann_info['masks']
        if self.poly2mask:
            # 如果是需要poly格式轉到mask就會到這裡
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            # 否則就會到這裡轉換，gt_masks會是PolygonMasks的實例對象
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        # 獲取有被標記成crowd的或是標記有問題的部分
        gt_masks_ignore = ann_info.get('masks_ignore', None)
        if gt_masks_ignore is not None:
            # 這裡會做的事情與上面相同
            if self.poly2mask:
                gt_masks_ignore = BitmapMasks(
                    [self._poly2mask(mask, h, w) for mask in gt_masks_ignore],
                    h, w)
            else:
                gt_masks_ignore = PolygonMasks([
                    self.process_polygons(polygons)
                    for polygons in gt_masks_ignore
                ], h, w)
            # 將資料保存下來
            results['gt_masks_ignore'] = gt_masks_ignore
            results['mask_fields'].append('gt_masks_ignore')

        # 將gt_masks資料保存下來
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        # 回傳results
        return results


@PIPELINES.register_module()
class LoadImageFromNdarray(LoadImageFromFile):
    """Load an image from np.ndarray.

    Similar with :obj:`LoadImageFromFile`, but the image read from
    ``results['img']``, which is np.ndarray.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        # 已看過，從ndarray當中讀取圖像
        # 檢查圖像是否為uint8格式，如果不是就會報錯
        assert results['img'].dtype == 'uint8'

        # 將img資料讀取出來
        img = results['img']
        if self.color_type == 'grayscale' and img.shape[2] == 3:
            # 如果需要轉成灰階圖像就會到這裡，這裡是將彩色圖像轉成灰階所以會先檢查是否為RGB圖像
            img = mmcv.bgr2gray(img, keepdim=True)
        if self.color_type == 'color' and img.shape[2] == 1:
            # 如果需要轉成RGB圖像會到這裡，這裡是將灰階圖像轉成RGB圖像所以會先檢查是否為灰階圖像
            img = mmcv.gray2bgr(img)
        if self.to_float32:
            # 如果需要轉成float32格式就會在這裡轉換
            img = img.astype(np.float32)

        # 更新results當中資料
        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadImageFromLMDB(object):
    """Load an image from lmdb file.

    Similar with :obj:'LoadImageFromFile', but the image read from
    "results['img_info']['filename']", which is a data index of lmdb file.
    """

    def __init__(self, color_type='color'):
        self.color_type = color_type
        self.env = None
        self.txn = None

    def __call__(self, results):
        img_key = results['img_info']['filename']
        lmdb_path = results['img_prefix']

        # lmdb env
        if self.env is None:
            self.env = lmdb.open(
                lmdb_path,
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        # read image
        with self.env.begin(write=False) as txn:
            imgbuf = txn.get(img_key.encode('utf-8'))
            try:
                img = mmcv.imfrombytes(imgbuf, flag=self.color_type)
            except IOError:
                print('Corrupted image for {}'.format(img_key))
                return None

            results['filename'] = img_key
            results['ori_filename'] = img_key
            results['img'] = img
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            results['img_fields'] = ['img']
            return results

    def __repr__(self):
        return '{} (color_type={})'.format(self.__class__.__name__,
                                           self.color_type)

    def __del__(self):
        if self.env is not None:
            self.env.close()
