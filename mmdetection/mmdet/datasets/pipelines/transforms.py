# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random

from mmdet.core import BitmapMasks, PolygonMasks, find_inside_bboxes
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.utils import log_img_scale
from ..builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None


@PIPELINES.register_module()
class Resize:
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used. If the input dict contains the key
    "scale_factor" (if MultiScaleFlipAug does not give img_scale but
    scale_factor), the actual scale will be computed by image shape and
    scale_factor.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:

    - ``ratio_range is not None``: randomly sample a ratio from the ratio \
      range and multiply it with the image scale.
    - ``ratio_range is None`` and ``multiscale_mode == "range"``: randomly \
      sample a scale from the multiscale range.
    - ``ratio_range is None`` and ``multiscale_mode == "value"``: randomly \
      sample a scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        override (bool, optional): Whether to override `scale` and
            `scale_factor` so as to call resize twice. Default False. If True,
            after the first resizing, the existed `scale` and `scale_factor`
            will be ignored so the second resizing can be allowed.
            This option is a work-around for multiple times of resize in DETR.
            Defaults to False.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 interpolation='bilinear',
                 override=False):
        """ 已看過，對圖像以及標註匡以及mask進行大小調整
        Args:
            img_scale: 圖像可能調整到的大小，會是一個tuple或是list，裡面會有需多不同的尺度
            multiscale_mode: 會有range以及value兩個選項，range就是給定一個範圍然後隨機選，value就會是給幾個固定的且從中選一個
            ratio_range: 透過高寬比進行大小調整
            keep_ratio: 在回傳時是否需要回傳比例
            bbox_clip_border: 是否需要將圖像外的標註匡進行裁切
            backend: 使用的模組
            interpolation: 差值方式
            override: 是否需要兩次resize，預設為False
        """
        if img_scale is None:
            # 如果沒有傳入img_scale就直接給None
            self.img_scale = None
        else:
            # 有傳入img_scale會到這裡
            if isinstance(img_scale, list):
                # 如果是list型態就直接保存
                self.img_scale = img_scale
            else:
                # 不是list就在外面加上一層list
                self.img_scale = [img_scale]
            # 檢查list當中的資料要是tuple格式
            assert mmcv.is_list_of(self.img_scale, tuple)

        # 這裡會提供3種resize決定大小的方式
        if ratio_range is not None:
            # 如果ratio_range不是空就會進來
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            # 如果沒有傳入ratio_range就看我們使用的模式
            assert multiscale_mode in ['value', 'range']

        # 保存一些傳入的參數
        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.interpolation = interpolation
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, \
                where ``img_scale`` is the selected image scale and \
                ``scale_idx`` is the selected index in the given candidates.
        """

        # 已看過，從list當中進行雖機選取
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and upper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where \
                ``img_scale`` is sampled scale and None is just a placeholder \
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where \
                ``scale`` is sampled ratio multiplied with ``img_scale`` and \
                None is just a placeholder to be consistent with \
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """
        # 已看過，獲取隨機的resize後的圖像大小

        if self.ratio_range is not None:
            # 如果設定的是縮放比例的範圍會在這裡
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            # 如果設定的是固定值會在這裡，會從img_scale中選擇一種尺寸
            # 回傳選到的尺寸以及選到的index
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            # 其他情況就直接報錯
            raise NotImplementedError

        # 記錄下scale以及index
        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        # 已看過，對圖像進行resize
        # 遍歷與圖像相關的資料
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                # 如果有需要保存縮放比例就會到這裡
                # 透過imrescale進行調整
                img, scale_factor = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                # 獲取新的圖像高寬
                new_h, new_w = img.shape[:2]
                # 獲取原始圖像的高寬
                h, w = results[key].shape[:2]
                # 計算高寬縮放多少倍
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                # 透過imresize進行調整
                img, w_scale, h_scale = mmcv.imresize(
                    results[key],
                    results['scale'],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)
            # 更新圖像
            results[key] = img

            # 縮放比例，可以給標註匡進行使用
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            # 保存新圖像的大小
            results['img_shape'] = img.shape
            # in case that there is no padding
            # 更新pad的大小
            results['pad_shape'] = img.shape
            # 保存縮放大小
            results['scale_factor'] = scale_factor
            # 使否保存ratio
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        # 已看過，對標註匡進行調整
        for key in results.get('bbox_fields', []):
            # 將標註匡內容乘上縮放比例
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                # 如果需要將超出圖像的部分去除就會在這裡去除
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            # 更新標註匡訊息
            results[key] = bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        # 已看過，對mask部分進行resize
        # 遍歷results當中與masks有關的部分
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                results[key] = results[key].rescale(results['scale'])
            else:
                # 使用mask實例對象當中的resize進行大小變換
                results[key] = results[key].resize(results['img_shape'][:2])

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results[key],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        # 已看過，對圖像進行resize

        if 'scale' not in results:
            # 如果results當中沒有指定scale就會到這裡
            if 'scale_factor' in results:
                # 如果有指定scale_factor會到這
                # 獲取原始圖像大小
                img_shape = results['img'].shape[:2]
                # 獲取縮放比例
                scale_factor = results['scale_factor']
                # 縮放比例需要是float格式
                assert isinstance(scale_factor, float)
                # 獲取縮放的大小，會是原始大小乘上縮放倍率
                results['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                # 否則就會是隨機的大小
                self._random_scale(results)
        else:
            # 如果已經經過一次resize會到這裡
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                # 如果可以重新再選擇一次resize大小會到這裡
                # 將之前的scale去除
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                # 重新隨機獲取一個
                self._random_scale(results)

        # 將圖像進行resize
        self._resize_img(results)
        # 將標註匡進行resize
        self._resize_bboxes(results)
        # 將mask進行resize
        self._resize_masks(results)
        # 將seg進行resize
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class RandomFlip:
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    When random flip is enabled, ``flip_ratio``/``direction`` can either be a
    float/string or tuple of float/string. There are 3 flip modes:

    - ``flip_ratio`` is float, ``direction`` is string: the image will be
        ``direction``ly flipped with probability of ``flip_ratio`` .
        E.g., ``flip_ratio=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.
    - ``flip_ratio`` is float, ``direction`` is list of string: the image will
        be ``direction[i]``ly flipped with probability of
        ``flip_ratio/len(direction)``.
        E.g., ``flip_ratio=0.5``, ``direction=['horizontal', 'vertical']``,
        then image will be horizontally flipped with probability of 0.25,
        vertically with probability of 0.25.
    - ``flip_ratio`` is list of float, ``direction`` is list of string:
        given ``len(flip_ratio) == len(direction)``, the image will
        be ``direction[i]``ly flipped with probability of ``flip_ratio[i]``.
        E.g., ``flip_ratio=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with probability
        of 0.3, vertically with probability of 0.5.

    Args:
        flip_ratio (float | list[float], optional): The flipping probability.
            Default: None.
        direction(str | list[str], optional): The flipping direction. Options
            are 'horizontal', 'vertical', 'diagonal'. Default: 'horizontal'.
            If input is a list, the length must equal ``flip_ratio``. Each
            element in ``flip_ratio`` indicates the flip probability of
            corresponding direction.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        """ 已看過，隨機翻轉
        Args:
            flip_ratio: 隨機翻轉概率
            direction: 翻轉方向
        """
        if isinstance(flip_ratio, list):
            # flip_ratio是list會進入到這裡
            assert mmcv.is_list_of(flip_ratio, float)
            # 翻轉概率的總和不能大於1
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            # 如果傳入的是一個值就會到這裡，會檢查該值必須在[0, 1]之間
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            # 如果沒有傳入翻轉概率就直接pass
            pass
        else:
            # 其他狀態就是報錯
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        # 保存翻轉概率
        self.flip_ratio = flip_ratio

        # 可以接受3中翻轉方式
        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            # 檢查是否為合法的3種方式
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            # 檢查list當中的翻轉方式是否都是合法
            assert set(direction).issubset(set(valid_directions))
        else:
            # 其他狀況就直接報錯
            raise ValueError('direction must be either str or list of str')
        # 保存翻轉方式
        self.direction = direction

        if isinstance(flip_ratio, list):
            # 翻轉概率以及翻轉方式要可以對應上
            assert len(self.flip_ratio) == len(self.direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        # 已看過，對標註匡進行指定方向的翻轉

        # 檢查標註匡是否有問題
        assert bboxes.shape[-1] % 4 == 0
        # 將標註匡拷貝一份
        flipped = bboxes.copy()
        if direction == 'horizontal':
            # 如果是水平翻轉會在這裡
            w = img_shape[1]
            # 水平翻轉只會跟x會有關係
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            # 如果是垂直翻轉會在這裡
            h = img_shape[0]
            # 垂直翻轉會跟y有關係
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            # 如果是對角線翻轉會到這裡
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            # 其他類型的翻轉就會直接報錯
            raise ValueError(f"Invalid flipping direction '{direction}'")
        # 回傳調整後的結果
        return flipped

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        # 已看過，對圖像進行翻轉
        if 'flip' not in results:
            # 如果results當中沒有flip內容就會進行構建
            if isinstance(self.direction, list):
                # None means non-flip
                # 如果是個list就會在最後面添加上None
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                # 如果不是list就轉成list並且最後面會有None
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                # 獲取不翻轉的概率，就是1-翻轉概率
                non_flip_ratio = 1 - sum(self.flip_ratio)
                # 獲取翻轉概率的list
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                # 獲取不翻轉的概率
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                # 構建翻轉概率的list
                flip_ratio_list = [single_ratio] * (len(direction_list) - 1) + [non_flip_ratio]

            # choice = 從direction_list當中選出一個，flip_ratio_list就會是每一個被選中的概率
            # cur_dir = 選出的那個，如果選到的是None表示不進行翻轉
            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)

            # 如果cur_dir不是None就將結果放到result當中
            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            # 將翻轉的方向保留
            results['flip_direction'] = cur_dir
        if results['flip']:
            # 如果需要進行翻轉就會進來
            # flip image，遍歷img_fields當中的內容
            for key in results.get('img_fields', ['img']):
                # 將指定的key的value進行翻轉
                results[key] = mmcv.imflip(results[key], direction=results['flip_direction'])
            # flip bboxes，遍歷bbox_fields當中的內容
            for key in results.get('bbox_fields', []):
                # 將指定的key的value進行翻轉，這裡會是標註匡的翻轉
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks，mask翻轉
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])

            # flip seg，seg翻轉
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(results[key], direction=results['flip_direction'])
        # 將調整過後的results回傳
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@PIPELINES.register_module()
class RandomShift:
    """Shift the image and box given shift pixels and probability.

    Args:
        shift_ratio (float): Probability of shifts. Default 0.5.
        max_shift_px (int): The max pixels for shifting. Default 32.
        filter_thr_px (int): The width and height threshold for filtering.
            The bbox and the rest of the targets below the width and
            height threshold will be filtered. Default 1.
    """

    def __init__(self, shift_ratio=0.5, max_shift_px=32, filter_thr_px=1):
        assert 0 <= shift_ratio <= 1
        assert max_shift_px >= 0
        self.shift_ratio = shift_ratio
        self.max_shift_px = max_shift_px
        self.filter_thr_px = int(filter_thr_px)
        # The key correspondence from bboxes to labels.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }

    def __call__(self, results):
        """Call function to random shift images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Shift results.
        """
        if random.random() < self.shift_ratio:
            img_shape = results['img'].shape[:2]

            random_shift_x = random.randint(-self.max_shift_px,
                                            self.max_shift_px)
            random_shift_y = random.randint(-self.max_shift_px,
                                            self.max_shift_px)
            new_x = max(0, random_shift_x)
            ori_x = max(0, -random_shift_x)
            new_y = max(0, random_shift_y)
            ori_y = max(0, -random_shift_y)

            # TODO: support mask and semantic segmentation maps.
            for key in results.get('bbox_fields', []):
                bboxes = results[key].copy()
                bboxes[..., 0::2] += random_shift_x
                bboxes[..., 1::2] += random_shift_y

                # clip border
                bboxes[..., 0::2] = np.clip(bboxes[..., 0::2], 0, img_shape[1])
                bboxes[..., 1::2] = np.clip(bboxes[..., 1::2], 0, img_shape[0])

                # remove invalid bboxes
                bbox_w = bboxes[..., 2] - bboxes[..., 0]
                bbox_h = bboxes[..., 3] - bboxes[..., 1]
                valid_inds = (bbox_w > self.filter_thr_px) & (
                    bbox_h > self.filter_thr_px)
                # If the shift does not contain any gt-bbox area, skip this
                # image.
                if key == 'gt_bboxes' and not valid_inds.any():
                    return results
                bboxes = bboxes[valid_inds]
                results[key] = bboxes

                # label fields. e.g. gt_labels and gt_labels_ignore
                label_key = self.bbox2label.get(key)
                if label_key in results:
                    results[label_key] = results[label_key][valid_inds]

            for key in results.get('img_fields', ['img']):
                img = results[key]
                new_img = np.zeros_like(img)
                img_h, img_w = img.shape[:2]
                new_h = img_h - np.abs(random_shift_y)
                new_w = img_w - np.abs(random_shift_x)
                new_img[new_y:new_y + new_h, new_x:new_x + new_w] \
                    = img[ori_y:ori_y + new_h, ori_x:ori_x + new_w]
                results[key] = new_img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_shift_px={self.max_shift_px}, '
        return repr_str


@PIPELINES.register_module()
class Pad:
    """Pad the image & masks & segmentation map.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, masks=0, seg=255)):
        """ 已看過，添加padding
        Args:
            size: 固定的padding大小
            size_divisor: padding size的除數
            pad_to_square: 是否需要將圖像變成正方形
            pad_val: padding的數值，這裡會根據對不同東西進行padding會有不同的數值
        """
        # 將指定的size保存
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            # padding的值如果是float型態會跳出警告，現在只希望是int
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        # 保存資料
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            # 如果經過padding後希望要是正方形就會到這裡
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            # size與size_divisor只能有一個是None
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        # 已看過，對圖像進行padding
        # 獲取padding的值
        pad_val = self.pad_val.get('img', 0)
        # 遍歷所有的圖像
        for key in results.get('img_fields', ['img']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                # 使用size_divisor進行padding
                padded_img = mmcv.impad_to_multiple(
                    results[key], self.size_divisor, pad_val=pad_val)
            # 更新圖像
            results[key] = padded_img
        # 更新padding後的shape
        results['pad_shape'] = padded_img.shape
        # 保存一些參數
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        # 已看過，對mask進行padding
        # padding後的圖像大小
        pad_shape = results['pad_shape'][:2]
        # padding的值
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            # 透過pad進行padding
            results[key] = results[key].pad(pad_shape, pad_val=pad_val)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get('seg', 255)
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key], shape=results['pad_shape'][:2], pad_val=pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        # 已看過，對圖像進行padding

        # 對圖像進行padding
        self._pad_img(results)
        # 對masks進行padding
        self._pad_masks(results)
        # 對segmentation進行padding
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class Normalize:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        """ 已看過，對圖像進行均值標準化調整
        Args:
            mean: 均值
            std: 方差
            to_rgb: 是否要將圖像轉成RGB格式
        """
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        # 已看過，對圖像進行均值標準差調整
        for key in results.get('img_fields', ['img']):
            results[key] = mmcv.imnormalize(results[key], self.mean, self.std,
                                            self.to_rgb)
        # 記錄下調整方式
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class RandomCrop:
    """Random crop the image & bboxes & masks.

    The absolute `crop_size` is sampled based on `crop_type` and `image_size`,
    then the cropped results are generated.

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            height and width.
        crop_type (str, optional): one of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])]. Default "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Default False.
        recompute_bbox (bool, optional): Whether to re-compute the boxes based
            on cropped instance masks. Default False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and
          `gt_bboxes_ignore` corresponds to `gt_labels_ignore` and
          `gt_masks_ignore`.
        - If the crop does not contain any gt-bbox region and
          `allow_negative_crop` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 recompute_bbox=False,
                 bbox_clip_border=True):
        """ 已看過，隨機剪裁，會同時剪裁圖像標註匡以及mask
        Args:
            crop_size: 剪裁後的圖像大小或是高寬比
            crop_type: 剪裁的方式
            allow_negative_crop: 是否允許裁切後的圖像當中沒有任何標註匡
            recompute_bbox: 是否需要根據裁切後的圖像進行計算標註匡數量
            bbox_clip_border: 是否要將圖像外的標註匡裁切掉
        """

        # 檢查剪裁的方式是否合法
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            # 如果選擇的剪裁方式是絕對座標就會需要檢查是否皆大於0
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
        else:
            # 其他就會是比例，所以需要在[0, 1]之間
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        # 將傳入的值進行保存
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox
        # The key correspondence from bboxes to labels and masks.
        # 建立一些對應關係
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        # 已看過，對圖像進行裁切
        # results = 圖像詳細資料
        # crop_size = 最後圖像要裁切到的大小
        # allow_negative_crop = 是否允許裁切後圖像不包含任何標註匡
        assert crop_size[0] > 0 and crop_size[1] > 0
        # 遍歷所有圖像
        for key in results.get('img_fields', ['img']):
            # 獲取圖像資料
            img = results[key]
            # margin_h, margin_w = 需要裁掉的大小
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            # 獲取高寬需要偏移多少，加一是因為裁切時右邊是開
            offset_h = np.random.randint(0, margin_h + 1)
            offset_w = np.random.randint(0, margin_w + 1)
            # 從crop_y1到crop_y2範圍內的圖像是我們要的
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            # 從crop_x1到crop_x2範圍內的圖像是我們要的
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # crop the image，根據需要的範圍對圖像進行剪裁
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            # 獲取新圖像大小
            img_shape = img.shape
            # 更新圖像
            results[key] = img
        # 更新圖像大小
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        # 剪裁標註匡部分
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            # 獲取偏移的部分
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            # 對標註匡進行偏移，也就是扣掉偏移量就會對應到剪裁後的原圖上面
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                # 將超出圖像的部分變成圖像邊界上面
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            # 過濾出合法的標註匡，右邊界要大於左邊界同時下邊界要大於上邊界
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                # 如果不允許圖像內沒有半個標註匡時，遇到沒有半個合法的匡就直接回傳None
                return None
            # 將不合法的標註匡去除
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore，找尋到對應的資料像是gt_box會配上一個label
            label_key = self.bbox2label.get(key)
            if label_key in results:
                # 將分類資料也進行過濾
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # 將結果回傳
        return results

    def _get_crop_size(self, image_size):
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (tuple): (h, w).

        Returns:
            crop_size (tuple): (crop_h, crop_w) in absolute pixels.
        """
        # 已看過，生成隨機剪裁大小
        # 獲取圖像高寬
        h, w = image_size
        if self.crop_type == 'absolute':
            # 如果剪裁的模式是absolute就直接返回要剪裁的大小與當前圖像的大小取小的
            return min(self.crop_size[0], h), min(self.crop_size[1], w)
        elif self.crop_type == 'absolute_range':
            # 如果剪裁的模式是absolute_range就會到這裡
            # 檢查裁切大小是否合法
            assert self.crop_size[0] <= self.crop_size[1]
            # 隨機選取裁切後的高寬，範圍會是[crop_size[0], crop_size[1]]之間不過不能大於原始圖像
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            # 回傳
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_h, crop_w = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        elif self.crop_type == 'relative_range':
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        # 已看過，進行隨機剪裁
        # 獲取當前圖像高寬
        image_size = results['img'].shape[:2]
        # 獲取隨機剪裁大小，crop_size會是tuple(new_h, new_w)
        crop_size = self._get_crop_size(image_size)
        # 對圖像進行隨機剪裁
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class SegRescale:
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
        backend (str): Image rescale backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
    """

    def __init__(self, scale_factor=1, backend='cv2'):
        self.scale_factor = scale_factor
        self.backend = backend

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """

        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key],
                    self.scale_factor,
                    interpolation='nearest',
                    backend=self.backend)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@PIPELINES.register_module()
class PhotoMetricDistortion:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        img = img.astype(np.float32)
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class Expand:
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 to_rgb=True,
                 ratio_range=(1, 4),
                 seg_ignore_label=None,
                 prob=0.5):
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob

    def __call__(self, results):
        """Call function to expand images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images, bounding boxes expanded
        """

        if random.uniform(0, 1) > self.prob:
            return results

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        # speedup expand when meets large image
        if np.all(self.mean == self.mean[0]):
            expand_img = np.empty((int(h * ratio), int(w * ratio), c),
                                  img.dtype)
            expand_img.fill(self.mean[0])
        else:
            expand_img = np.full((int(h * ratio), int(w * ratio), c),
                                 self.mean,
                                 dtype=img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img

        results['img'] = expand_img
        # expand bboxes
        for key in results.get('bbox_fields', []):
            results[key] = results[key] + np.tile(
                (left, top), 2).astype(results[key].dtype)

        # expand masks
        for key in results.get('mask_fields', []):
            results[key] = results[key].expand(
                int(h * ratio), int(w * ratio), top, left)

        # expand segs
        for key in results.get('seg_fields', []):
            gt_seg = results[key]
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label,
                                    dtype=gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results[key] = expand_gt_seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, to_rgb={self.to_rgb}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label})'
        return repr_str


@PIPELINES.register_module()
class MinIoURandomCrop:
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        The keys for bboxes, labels and masks should be paired. That is, \
        `gt_bboxes` corresponds to `gt_labels` and `gt_masks`, and \
        `gt_bboxes_ignore` to `gt_labels_ignore` and `gt_masks_ignore`.
    """

    def __init__(self,
                 min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                 min_crop_size=0.3,
                 bbox_clip_border=True):
        # 1: return ori img
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size
        self.bbox_clip_border = bbox_clip_border
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }

    def __call__(self, results):
        """Call function to crop images and bounding boxes with minimum IoU
        constraint.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images and bounding boxes cropped, \
                'img_shape' key is updated.
        """

        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        boxes = [results[key] for key in results['bbox_fields']]
        boxes = np.concatenate(boxes, 0)
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) *
                                (center[:, 1] > patch[1]) *
                                (center[:, 0] < patch[2]) *
                                (center[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    for key in results.get('bbox_fields', []):
                        boxes = results[key].copy()
                        mask = is_center_of_bboxes_in_patch(boxes, patch)
                        boxes = boxes[mask]
                        if self.bbox_clip_border:
                            boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                            boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                        boxes -= np.tile(patch[:2], 2)

                        results[key] = boxes
                        # labels
                        label_key = self.bbox2label.get(key)
                        if label_key in results:
                            results[label_key] = results[label_key][mask]

                        # mask fields
                        mask_key = self.bbox2mask.get(key)
                        if mask_key in results:
                            results[mask_key] = results[mask_key][
                                mask.nonzero()[0]].crop(patch)
                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape

                # seg fields
                for key in results.get('seg_fields', []):
                    results[key] = results[key][patch[1]:patch[3],
                                                patch[0]:patch[2]]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_ious={self.min_ious}, '
        repr_str += f'min_crop_size={self.min_crop_size}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class Corrupt:
    """Corruption augmentation.

    Corruption transforms implemented based on
    `imagecorruptions <https://github.com/bethgelab/imagecorruptions>`_.

    Args:
        corruption (str): Corruption name.
        severity (int, optional): The severity of corruption. Default: 1.
    """

    def __init__(self, corruption, severity=1):
        self.corruption = corruption
        self.severity = severity

    def __call__(self, results):
        """Call function to corrupt image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images corrupted.
        """

        if corrupt is None:
            raise RuntimeError('imagecorruptions is not installed')
        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        results['img'] = corrupt(
            results['img'].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(corruption={self.corruption}, '
        repr_str += f'severity={self.severity})'
        return repr_str


@PIPELINES.register_module()
class Albu:
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.

    An example of ``transforms`` is as followed:

    .. code-block::

        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict): Bbox_params for albumentation `Compose`
        keymap (dict): Contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug
    """

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        if bbox_params is not None:
            bbox_params = copy.deepcopy(bbox_params)
        if keymap is not None:
            keymap = copy.deepcopy(keymap)
        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper. Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """

        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        # TODO: add bbox_fields
        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        # TODO: Support mask structure in albu
        if 'masks' in results:
            if isinstance(results['masks'], PolygonMasks):
                raise NotImplementedError(
                    'Albu only supports BitMap masks now')
            ori_masks = results['masks']
            if albumentations.__version__ < '0.5':
                results['masks'] = results['masks'].masks
            else:
                results['masks'] = [mask for mask in results['masks'].masks]

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])
                    results['masks'] = ori_masks.__class__(
                        results['masks'], results['image'].shape[0],
                        results['image'].shape[1])

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])
            results['gt_labels'] = results['gt_labels'].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module()
class RandomCenterCropPad:
    """Random center crop and random around padding for CornerNet.

    This operation generates randomly cropped image from the original image and
    pads it simultaneously. Different from :class:`RandomCrop`, the output
    shape may not equal to ``crop_size`` strictly. We choose a random value
    from ``ratios`` and the output shape could be larger or smaller than
    ``crop_size``. The padding operation is also different from :class:`Pad`,
    here we use around padding instead of right-bottom padding.

    The relation between output image (padding image) and original image:

    .. code:: text

                        output image

               +----------------------------+
               |          padded area       |
        +------|----------------------------|----------+
        |      |         cropped area       |          |
        |      |         +---------------+  |          |
        |      |         |    .   center |  |          | original image
        |      |         |        range  |  |          |
        |      |         +---------------+  |          |
        +------|----------------------------|----------+
               |          padded area       |
               +----------------------------+

    There are 5 main areas in the figure:

    - output image: output image of this operation, also called padding
      image in following instruction.
    - original image: input image of this operation.
    - padded area: non-intersect area of output image and original image.
    - cropped area: the overlap of output image and original image.
    - center range: a smaller area where random center chosen from.
      center range is computed by ``border`` and original image's shape
      to avoid our random center is too close to original image's border.

    Also this operation act differently in train and test mode, the summary
    pipeline is listed below.

    Train pipeline:

    1. Choose a ``random_ratio`` from ``ratios``, the shape of padding image
       will be ``random_ratio * crop_size``.
    2. Choose a ``random_center`` in center range.
    3. Generate padding image with center matches the ``random_center``.
    4. Initialize the padding image with pixel value equals to ``mean``.
    5. Copy the cropped area to padding image.
    6. Refine annotations.

    Test pipeline:

    1. Compute output shape according to ``test_pad_mode``.
    2. Generate padding image with center matches the original image
       center.
    3. Initialize the padding image with pixel value equals to ``mean``.
    4. Copy the ``cropped area`` to padding image.

    Args:
        crop_size (tuple | None): expected size after crop, final size will
            computed according to ratio. Requires (h, w) in train mode, and
            None in test mode.
        ratios (tuple): random select a ratio from tuple and crop image to
            (crop_size[0] * ratio) * (crop_size[1] * ratio).
            Only available in train mode.
        border (int): max distance from center select area to image border.
            Only available in train mode.
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB.
        test_mode (bool): whether involve random variables in transform.
            In train mode, crop_size is fixed, center coords and ratio is
            random selected from predefined lists. In test mode, crop_size
            is image's original shape, center coords and ratio is fixed.
        test_pad_mode (tuple): padding method and padding shape value, only
            available in test mode. Default is using 'logical_or' with
            127 as padding shape value.

            - 'logical_or': final_shape = input_shape | padding_shape_value
            - 'size_divisor': final_shape = int(
              ceil(input_shape / padding_shape_value) * padding_shape_value)
        test_pad_add_pix (int): Extra padding pixel in test mode. Default 0.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.
    """

    def __init__(self,
                 crop_size=None,
                 ratios=(0.9, 1.0, 1.1),
                 border=128,
                 mean=None,
                 std=None,
                 to_rgb=None,
                 test_mode=False,
                 test_pad_mode=('logical_or', 127),
                 test_pad_add_pix=0,
                 bbox_clip_border=True):
        if test_mode:
            assert crop_size is None, 'crop_size must be None in test mode'
            assert ratios is None, 'ratios must be None in test mode'
            assert border is None, 'border must be None in test mode'
            assert isinstance(test_pad_mode, (list, tuple))
            assert test_pad_mode[0] in ['logical_or', 'size_divisor']
        else:
            assert isinstance(crop_size, (list, tuple))
            assert crop_size[0] > 0 and crop_size[1] > 0, (
                'crop_size must > 0 in train mode')
            assert isinstance(ratios, (list, tuple))
            assert test_pad_mode is None, (
                'test_pad_mode must be None in train mode')

        self.crop_size = crop_size
        self.ratios = ratios
        self.border = border
        # We do not set default value to mean, std and to_rgb because these
        # hyper-parameters are easy to forget but could affect the performance.
        # Please use the same setting as Normalize for performance assurance.
        assert mean is not None and std is not None and to_rgb is not None
        self.to_rgb = to_rgb
        self.input_mean = mean
        self.input_std = std
        if to_rgb:
            self.mean = mean[::-1]
            self.std = std[::-1]
        else:
            self.mean = mean
            self.std = std
        self.test_mode = test_mode
        self.test_pad_mode = test_pad_mode
        self.test_pad_add_pix = test_pad_add_pix
        self.bbox_clip_border = bbox_clip_border

    def _get_border(self, border, size):
        """Get final border for the target size.

        This function generates a ``final_border`` according to image's shape.
        The area between ``final_border`` and ``size - final_border`` is the
        ``center range``. We randomly choose center from the ``center range``
        to avoid our random center is too close to original image's border.
        Also ``center range`` should be larger than 0.

        Args:
            border (int): The initial border, default is 128.
            size (int): The width or height of original image.
        Returns:
            int: The final border.
        """
        k = 2 * border / size
        i = pow(2, np.ceil(np.log2(np.ceil(k))) + (k == int(k)))
        return border // i

    def _filter_boxes(self, patch, boxes):
        """Check whether the center of each box is in the patch.

        Args:
            patch (list[int]): The cropped area, [left, top, right, bottom].
            boxes (numpy array, (N x 4)): Ground truth boxes.

        Returns:
            mask (numpy array, (N,)): Each box is inside or outside the patch.
        """
        center = (boxes[:, :2] + boxes[:, 2:]) / 2
        mask = (center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) * (
            center[:, 0] < patch[2]) * (
                center[:, 1] < patch[3])
        return mask

    def _crop_image_and_paste(self, image, center, size):
        """Crop image with a given center and size, then paste the cropped
        image to a blank image with two centers align.

        This function is equivalent to generating a blank image with ``size``
        as its shape. Then cover it on the original image with two centers (
        the center of blank image and the random center of original image)
        aligned. The overlap area is paste from the original image and the
        outside area is filled with ``mean pixel``.

        Args:
            image (np array, H x W x C): Original image.
            center (list[int]): Target crop center coord.
            size (list[int]): Target crop size. [target_h, target_w]

        Returns:
            cropped_img (np array, target_h x target_w x C): Cropped image.
            border (np array, 4): The distance of four border of
                ``cropped_img`` to the original image area, [top, bottom,
                left, right]
            patch (list[int]): The cropped area, [left, top, right, bottom].
        """
        center_y, center_x = center
        target_h, target_w = size
        img_h, img_w, img_c = image.shape

        x0 = max(0, center_x - target_w // 2)
        x1 = min(center_x + target_w // 2, img_w)
        y0 = max(0, center_y - target_h // 2)
        y1 = min(center_y + target_h // 2, img_h)
        patch = np.array((int(x0), int(y0), int(x1), int(y1)))

        left, right = center_x - x0, x1 - center_x
        top, bottom = center_y - y0, y1 - center_y

        cropped_center_y, cropped_center_x = target_h // 2, target_w // 2
        cropped_img = np.zeros((target_h, target_w, img_c), dtype=image.dtype)
        for i in range(img_c):
            cropped_img[:, :, i] += self.mean[i]
        y_slice = slice(cropped_center_y - top, cropped_center_y + bottom)
        x_slice = slice(cropped_center_x - left, cropped_center_x + right)
        cropped_img[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

        border = np.array([
            cropped_center_y - top, cropped_center_y + bottom,
            cropped_center_x - left, cropped_center_x + right
        ],
                          dtype=np.float32)

        return cropped_img, border, patch

    def _train_aug(self, results):
        """Random crop and around padding the original image.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        h, w, c = img.shape
        boxes = results['gt_bboxes']
        while True:
            scale = random.choice(self.ratios)
            new_h = int(self.crop_size[0] * scale)
            new_w = int(self.crop_size[1] * scale)
            h_border = self._get_border(self.border, h)
            w_border = self._get_border(self.border, w)

            for i in range(50):
                center_x = random.randint(low=w_border, high=w - w_border)
                center_y = random.randint(low=h_border, high=h - h_border)

                cropped_img, border, patch = self._crop_image_and_paste(
                    img, [center_y, center_x], [new_h, new_w])

                mask = self._filter_boxes(patch, boxes)
                # if image do not have valid bbox, any crop patch is valid.
                if not mask.any() and len(boxes) > 0:
                    continue

                results['img'] = cropped_img
                results['img_shape'] = cropped_img.shape
                results['pad_shape'] = cropped_img.shape

                x0, y0, x1, y1 = patch

                left_w, top_h = center_x - x0, center_y - y0
                cropped_center_x, cropped_center_y = new_w // 2, new_h // 2

                # crop bboxes accordingly and clip to the image boundary
                for key in results.get('bbox_fields', []):
                    mask = self._filter_boxes(patch, results[key])
                    bboxes = results[key][mask]
                    bboxes[:, 0:4:2] += cropped_center_x - left_w - x0
                    bboxes[:, 1:4:2] += cropped_center_y - top_h - y0
                    if self.bbox_clip_border:
                        bboxes[:, 0:4:2] = np.clip(bboxes[:, 0:4:2], 0, new_w)
                        bboxes[:, 1:4:2] = np.clip(bboxes[:, 1:4:2], 0, new_h)
                    keep = (bboxes[:, 2] > bboxes[:, 0]) & (
                        bboxes[:, 3] > bboxes[:, 1])
                    bboxes = bboxes[keep]
                    results[key] = bboxes
                    if key in ['gt_bboxes']:
                        if 'gt_labels' in results:
                            labels = results['gt_labels'][mask]
                            labels = labels[keep]
                            results['gt_labels'] = labels
                        if 'gt_masks' in results:
                            raise NotImplementedError(
                                'RandomCenterCropPad only supports bbox.')

                # crop semantic seg
                for key in results.get('seg_fields', []):
                    raise NotImplementedError(
                        'RandomCenterCropPad only supports bbox.')
                return results

    def _test_aug(self, results):
        """Around padding the original image without cropping.

        The padding mode and value are from ``test_pad_mode``.

        Args:
            results (dict): Image infomations in the augment pipeline.

        Returns:
            results (dict): The updated dict.
        """
        img = results['img']
        h, w, c = img.shape
        results['img_shape'] = img.shape
        if self.test_pad_mode[0] in ['logical_or']:
            # self.test_pad_add_pix is only used for centernet
            target_h = (h | self.test_pad_mode[1]) + self.test_pad_add_pix
            target_w = (w | self.test_pad_mode[1]) + self.test_pad_add_pix
        elif self.test_pad_mode[0] in ['size_divisor']:
            divisor = self.test_pad_mode[1]
            target_h = int(np.ceil(h / divisor)) * divisor
            target_w = int(np.ceil(w / divisor)) * divisor
        else:
            raise NotImplementedError(
                'RandomCenterCropPad only support two testing pad mode:'
                'logical-or and size_divisor.')

        cropped_img, border, _ = self._crop_image_and_paste(
            img, [h // 2, w // 2], [target_h, target_w])
        results['img'] = cropped_img
        results['pad_shape'] = cropped_img.shape
        results['border'] = border
        return results

    def __call__(self, results):
        img = results['img']
        assert img.dtype == np.float32, (
            'RandomCenterCropPad needs the input image of dtype np.float32,'
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline')
        h, w, c = img.shape
        assert c == len(self.mean)
        if self.test_mode:
            return self._test_aug(results)
        else:
            return self._train_aug(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'ratios={self.ratios}, '
        repr_str += f'border={self.border}, '
        repr_str += f'mean={self.input_mean}, '
        repr_str += f'std={self.input_std}, '
        repr_str += f'to_rgb={self.to_rgb}, '
        repr_str += f'test_mode={self.test_mode}, '
        repr_str += f'test_pad_mode={self.test_pad_mode}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class CutOut:
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.

    Args:
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
    """

    def __init__(self,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0)):

        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    def __call__(self, results):
        """Call function to drop some regions of image."""
        h, w, c = results['img'].shape
        n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
        for _ in range(n_holes):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)
            index = np.random.randint(0, len(self.candidates))
            if not self.with_ratio:
                cutout_w, cutout_h = self.candidates[index]
            else:
                cutout_w = int(self.candidates[index][0] * w)
                cutout_h = int(self.candidates[index][1] * h)

            x2 = np.clip(x1 + cutout_w, 0, w)
            y2 = np.clip(y1 + cutout_h, 0, h)
            results['img'][y1:y2, x1:x2, :] = self.fill_in

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in})'
        return repr_str


@PIPELINES.register_module()
class Mosaic:
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |  pad      |
                |      +-----------+           |
                |      |           |           |
                |      |  image1   |--------+  |
                |      |           |        |  |
                |      |           | image2 |  |
     center_y   |----+-------------+-----------|
                |    |   cropped   |           |
                |pad |   image3    |  image4   |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Default to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Default to (0.5, 1.5).
        min_bbox_size (int | float): The minimum pixel for filtering
            invalid bboxes after the mosaic pipeline. Default to 0.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` is invalid. Default to True.
        pad_val (int): Pad value. Default to 114.
        prob (float): Probability of applying this transformation.
            Default to 1.0.
    """

    def __init__(self,
                 img_scale=(640, 640),
                 center_ratio_range=(0.5, 1.5),
                 min_bbox_size=0,
                 bbox_clip_border=True,
                 skip_filter=True,
                 pad_val=114,
                 prob=1.0):
        """ Mosaic圖像增強方式
        Args:
            img_scale: 最終圖像輸出大小
            center_ratio_range: 中心點範圍
            min_bbox_size: 經過mosaic後當標註匡小於多少像素會認定為無效匡
            bbox_clip_border: 是否需要對超出圖像的標註匡進行剪裁
            skip_filter: 是否需要跳過過濾器規則
            pad_val: padding的值
            prob: 使用mosaic的概率
        """
        # 檢查img_scale是否為tuple格式
        assert isinstance(img_scale, tuple)
        # 檢查觸發概率是否設定正確
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. '\
            f'got {prob}.'

        # 紀錄log資料
        log_img_scale(img_scale, skip_square=True)
        # 保存傳入資料
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter
        self.pad_val = pad_val
        self.prob = prob

    def __call__(self, results):
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """
        # 圖像增強Mosaic的call函數

        if random.uniform(0, 1) > self.prob:
            # 隨機決定是否進行mosaic增強，如果不需要進行增強就直接回傳results
            return results

        # 否則就進行mosaic圖像增強
        results = self._mosaic_transform(results)
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """

        # 隨機獲取dataset當中的圖像，這裡是要給mosaic使用，因為需要四張圖像進行混合，所以還需要挑出隨機三張圖像
        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
        # 將挑出的圖像index回傳
        return indexes

    def _mosaic_transform(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # 進行mosaic圖像增強

        # 這裡會需要其他三張圖像進行融合，所以需要獲取mix_results資訊，在mix_results當中會有其他三張圖像的資訊
        assert 'mix_results' in results
        # 保存四張圖像的標籤
        mosaic_labels = []
        # 保存四張圖像的標註匡
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:
            # 如果圖像是RGB三通道的就會到這裡
            # 構建一個高寬為指定圖像大小兩倍的空圖像，這裡會先用指定的顏色進行填充，這裡會構建的是三通道的圖像
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            # 如果圖像是Gray單通道的就會到這裡
            # 構建一個高寬為指定圖像大小兩倍的空圖像，這裡會先用指定的顏色進行填充，這裡會構建的是單通道的圖像
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        # 這裡會隨機選取中心點位置，會從指定的範圍內進行隨機選取並且縮放到絕對位置上
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        # 將指定的中心點位置做成tuple格式
        center_position = (center_x, center_y)

        # 在mosaic當中會將圖像分成四個部分(左上角, 右上角, 左下角, 右下角)
        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        # 遍歷四個地方
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                # 如果是左上角就會是當前主圖像
                results_patch = copy.deepcopy(results)
            else:
                # 其他位置就是額外挑出來的圖像
                results_patch = copy.deepcopy(results['mix_results'][i - 1])

            # 獲取圖像ndarray shape [height, width, channel]
            img_i = results_patch['img']
            # 獲取圖像高寬資料
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            # 獲取將較長的邊縮放到指定長度時需縮放的大小，這裡主要是為了進行高寬比例不變的縮放
            scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)
            # 使用resize將圖像縮放到指定的高寬
            img_i = mmcv.imresize(img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            # 獲取需要剪裁的大小，將當前圖像要放的位置以及中心點位置以及當前圖像高寬傳入
            # paste_coord = 當前圖像要貼在mosaic圖像的哪個位置中
            # crop_coord = 要使用當前圖像的哪個位置
            paste_coord, crop_coord = self._mosaic_combine(loc, center_position, img_i.shape[:2][::-1])
            # 將座標解出來
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            # 在mosaic上的指定位置貼上指定圖像
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            # 獲取當前圖像的標註匡訊息以及標註訊息
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                # 如果該圖像有標註訊息就會到這裡
                # 獲取經過剪裁後圖像在高寬上偏移多少距離
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                # 將原始標註資訊經過與圖像相同縮放後進行偏移
                gt_bboxes_i[:, 0::2] = scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = scale_ratio_i * gt_bboxes_i[:, 1::2] + padh

            # 將標註匡資料以及標註類別保存，這裡先暫時不處理標註匡跑到圖外的問題
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)

        if len(mosaic_labels) > 0:
            # 如果進融合結果有標註資訊就會到這裡
            # 將標註資訊進行拼接，這裡會在第一個維度進行拼接
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)

            if self.bbox_clip_border:
                # 將標註匡資訊限縮在圖像當中，會將超出去的部分限制到圖像邊緣上
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0, 2 * self.img_scale[1])
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0, 2 * self.img_scale[0])

            if not self.skip_filter:
                # 如果沒有要跳過過濾器就會到這裡，這裡會過濾掉標註匡過小的部分
                mosaic_bboxes, mosaic_labels = self._filter_box_candidates(mosaic_bboxes, mosaic_labels)

        # remove outside bboxes
        # 獲取哪些標註匡在圖像裡面
        inside_inds = find_inside_bboxes(mosaic_bboxes, 2 * self.img_scale[0], 2 * self.img_scale[1])
        # 將在圖像當中的標註匡過濾出來，更新資訊
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]

        # 將融合後的圖像保存到img當中
        results['img'] = mosaic_img
        # 更新圖像大小
        results['img_shape'] = mosaic_img.shape
        # 更新標註匡訊息
        results['gt_bboxes'] = mosaic_bboxes
        # 更新標註對象類別
        results['gt_labels'] = mosaic_labels

        # 回傳更新後的results資訊
        return results

    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        # 計算圖像放到mosaic指定位置時的剪裁大小
        # loc = 當前圖像要放的位置
        # center_position_xy = 拼接圖像的中心點座標
        # img_shape_wh = 圖像的寬高

        # 檢查loc需要使以下幾種，其他就會直接報錯
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # 如果是要放在左上角就會到這裡
            # index0 to top left part of image
            # 計算當前圖像要從mosaic圖像的哪個位置貼到哪個位置
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            # 獲取我們需要圖像的哪個部分，這裡會選出的是圖像當中哪個部分是我們需要的
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # 如果是要放在右上角就會到這裡
            # index1 to top right part of image
            # 計算當前圖像要從mosaic圖像的哪個位置貼到哪個位置
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            # 獲取我們需要圖像的哪個部分，這裡會選出的是圖像當中哪個部分是我們需要的
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # 如果是要放在左下角就會到這裡
            # index2 to bottom left part of image
            # 計算當前圖像要從mosaic圖像的哪個位置貼到哪個位置
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            # 獲取我們需要圖像的哪個部分，這裡會選出的是圖像當中哪個部分是我們需要的
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # 如果是要放在右下角就會到這裡
            # index3 to bottom right part of image
            # 計算當前圖像要從mosaic圖像的哪個位置貼到哪個位置
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            # 獲取我們需要圖像的哪個部分，這裡會選出的是圖像當中哪個部分是我們需要的
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        # 回傳結果
        return paste_coord, crop_coord

    def _filter_box_candidates(self, bboxes, labels):
        """Filter out bboxes too small after Mosaic."""
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (bbox_w > self.min_bbox_size) & \
                     (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'skip_filter={self.skip_filter})'
        return repr_str


@PIPELINES.register_module()
class MixUp:
    """MixUp data augmentation.

    .. code:: text

                         mixup transform
                +------------------------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                |---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      |-----------------+     |
                |             pad              |
                +------------------------------+

     The mixup transform steps are as follows:

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (height, width). Default: (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Default: (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Default: 0.5.
        pad_val (int): Pad value. Default: 114.
        max_iters (int): The maximum number of iterations. If the number of
            iterations is greater than `max_iters`, but gt_bbox is still
            empty, then the iteration is terminated. Default: 15.
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 5.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed. Default: 20.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """

    def __init__(self,
                 img_scale=(640, 640),
                 ratio_range=(0.5, 1.5),
                 flip_ratio=0.5,
                 pad_val=114,
                 max_iters=15,
                 min_bbox_size=5,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20,
                 bbox_clip_border=True,
                 skip_filter=True):
        """ 在原始圖像的左上角再隨機添加上一張圖像
        Args:
            img_scale: 輸出圖像的大小
            ratio_range: 對於要插入的圖像的縮放比率
            flip_ratio: 對於要插入的圖像的翻轉概率
            pad_val: 填充的值
            max_iters: 最多可以重新獲取多少次圖像
            min_bbox_size: 最小標註匡的面積
            min_area_ratio: 經過圖像變換後，標註匡需要是原始標註匡最小多少倍
            max_aspect_ratio: 最大高寬比
            bbox_clip_border: 是否需要將超過圖像的標註匡進行限縮
            skip_filter: 是否要跳過，過濾過小標註匡
        """
        # 檢查img_scale要是tuple格式
        assert isinstance(img_scale, tuple)
        # 記錄到log當中
        log_img_scale(img_scale, skip_square=True)
        # 將傳入資料進行保存
        self.dynamic_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.max_iters = max_iters
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter

    def __call__(self, results):
        """Call function to make a mixup of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mixup transformed.
        """
        # 將獲取的圖像添加到左上角當中

        # 透過_mixup_transform進行融合
        results = self._mixup_transform(results)
        # 將結果回傳
        return results

    def get_indexes(self, dataset):
        """Call function to collect indexes.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.

        Returns:
            list: indexes.
        """
        # 隨機獲取要額外添加的圖像
        index = 0

        for i in range(self.max_iters):
            # 從資料集當中隨機獲取一張圖像
            index = random.randint(0, len(dataset))
            # 將圖像經過圖像處理流
            gt_bboxes_i = dataset.get_ann_info(index)['bboxes']
            if len(gt_bboxes_i) != 0:
                # 如果當中有獲取到標註匡資訊就表示成功，跳出迴圈
                break

        # 回傳選中的圖像index
        return index

    def _mixup_transform(self, results):
        """MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        # 進行圖像增強融合

        # 檢查results當中需要有mix_results才可以進行融合
        assert 'mix_results' in results
        # 這裡目前只能將一張圖像添加到左上角當中
        assert len(results['mix_results']) == 1, 'MixUp only support 2 images now !'

        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            # 如果獲取的圖像沒有任何標註匡就直接回傳results不進行融合
            # empty bbox
            return results

        # 獲取要融合進的圖像資訊
        retrieve_results = results['mix_results'][0]
        # 獲取圖像
        retrieve_img = retrieve_results['img']

        # 獲取要縮放的倍率
        jit_factor = random.uniform(*self.ratio_range)
        # 獲取是否需要對圖像進行旋轉
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            # 如果插入圖像是彩色圖像就會到這裡，構建3通道且全為填充值的ndarray
            out_img = np.ones(
                (self.dynamic_scale[0], self.dynamic_scale[1], 3),
                dtype=retrieve_img.dtype) * self.pad_val
        else:
            # 如果是灰度圖像就會到這裡
            out_img = np.ones(self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val

        # 1. keep_ratio resize
        # 獲取原圖需要進行多少倍的縮放才可以讓最長邊貼合到指定長度
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0],
                          self.dynamic_scale[1] / retrieve_img.shape[1])
        # 透過resize進行圖像縮放，這裡進行的會是保持高寬不變的縮放
        retrieve_img = mmcv.imresize(
            retrieve_img, (int(retrieve_img.shape[1] * scale_ratio),
                           int(retrieve_img.shape[0] * scale_ratio)))

        # 2. paste
        # 將縮放後的圖像貼到剛生成的空圖上，這裡是沿著左上角進行貼合
        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img

        # 3. scale jit
        # 最後一步進行縮放
        scale_ratio *= jit_factor
        # 透過resize將圖像縮放到指定大小
        out_img = mmcv.imresize(out_img, (int(out_img.shape[1] * jit_factor), int(out_img.shape[0] * jit_factor)))

        # 4. flip
        if is_filp:
            # 如果有需要進行翻轉就會沿著寬度方向進行左右翻轉
            out_img = out_img[:, ::-1, :]

        # 5. random crop
        # 獲取原始圖像
        ori_img = results['img']
        # 獲取當前要貼上的圖像高寬資訊
        origin_h, origin_w = out_img.shape[:2]
        # 獲取原始圖像的高寬資訊
        target_h, target_w = ori_img.shape[:2]
        # 構建一個空圖，這裡的高寬會是兩個圖像取大的，也就是空圖會大於等於原始圖像以及當前要貼上的圖像
        padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), 3)).astype(np.uint8)
        # 將要添加上去的圖像放到padded_img上
        padded_img[:origin_h, :origin_w] = out_img

        # 先將x與y方向的偏移量設定成0
        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            # 如果padded_img的高度比原始圖像高就會到這裡
            # 隨機獲取y方向上的偏移量，因為padded_img高度較高，所以會有可以移動的空間
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            # # 如果padded_img的寬度比原始圖像寬就會到這裡
            # 隨機獲取x方向上的偏移量，因為padded_img寬度較寬，所以會有可以移動的空間
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        # 這裡會將padded_img的高寬限縮到與原始圖像大小相同
        padded_cropped_img = padded_img[y_offset:y_offset + target_h, x_offset:x_offset + target_w]

        # 6. adjust bbox
        # 調整標註匡資訊
        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        # 先將標註匡也進行與圖像相同的縮放
        retrieve_gt_bboxes[:, 0::2] = retrieve_gt_bboxes[:, 0::2] * scale_ratio
        retrieve_gt_bboxes[:, 1::2] = retrieve_gt_bboxes[:, 1::2] * scale_ratio
        if self.bbox_clip_border:
            # 將超出圖像的標註匡進行限縮
            retrieve_gt_bboxes[:, 0::2] = np.clip(retrieve_gt_bboxes[:, 0::2], 0, origin_w)
            retrieve_gt_bboxes[:, 1::2] = np.clip(retrieve_gt_bboxes[:, 1::2], 0, origin_h)

        if is_filp:
            # 如果圖像有進行翻轉則標註匡也需要進行翻轉
            retrieve_gt_bboxes[:, 0::2] = (origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1])

        # 7. filter
        # 先拷貝一份標註匡資訊
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        # 將要貼上的標註匡進行調整，根據偏移量進行調整
        cp_retrieve_gt_bboxes[:, 0::2] = cp_retrieve_gt_bboxes[:, 0::2] - x_offset
        cp_retrieve_gt_bboxes[:, 1::2] = cp_retrieve_gt_bboxes[:, 1::2] - y_offset
        if self.bbox_clip_border:
            # 如果需要將超出去的部分限縮就會到這裡
            cp_retrieve_gt_bboxes[:, 0::2] = np.clip(cp_retrieve_gt_bboxes[:, 0::2], 0, target_w)
            cp_retrieve_gt_bboxes[:, 1::2] = np.clip(cp_retrieve_gt_bboxes[:, 1::2], 0, target_h)

        # 8. mix up
        # 將原始圖像轉成float32格式
        ori_img = ori_img.astype(np.float32)
        # 進行圖像融合，這裡就直接進行圖像相加
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        # 獲取疊加上去的圖像標註分類類別
        retrieve_gt_labels = retrieve_results['gt_labels']
        if not self.skip_filter:
            # 如果需要將較小的標註匡過濾就會到這裡
            keep_list = self._filter_box_candidates(retrieve_gt_bboxes.T, cp_retrieve_gt_bboxes.T)

            # 進行過濾
            retrieve_gt_labels = retrieve_gt_labels[keep_list]
            cp_retrieve_gt_bboxes = cp_retrieve_gt_bboxes[keep_list]

        # 將標註匡與標註分類類別進行融合
        mixup_gt_bboxes = np.concatenate((results['gt_bboxes'], cp_retrieve_gt_bboxes), axis=0)
        mixup_gt_labels = np.concatenate((results['gt_labels'], retrieve_gt_labels), axis=0)

        # remove outside bbox
        # 獲取哪些標註匡有在圖像內
        inside_inds = find_inside_bboxes(mixup_gt_bboxes, target_h, target_w)
        mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
        mixup_gt_labels = mixup_gt_labels[inside_inds]

        # 更新圖像資訊
        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_labels'] = mixup_gt_labels

        # 回傳更新後的圖像
        return results

    def _filter_box_candidates(self, bbox1, bbox2):
        """Compute candidate boxes which include following 5 things:

        bbox1 before augment, bbox2 after augment, min_bbox_size (pixels),
        min_area_ratio, max_aspect_ratio.
        """

        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return ((w2 > self.min_bbox_size)
                & (h2 > self.min_bbox_size)
                & (w2 * h2 / (w1 * h1 + 1e-16) > self.min_area_ratio)
                & (ar < self.max_aspect_ratio))

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'dynamic_scale={self.dynamic_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'flip_ratio={self.flip_ratio}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'max_iters={self.max_iters}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_aspect_ratio={self.max_aspect_ratio}, '
        repr_str += f'skip_filter={self.skip_filter})'
        return repr_str


@PIPELINES.register_module()
class RandomAffine:
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Default: 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Default: 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Default: (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Default: 2.
        border (tuple[int]): Distance from height and width sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Default: (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Default: (114, 114, 114).
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Default: 2.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Default: 0.2.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        skip_filter (bool): Whether to skip filtering rules. If it
            is True, the filter rule will not be applied, and the
            `min_bbox_size` and `min_area_ratio` and `max_aspect_ratio`
            is invalid. Default to True.
    """

    def __init__(self,
                 max_rotate_degree=10.0,
                 max_translate_ratio=0.1,
                 scaling_ratio_range=(0.5, 1.5),
                 max_shear_degree=2.0,
                 border=(0, 0),
                 border_val=(114, 114, 114),
                 min_bbox_size=2,
                 min_area_ratio=0.2,
                 max_aspect_ratio=20,
                 bbox_clip_border=True,
                 skip_filter=True):
        """ 對圖像進行隨機仿射變換
        Args:
            max_rotate_degree: 最大旋轉角度
            max_translate_ratio: 最大平移比例
            scaling_ratio_range: 圖像縮放比例
            max_shear_degree: 最大剪切變換度
            border: 與輸入圖像的高度和寬度邊的距離以調整輸出形狀
            border_val: 邊緣填充值
            min_bbox_size: 最小的標註匡大小，如果小於該值就會拋棄
            min_area_ratio: 最小面積比例，這裡是與原圖進行大小計算比例
            max_aspect_ratio: 最大高寬比
            bbox_clip_border: 是否需要將超過圖像的標註匡進行剪裁
            skip_filter: 是否需要跳過過濾
        """
        # 檢查比例是否正確
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        # 將傳入的值進行保存
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter

    def __call__(self, results):
        # 對圖像進行隨機仿射變換

        # 從results當中獲取圖像資訊
        img = results['img']
        # 獲取輸出時的圖像大小，這裡會是原始圖像大小加上border，這裡如果border是正數就會填充否則就會裁減
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2

        # Rotation
        # 獲取隨機選轉的角度，這裡會從指定的角度範圍當中隨機選取
        rotation_degree = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
        # 透過_get_rotation_matrix獲取旋轉使用的矩陣
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        # 獲取隨機縮放比例，這裡會從指定的縮放範圍挑取一個縮放大小值
        scaling_ratio = random.uniform(self.scaling_ratio_range[0], self.scaling_ratio_range[1])
        # 透過_get_scaling_matrix獲取縮放使用的矩陣
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        # 獲取裁切資訊
        # 分別獲取對於x與y的裁切度
        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        # 獲取對應的矩陣
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        # 獲取變換矩陣
        trans_x = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        # 將所有變換矩陣透過矩陣乘法進行融合
        warp_matrix = (translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)

        # 透過cv2進行對應變換
        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape

        # 以下要對標註匡進行仿射變換
        for key in results.get('bbox_fields', []):
            # 獲取標註匡資料
            bboxes = results[key]
            # 獲取總共有多少個標註匡
            num_bboxes = len(bboxes)
            if num_bboxes:
                # 如果有標註匡資料就會到這裡
                # homogeneous coordinates
                # 這裡因為經過仿射變換後可能原先的[xmin, ymin, xmax, ymax]不會是原先轉換過去的
                # 也就是原先的(xmin, ymax)可能變成新的標註匡的(xmin, ymin)所以四個角我們都需要進行搜集，這裡將x與y分開
                xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
                ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                # 構建一個全為1的ndarray且shape與xs相同
                ones = np.ones_like(xs)
                # 將x與y與ones進行堆疊，shape [3, num_bboxes * 4]
                points = np.vstack([xs, ys, ones])

                # 進行仿射變換
                warp_points = warp_matrix @ points
                # 這裡不會發生任何變化，因為warp_points[2]是1
                warp_points = warp_points[:2] / warp_points[2]
                # 將shape進行調整，shape [num_bboxes, 4]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)

                # 獲取新的[xmin, ymin, xmax, ymax]
                warp_bboxes = np.vstack((xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T

                if self.bbox_clip_border:
                    # 如果有需要將標註匡限縮到圖像當中就會到這裡
                    warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
                    warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)

                # remove outside bbox
                # 獲取哪些標註匡有在圖像當中
                valid_index = find_inside_bboxes(warp_bboxes, height, width)
                if not self.skip_filter:
                    # filter bboxes
                    # 如果有需要將面積過小的標註匡過濾掉就會到這裡
                    filter_index = self.filter_gt_bboxes(bboxes * scaling_ratio, warp_bboxes)
                    valid_index = valid_index & filter_index

                # 過濾出標註匡資料
                results[key] = warp_bboxes[valid_index]
                if key in ['gt_bboxes']:
                    if 'gt_labels' in results:
                        # 將標註匡對應上的分類類別進行過濾
                        results['gt_labels'] = results['gt_labels'][valid_index]

                if 'gt_masks' in results:
                    # 這裡不支持masks資訊進行變換
                    raise NotImplementedError('RandomAffine only supports bbox.')
        # 回傳更新後的results資訊
        return results

    def filter_gt_bboxes(self, origin_bboxes, wrapped_bboxes):
        origin_w = origin_bboxes[:, 2] - origin_bboxes[:, 0]
        origin_h = origin_bboxes[:, 3] - origin_bboxes[:, 1]
        wrapped_w = wrapped_bboxes[:, 2] - wrapped_bboxes[:, 0]
        wrapped_h = wrapped_bboxes[:, 3] - wrapped_bboxes[:, 1]
        aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16),
                                  wrapped_h / (wrapped_w + 1e-16))

        wh_valid_idx = (wrapped_w > self.min_bbox_size) & \
                       (wrapped_h > self.min_bbox_size)
        area_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h +
                                                  1e-16) > self.min_area_ratio
        aspect_ratio_valid_idx = aspect_ratio < self.max_aspect_ratio
        return wh_valid_idx & area_valid_idx & aspect_ratio_valid_idx

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio={self.scaling_ratio_range}, '
        repr_str += f'max_shear_degree={self.max_shear_degree}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        repr_str += f'min_bbox_size={self.min_bbox_size}, '
        repr_str += f'min_area_ratio={self.min_area_ratio}, '
        repr_str += f'max_aspect_ratio={self.max_aspect_ratio}, '
        repr_str += f'skip_filter={self.skip_filter})'
        return repr_str

    @staticmethod
    def _get_rotation_matrix(rotate_degrees):
        # 獲取對圖像進行旋轉需要的變換矩陣
        # 將角度換成度
        radian = math.radians(rotate_degrees)
        # 構建轉換矩陣，這裡會生成的是ndarray
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        # 回傳變換矩陣
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        # 獲取縮放矩陣
        # 這裡會在高寬上面進行縮放，不會在channel發生變化
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        # 回傳縮放矩陣
        return scaling_matrix

    @staticmethod
    def _get_share_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees, y_shear_degrees):
        # 根據傳入的裁切度獲取對應的變換矩陣
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
        # 將變換矩陣回傳
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x, y):
        # 獲取變換矩陣
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix


@PIPELINES.register_module()
class YOLOXHSVRandomAug:
    """Apply HSV augmentation to image sequentially. It is referenced from
    https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L21.

    Args:
        hue_delta (int): delta of hue. Default: 5.
        saturation_delta (int): delta of saturation. Default: 30.
        value_delta (int): delat of value. Default: 30.
    """

    def __init__(self, hue_delta=5, saturation_delta=30, value_delta=30):
        """ 對圖像進行HSV轉換
        """
        # 將傳入的值進行保存
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def __call__(self, results):
        # 進行圖像HSV轉換
        # 獲取當前圖像
        img = results['img']
        # 獲取hsv變換矩陣
        hsv_gains = np.random.uniform(-1, 1, 3) * [self.hue_delta, self.saturation_delta, self.value_delta]
        # random selection of h, s, v
        # 隨機獲取增益
        hsv_gains *= np.random.randint(0, 2, 3)
        # prevent overflow
        hsv_gains = hsv_gains.astype(np.int16)
        # 將原先圖像的BGR通道轉成HSV通道
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

        # 進行一系列變換
        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        # 將圖像從HSV轉回BGR通道格式
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)

        # 保存轉換後的圖像
        results['img'] = img
        # 回傳結過
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str


@PIPELINES.register_module()
class CopyPaste:
    """Simple Copy-Paste is a Strong Data Augmentation Method for Instance
    Segmentation The simple copy-paste transform steps are as follows:

    1. The destination image is already resized with aspect ratio kept,
       cropped and padded.
    2. Randomly select a source image, which is also already resized
       with aspect ratio kept, cropped and padded in a similar way
       as the destination image.
    3. Randomly select some objects from the source image.
    4. Paste these source objects to the destination image directly,
       due to the source and destination image have the same size.
    5. Update object masks of the destination image, for some origin objects
       may be occluded.
    6. Generate bboxes from the updated destination masks and
       filter some objects which are totally occluded, and adjust bboxes
       which are partly occluded.
    7. Append selected source bboxes, masks, and labels.

    Args:
        max_num_pasted (int): The maximum number of pasted objects.
            Default: 100.
        bbox_occluded_thr (int): The threshold of occluded bbox.
            Default: 10.
        mask_occluded_thr (int): The threshold of occluded mask.
            Default: 300.
        selected (bool): Whether select objects or not. If select is False,
            all objects of the source image will be pasted to the
            destination image.
            Default: True.
    """

    def __init__(
        self,
        max_num_pasted=100,
        bbox_occluded_thr=10,
        mask_occluded_thr=300,
        selected=True,
    ):
        self.max_num_pasted = max_num_pasted
        self.bbox_occluded_thr = bbox_occluded_thr
        self.mask_occluded_thr = mask_occluded_thr
        self.selected = selected

    def get_indexes(self, dataset):
        """Call function to collect indexes.s.

        Args:
            dataset (:obj:`MultiImageMixDataset`): The dataset.
        Returns:
            list: Indexes.
        """
        return random.randint(0, len(dataset))

    def __call__(self, results):
        """Call function to make a copy-paste of image.

        Args:
            results (dict): Result dict.
        Returns:
            dict: Result dict with copy-paste transformed.
        """

        assert 'mix_results' in results
        num_images = len(results['mix_results'])
        assert num_images == 1, \
            f'CopyPaste only supports processing 2 images, got {num_images}'
        if self.selected:
            selected_results = self._select_object(results['mix_results'][0])
        else:
            selected_results = results['mix_results'][0]
        return self._copy_paste(results, selected_results)

    def _select_object(self, results):
        """Select some objects from the source results."""
        bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        masks = results['gt_masks']
        max_num_pasted = min(bboxes.shape[0] + 1, self.max_num_pasted)
        num_pasted = np.random.randint(0, max_num_pasted)
        selected_inds = np.random.choice(
            bboxes.shape[0], size=num_pasted, replace=False)

        selected_bboxes = bboxes[selected_inds]
        selected_labels = labels[selected_inds]
        selected_masks = masks[selected_inds]

        results['gt_bboxes'] = selected_bboxes
        results['gt_labels'] = selected_labels
        results['gt_masks'] = selected_masks
        return results

    def _copy_paste(self, dst_results, src_results):
        """CopyPaste transform function.

        Args:
            dst_results (dict): Result dict of the destination image.
            src_results (dict): Result dict of the source image.
        Returns:
            dict: Updated result dict.
        """
        dst_img = dst_results['img']
        dst_bboxes = dst_results['gt_bboxes']
        dst_labels = dst_results['gt_labels']
        dst_masks = dst_results['gt_masks']

        src_img = src_results['img']
        src_bboxes = src_results['gt_bboxes']
        src_labels = src_results['gt_labels']
        src_masks = src_results['gt_masks']

        if len(src_bboxes) == 0:
            return dst_results

        # update masks and generate bboxes from updated masks
        composed_mask = np.where(np.any(src_masks.masks, axis=0), 1, 0)
        updated_dst_masks = self.get_updated_masks(dst_masks, composed_mask)
        updated_dst_bboxes = updated_dst_masks.get_bboxes()
        assert len(updated_dst_bboxes) == len(updated_dst_masks)

        # filter totally occluded objects
        bboxes_inds = np.all(
            np.abs(
                (updated_dst_bboxes - dst_bboxes)) <= self.bbox_occluded_thr,
            axis=-1)
        masks_inds = updated_dst_masks.masks.sum(
            axis=(1, 2)) > self.mask_occluded_thr
        valid_inds = bboxes_inds | masks_inds

        # Paste source objects to destination image directly
        img = dst_img * (1 - composed_mask[..., np.newaxis]
                         ) + src_img * composed_mask[..., np.newaxis]
        bboxes = np.concatenate([updated_dst_bboxes[valid_inds], src_bboxes])
        labels = np.concatenate([dst_labels[valid_inds], src_labels])
        masks = np.concatenate(
            [updated_dst_masks.masks[valid_inds], src_masks.masks])

        dst_results['img'] = img
        dst_results['gt_bboxes'] = bboxes
        dst_results['gt_labels'] = labels
        dst_results['gt_masks'] = BitmapMasks(masks, masks.shape[1],
                                              masks.shape[2])

        return dst_results

    def get_updated_masks(self, masks, composed_mask):
        assert masks.masks.shape[-2:] == composed_mask.shape[-2:], \
            'Cannot compare two arrays of different size'
        masks.masks = np.where(composed_mask, 0, masks.masks)
        return masks

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'max_num_pasted={self.max_num_pasted}, '
        repr_str += f'bbox_occluded_thr={self.bbox_occluded_thr}, '
        repr_str += f'mask_occluded_thr={self.mask_occluded_thr}, '
        repr_str += f'selected={self.selected}, '
        return repr_str
