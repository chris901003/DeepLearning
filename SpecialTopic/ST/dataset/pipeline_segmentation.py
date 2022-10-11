import os
from numpy import random
import cv2
import numpy as np
from SpecialTopic.ST.dataset.utils import imrescale, imresize, imflip, imnormalize, impad, impad_to_multiple


# 本區部分全部由MMSegmentation模塊提供
class LoadImageFormFile:
    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        image_path = results.get('image_path', None)
        assert image_path is not None, '需提供image_path資訊'
        assert os.path.exists(image_path), f'圖像檔案{image_path}不存在'
        img = cv2.imread(image_path)
        if self.to_float32:
            img = img.astype(np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scaler_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float),
            to_rgb=False)
        return results


class LoadAnnotations:
    def __init__(self, reduce_zero_label=False):
        # 如果一開始圖像的背景像素是0就會需要是True，如果本身設計背景為255就會需要是False
        # 如果第一個類別在標註圖像是1就會需要是True，如果是0就需要設定成False
        self.reduce_zero_label = reduce_zero_label

    def __call__(self, results):
        label_path = results.get('label_path', None)
        assert label_path is not None, '需提供label_path以獲取標註圖像'
        assert os.path.exists(label_path), f'給定的{label_path}不存在'
        label_image = cv2.imread(label_path)
        label_image = label_image[:, :, 0:1]
        if self.reduce_zero_label:
            label_image[label_image == 0] = 255
            label_image = label_image - 1
            label_image[label_image == 254] = 255
        results['gt_sematic_seg'] = label_image
        results['seg_fields'] = ['gt_sematic_seg']
        return results


class Resize:
    def __init__(self, img_scale=None, multiscale_mode='range', ratio_range=None, keep_ratio=True, min_size=None):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
        if ratio_range is not None:
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            assert multiscale_mode in ['value', 'range']
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.min_size = min_size

    @staticmethod
    def random_select(img_scales):
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1]) * ratio
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h), self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError
        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        if self.keep_ratio:
            if self.min_size is not None:
                if min(results['scale']) < self.min_size:
                    new_short = self.min_size
                else:
                    new_short = min(results['scale'])
                h, w = results['img'].shape[:2]
                if h > w:
                    new_h, new_w = new_short * h / w, new_short
                else:
                    new_h, new_w = new_short, new_short * w / h
                results['scale'] = (new_h, new_w)
            img, scale_factor = imrescale(results['img'], results['scale'], return_scale=True)
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = imresize(results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = imrescale(results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = imresize(results[key], results['scale'], interpolation='nearest')
            results[key] = gt_seg

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results


class RandomCrop:
    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]
        return crop_y1, crop_y2, crop_x1, crop_x2

    @staticmethod
    def crop(img, crop_bbox):
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            for _ in range(10):
                seg_temp = self.crop(results['gt_sematic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)
        return results


class RandomFlip:
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert 0 <= prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            results['img'] = imflip(results['img'], direction=results['flip_direction'])
            for key in results.get('seg_fields', []):
                results[key] = imflip(results[key], direction=results['flip_direction'])
        return results


class PhotoMetricDistortion:
    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    @staticmethod
    def convert(img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if random.randint(2):
            return self.convert(img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img):
        if random.randint(2):
            return self.convert(img, alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        if random.randint(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(img[:, :, 1], alpha=random.uniform(self.contrast_lower, self.contrast_upper))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        if random.randint(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def __call__(self, results):
        img = results['img']
        img = self.brightness(img)
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)
        img = self.saturation(img)
        img = self.hue(img)
        if mode == 0:
            img = self.contrast(img)
        results['img'] = img
        return results


class Normalize:
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results['img'] = imnormalize(results['img'], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


class Pad:
    def __init__(self, size=None, size_divisor=None, pad_val=0, seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        if self.size is not None:
            padding_img = impad(results['img'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padding_img = impad_to_multiple(results['img'], self.size_divisor, pad_val=self.pad_val)
        else:
            raise ValueError('需至少提供一個resize方式')
        results['img'] = padding_img
        results['pad_shape'] = padding_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        for key in results.get('seg_fields', []):
            results[key] = impad(results[key], shape=results['pad_shape'][:2], pad_val=self.seg_pad_val)

    def __call__(self, results):
        self._pad_img(results)
        self._pad_seg(results)
        return results
