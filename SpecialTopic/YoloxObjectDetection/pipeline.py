import copy
import math
import os
import cv2
import numpy as np
import random
from utils import find_inside_bboxes, imgflip, bbox_flip, imresize, imrescale, impad, impad_to_multiple


class LoadImageFromFile:
    def __init__(self, save_key):
        support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
        self.support_image_format = support_image_format
        self.save_key = save_key

    def __call__(self, result):
        image_path = result.get('image_path', None)
        assert image_path is not None
        assert os.path.isfile(image_path)
        assert os.path.splitext(image_path)[1] in self.support_image_format
        image = cv2.imread(image_path)
        result[self.save_key] = image
        return result


class LoadAnnotations:
    def __init__(self, img_key, save_key, with_bbox=False, with_mask=False, with_keypoint=False):
        self.img_key = img_key
        self.save_key = save_key
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.with_keypoint = with_keypoint

    def __call__(self, data):
        annotation_path = data.get('annotation_path', None)
        assert annotation_path is not None
        image = data.get(self.img_key, None)
        assert image is not None, f'指定 {self.img_key} 不在data當中'
        img_height, img_width = image.shape[:2]
        bboxes = list()
        labels = list()
        with open(annotation_path) as f:
            annotations = f.readlines()
        for annotation in annotations:
            label, center_x, center_y, width, height = annotation.strip().split(' ')
            label = int(label)
            center_x = float(center_x) * img_width
            center_y = float(center_y) * img_height
            width = float(width) * img_width
            height = float(height) * img_height
            xmin = center_x - width / 2
            ymin = center_y - height / 2
            xmax = center_x + width / 2
            ymax = center_y + height / 2
            bbox = np.array([xmin, ymin, xmax, ymax])
            label = np.array([label])
            bboxes.append(bbox)
            labels.append(label)
        bboxes = np.array(bboxes)
        labels = np.array(labels)
        labels = np.squeeze(labels, axis=-1)
        data[self.save_key[0]] = labels
        data[self.save_key[1]] = bboxes
        return data


class Mosaic:
    def __init__(self, img_scale=(640, 640), center_ratio_range=(0.5, 1.5), min_bbox_size=0, bbox_clip_border=True,
                 skip_filter=True, pad_val=114, prob=1.0):
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, '觸發Mosaic概率需要在[0, 1.0]之間'
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter
        self.pad_val = pad_val
        self.prob = prob

    def __call__(self, results):
        if random.uniform(0, 1) > self.prob:
            return results
        results = self._mosaic_transform(results)
        return results

    @staticmethod
    def get_indexes(dataset):
        indexes = [np.random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (y2 - y1), img_shape_wh[0], img_shape_wh[1]
        elif loc == 'top_right':
            x1, y1, x2, y2 = center_position_xy[0], max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0], self.img_scale[1] * 2), center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(img_shape_wh[0], x2 - x1), img_shape_wh[1]
        elif loc == 'bottom_left':
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), center_position_xy[1], \
                             center_position_xy[0], min(center_position_xy[1] + img_shape_wh[1], self.img_scale[0] * 2)
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(y2 - y1, img_shape_wh[1])
        else:
            x1, y1, x2, y2 = center_position_xy[0], center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0], self.img_scale[1] * 2), \
                             min(center_position_xy[1] + img_shape_wh[1], self.img_scale[0] * 2)
            crop_coord = 0, 0, min(img_shape_wh[0], x2 - x1), min(y2 - y1, img_shape_wh[1])
        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def _filter_box_candidates(self, bboxes, labels):
        bbox_w = bboxes[:, 2] - bboxes[:, 0]
        bbox_h = bboxes[:, 3] - bboxes[:, 1]
        valid_inds = (bbox_w > self.min_bbox_size) & (bbox_h > self.min_bbox_size)
        valid_inds = np.nonzero(valid_inds)[0]
        return bboxes[valid_inds], labels[valid_inds]

    @staticmethod
    def find_inside_bboxes(bboxes, img_h, img_w):
        inside_inds = (bboxes[:, 0] < img_w) & (bboxes[:, 2] > 0) & (bboxes[:, 1] < img_h) & (bboxes[:, 3] > 0)
        return inside_inds

    def _mosaic_transform(self, results):
        assert 'mix_results' in results
        mosaic_labels = []
        mosaic_bboxes = []
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (center_x, center_y)
        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = copy.deepcopy(results)
            else:
                results_patch = copy.deepcopy(results['mix_results'][i - 1])
            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            scale_ratio_i = min(self.img_scale[0] / h_i, self.img_scale[1] / w_i)
            img_i = cv2.resize(img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
            paste_coord, crop_coord = self._mosaic_combine(loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord
            mosaic_img[y1_p: y2_p, x1_p: x2_p] = img_i[y1_c: y2_c, x1_c: x2_c]
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_labels_i = results_patch['gt_labels']

            if gt_bboxes_i.shape[0] > 0:
                padw = x1_p - x1_c
                padh = y1_p - y1_c
                gt_bboxes_i[:, 0::2] = scale_ratio_i * gt_bboxes_i[:, 0::2] + padw
                gt_bboxes_i[:, 1::2] = scale_ratio_i * gt_bboxes_i[:, 1::2] + padh
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_labels.append(gt_labels_i)
        if len(mosaic_labels) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes, 0)
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            if self.bbox_clip_border:
                mosaic_bboxes[:, 0::2] = np.clip(mosaic_bboxes[:, 0::2], 0, self.img_scale[1] * 2)
                mosaic_bboxes[:, 1::2] = np.clip(mosaic_bboxes[:, 1::2], 0, self.img_scale[0] * 2)
            if not self.skip_filter:
                mosaic_bboxes, mosaic_labels = self._filter_box_candidates(mosaic_bboxes, mosaic_labels)
        inside_inds = self.find_inside_bboxes(mosaic_bboxes, 2 * self.img_scale[0], 2 * self.img_scale[1])
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_labels = mosaic_labels[inside_inds]
        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_labels'] = mosaic_labels
        return results


class RandomAffine:
    def __init__(self, max_rotate_degree=10.0, max_translate_ratio=0.1, scaling_ratio_range=(0.5, 1.5),
                 max_shear_degree=2.0, border=(0, 0), border_val=(114, 114, 114), min_bbox_size=2, min_area_ratio=0.2,
                 max_aspect_ratio=20, bbox_clip_border=True, skip_filter=True):
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.min_bboxes_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.bbox_clip_border = bbox_clip_border
        self.skip_filter = skip_filter

    @staticmethod
    def _get_rotation_matrix(rotation_degrees):
        radian = math.radians(rotation_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.], [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32
        )
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array([[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]], dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees, y_shear_degrees):
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.], [np.tan(y_radian), 1, 0.], [0., 0., 1.]], dtype=np.float32)
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x, y):
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]], dtype=np.float32)
        return translation_matrix

    def __call__(self, results):
        img = results['img']
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2
        rotation_degree = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)
        scaling_ratio = random.uniform(self.scaling_ratio_range[0], self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)
        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)
        trans_x = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio, self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)
        img = cv2.warpPerspective(img, warp_matrix, dsize=(width, height), borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape
        for key in results.get('bbox_fields', ['gt_bboxes']):
            bboxes = results[key]
            num_bboxes = len(bboxes)
            if num_bboxes:
                xs = bboxes[:, [0, 0, 2, 2]].reshape(num_bboxes * 4)
                ys = bboxes[:, [1, 3, 3, 1]].reshape(num_bboxes * 4)
                ones = np.ones_like(xs)
                points = np.vstack([xs, ys, ones])
                warp_points = warp_matrix @ points
                warp_points = warp_points[:2] / warp_points[2]
                xs = warp_points[0].reshape(num_bboxes, 4)
                ys = warp_points[1].reshape(num_bboxes, 4)
                warp_bboxes = np.vstack((xs.min(1), ys.min(1), xs.max(1), ys.max(1))).T
                if self.bbox_clip_border:
                    warp_bboxes[:, [0, 2]] = warp_bboxes[:, [0, 2]].clip(0, width)
                    warp_bboxes[:, [1, 3]] = warp_bboxes[:, [1, 3]].clip(0, height)
                valid_index = find_inside_bboxes(warp_bboxes, height, width)
                if not self.skip_filter:
                    pass
                results[key] = warp_bboxes[valid_index]
                if key in ['gt_bboxes']:
                    if 'gt_labels' in results:
                        results['gt_labels'] = results['gt_labels'][valid_index]
        return results


class MixUp:
    def __init__(self, img_scale=(640, 640), ratio_range=(0.5, 1.5), flip_ratio=0.5, pad_val=114, max_iters=15,
                 min_bbox_size=5, min_area_ratio=0.2, max_aspect_ratio=20, bbox_clip_border=True, skip_filter=True):
        assert isinstance(img_scale, tuple)
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

    def get_indexes(self, dataset):
        index = 0
        for i in range(self.max_iters):
            index = np.random.randint(0, len(dataset))
            gt_bboxes_i = dataset[index]['gt_bboxes']
            if len(gt_bboxes_i) != 0:
                break
        return index

    def __call__(self, results):
        assert 'mix_results' in results
        assert len(results['mix_results']) == 1, '這裡只支援合成一張圖上去'
        if results['mix_results'][0]['gt_bboxes'].shape[0] == 0:
            return results
        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']
        jit_factor = random.uniform(*self.ratio_range)
        is_flip = random.uniform(0, 1) > self.flip_ratio
        if len(retrieve_img.shape) == 3:
            out_img = np.ones((self.dynamic_scale[0], self.dynamic_scale[1], 3),
                              dtype=retrieve_img.dtype) * self.pad_val
        else:
            out_img = np.ones(self.dynamic_scale, dtype=retrieve_img.dtype) * self.pad_val
        scale_ratio = min(self.dynamic_scale[0] / retrieve_img.shape[0], self.dynamic_scale[1] / retrieve_img.shape[1])
        retrieve_img = cv2.resize(retrieve_img,
                                  (int(retrieve_img.shape[0] * scale_ratio), int(retrieve_img.shape[1] * scale_ratio)),
                                  interpolation=cv2.INTER_NEAREST)

        out_img[:retrieve_img.shape[0], :retrieve_img.shape[1]] = retrieve_img
        scale_ratio *= jit_factor
        out_img = cv2.resize(out_img, (int(out_img.shape[0] * jit_factor), int(out_img.shape[1] * jit_factor)),
                             interpolation=cv2.INTER_NEAREST)
        if is_flip:
            out_img = out_img[:, ::-1, :]
        ori_img = results['img']
        origin_h, origin_w = out_img.shape[:2]
        target_h, target_w = ori_img.shape[:2]
        padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), 3)).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = out_img
        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w)
        padded_cropped_img = padded_img[y_offset: y_offset + target_h, x_offset: x_offset + target_w]

        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes[:, 0::2] = retrieve_gt_bboxes[:, 0::2] * scale_ratio
        retrieve_gt_bboxes[:, 1::2] = retrieve_gt_bboxes[:, 1::2] * scale_ratio
        if self.bbox_clip_border:
            retrieve_gt_bboxes[:, 0::2] = np.clip(retrieve_gt_bboxes[:, 0::2], 0, origin_w)
            retrieve_gt_bboxes[:, 1::2] = np.clip(retrieve_gt_bboxes[:, 1::2], 0, origin_h)
        if is_flip:
            retrieve_gt_bboxes[:, 0::2] = (origin_w - retrieve_gt_bboxes[:, 0::2][:, ::-1])
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.copy()
        cp_retrieve_gt_bboxes[:, 0::2] = cp_retrieve_gt_bboxes[:, 0::2] - x_offset
        cp_retrieve_gt_bboxes[:, 1::2] = cp_retrieve_gt_bboxes[:, 1::2] - y_offset
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes[:, 0::2] = np.clip(cp_retrieve_gt_bboxes[:, 0::2], 0, target_w)
            cp_retrieve_gt_bboxes[:, 1::2] = np.clip(cp_retrieve_gt_bboxes[:, 1::2], 0, target_h)
        ori_img = ori_img.astype(np.float32)
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img.astype(np.float32)

        retrieve_gt_labels = retrieve_results['gt_labels']
        if not self.skip_filter:
            pass
        mixup_gt_bboxes = np.concatenate((results['gt_bboxes'], cp_retrieve_gt_bboxes), axis=0)
        mixup_gt_labels = np.concatenate((results['gt_labels'], retrieve_gt_labels), axis=0)
        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_labels'] = mixup_gt_labels
        return results


class YOLOXHSVRandomAug:
    def __init__(self, hue_delta=5, saturation_delta=30, value_delta=30):
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def __call__(self, results):
        img = results['img']
        hsv_gains = np.random.uniform(-1, 1, 3) * [self.hue_delta, self.saturation_delta, self.value_delta]
        hsv_gains *= np.random.randint(0, 2, 3)
        hsv_gains = hsv_gains.astype(np.int16)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 100
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
        cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)
        results['img'] = img
        return results


class RandomFlip:
    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError
        self.flip_ratio = flip_ratio
        valid_directions = ['horizontal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError
        self.direction = direction
        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def __call__(self, results):
        cur_dir = 'horizontal'
        if 'flip' not in results:
            if isinstance(self.direction, list):
                direction_list = self.direction + [None]
            else:
                direction_list = [self.direction, None]
            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) - 1) + [non_flip_ratio]
            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
            results['flip'] = cur_dir is not None
        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir
        if results['flip']:
            for key in results.get('img_fields', ['img']):
                results[key] = imgflip(results[key], direction=results['flip_direction'])
            for key in results.get('bbox_fields', ['gt_bboxes']):
                results[key] = bbox_flip(results[key], results['img_shape'], results['flip_direction'])
        return results


class Resize:
    def __init__(self, img_scale=None, multiscale_mode='range', ratio_range=None, keep_ratio=True,
                 bbox_clip_border=True, interpolation='bilinear', override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
        if ratio_range is not None:
            assert len(self.img_scale) == 1
        else:
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.bbox_clip_border = bbox_clip_border
        self.override = override

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio < max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    @staticmethod
    def random_select(img_scales):
        scale_idx = np.random.randint(len(img_scales))
        img_scales = img_scales[scale_idx]
        return img_scales, scale_idx

    @staticmethod
    def random_sample(img_scales):
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
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
        for key in results.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = imrescale(
                    results[key], results['scale'], return_scale=True, interpolation=self.interpolation)
                new_h, new_w = img.shape[:2]
                h, w = results[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = imresize(
                    results[key], results['scale'], return_scale=True, interpolation=self.interpolation)
            results[key] = img
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            results['img_shape'] = img.shape
            results['pad_shape'] = img.shape
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        for key in results.get('bbox_fields', ['gt_bboxes']):
            bboxes = results[key] * results['scale_factor']
            if self.bbox_clip_border:
                img_shape = results['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            results[key] = bboxes

    def __call__(self, results):
        if 'scale' not in results:
            if 'scale_factor' in results:
                img_shape = results['img'].shape[:2]
                scale_factor = results['scale_factor']
                assert isinstance(scale_factor, float)
                results['scale'] = tuple([int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        return results


class Pad:
    def __init__(self, size=None, size_divisor=None, pad_to_square=False, pad_val='Default'):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square
        if pad_to_square:
            assert size is None and size_divisor is None
        else:
            assert size is not None or size_divisor is not None
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        pad_val = self.pad_val.get('img', 0)
        padded_img = np.array([])
        for key in results.get('img_fields', ['img']):
            if self.pad_to_square:
                max_size = max(results[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = impad(results[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                padded_img = impad_to_multiple(results[key], self.size_divisor, pad_val=pad_val)
            results[key] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        self._pad_img(results)
        return results


class FilterAnnotations:
    def __init__(self, min_gt_bbox_wh=(1., 1.), min_gt_mask_area=1, by_bbox=True, by_mask=False, keep_empty=True):
        assert by_bbox or by_mask
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.min_gt_mask_area = min_gt_mask_area
        self.by_box = by_bbox
        self.by_mask = by_mask
        self.keep_empty = keep_empty

    def __call__(self, results):
        instance_num = 0
        gt_bboxes = None
        if self.by_box:
            assert 'gt_bboxes' in results
            gt_bboxes = results['gt_bboxes']
            instance_num = gt_bboxes.shape[0]
        if self.by_mask:
            assert 'gt_masks' in results
            gt_masks = results['gt_masks']
            instance_num = len(gt_masks)
        if instance_num == 0:
            return results
        tests = []
        if self.by_box:
            w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            tests.append((w > self.min_gt_bbox_wh[0]) & (h > self.min_gt_bbox_wh[1]))
        if self.by_mask:
            gt_masks = results['gt_masks']
            tests.append(gt_masks.areas >= self.min_gt_mask_area)
        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t
        keys = ('gt_bboxes', 'gt_labels', 'gt_masks')
        for key in keys:
            if key in results:
                results[key] = results[key][keep]
        if not keep.any():
            if self.keep_empty:
                return None
        return results


class Collect:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        data = dict()
        for key in self.keys:
            info = results.get(key, None)
            assert info is not None, f'在results當中沒有 {info} 資料'
            data[key] = info
        return data
