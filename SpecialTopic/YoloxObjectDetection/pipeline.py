import copy
import math
import os
import cv2
import numpy as np
import random


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
        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
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

    def __call__(self, results):
        img = results['img']
        height = img.shape[0] + self.border[0] * 2
        width = img.shape[1] + self.border[1] * 2
        rotation_degree = random.uniform(-self.max_rotate_degree, self.max_shear_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)
        scaling_ratio = random.uniform(self.scaling_ratio_range[0], self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)


class MixUp:
    def __init__(self, **kwargs):
        pass

    def __call__(self):
        pass


class YOLOXHSVRandomAug:
    def __init__(self, **kwargs):
        pass

    def __call__(self):
        pass


class RandomFlip:
    def __init__(self, **kwargs):
        pass

    def __call__(self):
        pass


class Resize:
    def __init__(self, **kwargs):
        pass

    def __call__(self):
        pass


class Pad:
    def __init__(self, **kwargs):
        pass

    def __call__(self):
        pass


class FilterAnnotations:
    def __init__(self, **kwargs):
        pass

    def __call__(self):
        pass


class Collect:
    def __init__(self, **kwargs):
        pass

    def __call__(self):
        pass
