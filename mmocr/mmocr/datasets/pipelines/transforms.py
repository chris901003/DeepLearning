# Copyright (c) OpenMMLab. All rights reserved.
import math

import cv2
import mmcv
import numpy as np
import torchvision.transforms as transforms
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.transforms import Resize
from PIL import Image
from shapely.geometry import Polygon as plg

import mmocr.core.evaluation.utils as eval_utils
from mmocr.utils import check_argument


@PIPELINES.register_module()
class RandomCropInstances:
    """Randomly crop images and make sure to contain text instances.

    Args:
        target_size (tuple or int): (height, width)
        positive_sample_ratio (float): The probability of sampling regions
            that go through positive regions.
    """

    def __init__(
            self,
            target_size,
            instance_key,
            mask_type='inx0',  # 'inx0' or 'union_all'
            positive_sample_ratio=5.0 / 8.0):
        """ 已看過，隨機剪裁並且保證圖像當中包含文字檢測對象
        Args:
            target_size: tuple或是int，剪裁後圖像大小
            instance_key:
            mask_type: mask的方式
            positive_sample_ratio: 剪裁後圖像正負樣本的比例
        """

        # 檢查mask_type是否合法
        assert mask_type in ['inx0', 'union_all']

        # 保存傳入的參數
        self.mask_type = mask_type
        self.instance_key = instance_key
        self.positive_sample_ratio = positive_sample_ratio
        self.target_size = target_size if (target_size is None or isinstance(
            target_size, tuple)) else (target_size, target_size)

    def sample_offset(self, img_gt, img_size):
        """ 已看過，獲取offset
        Args:
            img_gt: mask的資訊，ndarray shape [height, width]
            img_size: 當前圖像大小，tuple (height, width)
        """
        # 獲取當前圖像高寬
        h, w = img_size
        # 獲取經過裁切後的高寬
        t_h, t_w = self.target_size

        # target size is bigger than origin size
        # 經過裁切後圖像只會變小或是不變，不會變大
        t_h = t_h if t_h < h else h
        t_w = t_w if t_w < w else w
        if (img_gt is not None
                and np.random.random_sample() < self.positive_sample_ratio
                and np.max(img_gt) > 0):

            # make sure to crop the positive region
            # 確保裁切後是取到正範圍

            # the minimum top left to crop positive region (h,w)
            # 獲取圖像當中最左上角有出現標註的點之後再扣去目標高寬
            tl = np.min(np.where(img_gt > 0), axis=1) - (t_h, t_w)
            # 如果小於0就變成0
            tl[tl < 0] = 0
            # the maximum top left to crop positive region
            # 獲取類似最右下角點之後再扣除目標高寬
            br = np.max(np.where(img_gt > 0), axis=1) - (t_h, t_w)
            # 如果小於0就變成0
            br[br < 0] = 0
            # if br is too big so that crop the outside region of img
            # 控制br長度
            br[0] = min(br[0], h - t_h)
            br[1] = min(br[1], w - t_w)
            # 在範圍內隨機挑選高寬
            h = np.random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
            w = np.random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            # make sure not to crop outside of img

            h = np.random.randint(0, h - t_h) if h - t_h > 0 else 0
            w = np.random.randint(0, w - t_w) if w - t_w > 0 else 0

        # 最後回傳高寬，這個高寬估計會是左上角點，右下角點就會是(h + t_h, w + t_w)位置
        return (h, w)

    @staticmethod
    def crop_img(img, offset, target_size):
        # 已看過，對圖像進行裁切
        h, w = img.shape[:2]
        # br = 右下角點
        br = np.min(
            np.stack((np.array(offset) + np.array(target_size), np.array(
                (h, w)))),
            axis=0)
        # 回傳裁切好的圖像以及裁切的範圍
        return img[offset[0]:br[0], offset[1]:br[1]], np.array(
            [offset[1], offset[0], br[1], br[0]])

    def crop_bboxes(self, bboxes, canvas_bbox):
        kept_bboxes = []
        kept_inx = []
        canvas_poly = eval_utils.box2polygon(canvas_bbox)
        tl = canvas_bbox[0:2]

        for idx, bbox in enumerate(bboxes):
            poly = eval_utils.box2polygon(bbox)
            area, inters = eval_utils.poly_intersection(
                poly, canvas_poly, return_poly=True)
            if area == 0:
                continue
            xmin, ymin, xmax, ymax = inters.bounds
            kept_bboxes += [
                np.array(
                    [xmin - tl[0], ymin - tl[1], xmax - tl[0], ymax - tl[1]],
                    dtype=np.float32)
            ]
            kept_inx += [idx]

        if len(kept_inx) == 0:
            return np.array([]).astype(np.float32).reshape(0, 4), kept_inx

        return np.stack(kept_bboxes), kept_inx

    @staticmethod
    def generate_mask(gt_mask, type):

        if type == 'inx0':
            return gt_mask.masks[0]
        if type == 'union_all':
            mask = gt_mask.masks[0].copy()
            for idx in range(1, len(gt_mask.masks)):
                mask = np.logical_or(mask, gt_mask.masks[idx])
            return mask

        raise NotImplementedError

    def __call__(self, results):
        # 已看過，進行隨機剪裁同時當中比必須要有文字

        # 獲取標註資訊
        gt_mask = results[self.instance_key]
        mask = None
        if len(gt_mask.masks) > 0:
            # 如果有gt_mask就會進來，mask就取出第一種縮放比例的mask，shape[height, width]
            mask = self.generate_mask(gt_mask, self.mask_type)
        # 獲取crop_offset，裁切偏移量
        results['crop_offset'] = self.sample_offset(mask,
                                                    results['img'].shape[:2])

        # crop img. bbox = [x1,y1,x2,y2]
        img, bbox = self.crop_img(results['img'], results['crop_offset'],
                                  self.target_size)
        # 更新圖像
        results['img'] = img
        img_shape = img.shape
        # 更新圖像大小
        results['img_shape'] = img_shape

        # crop masks，對mask進行裁切
        for key in results.get('mask_fields', []):
            results[key] = results[key].crop(bbox)

        # for mask rcnn
        for key in results.get('bbox_fields', []):
            results[key], kept_inx = self.crop_bboxes(results[key], bbox)
            if key == 'gt_bboxes':
                # ignore gt_labels accordingly
                if 'gt_labels' in results:
                    ori_labels = results['gt_labels']
                    ori_inst_num = len(ori_labels)
                    results['gt_labels'] = [
                        ori_labels[idx] for idx in range(ori_inst_num)
                        if idx in kept_inx
                    ]
                # ignore g_masks accordingly
                if 'gt_masks' in results:
                    ori_mask = results['gt_masks'].masks
                    kept_mask = [
                        ori_mask[idx] for idx in range(ori_inst_num)
                        if idx in kept_inx
                    ]
                    target_h, target_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                    if len(kept_inx) > 0:
                        kept_mask = np.stack(kept_mask)
                    else:
                        kept_mask = np.empty((0, target_h, target_w),
                                             dtype=np.float32)
                    results['gt_masks'] = BitmapMasks(kept_mask, target_h,
                                                      target_w)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomRotateTextDet:
    """Randomly rotate images."""

    def __init__(self, rotate_ratio=1.0, max_angle=10):
        """ 已看過，對圖像進行旋轉
        Args:
            rotate_ratio: 旋轉機率
            max_angle: 最大旋轉角度
        """
        self.rotate_ratio = rotate_ratio
        self.max_angle = max_angle

    @staticmethod
    def sample_angle(max_angle):
        # 已看過，從傳入的最大角度內隨機選擇一個數字
        angle = np.random.random_sample() * 2 * max_angle - max_angle
        return angle

    @staticmethod
    def rotate_img(img, angle):
        # 已看過，將圖像進行指定角度旋轉
        # 獲取當前圖像高寬
        h, w = img.shape[:2]
        # 透過getRotationMatrix2D進行旋轉(旋轉中心, 旋轉方向, 縮放比例)
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        # 透過warpAffine進行仿射變換(變換目標, 仿射矩陣, 變換後圖像大小)
        img_target = cv2.warpAffine(
            img, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        # 簡單檢查
        assert img_target.shape == img.shape
        return img_target

    def __call__(self, results):
        # 已看過，主要是旋轉圖像
        if np.random.random_sample() < self.rotate_ratio:
            # rotate imgs
            # 隨機挑選旋轉角度並且記錄到rotated_angle當中
            results['rotated_angle'] = self.sample_angle(self.max_angle)
            # 透過rotate_img將圖像進行旋轉
            img = self.rotate_img(results['img'], results['rotated_angle'])
            # 更新圖像
            results['img'] = img
            img_shape = img.shape
            # 更新圖像大小
            results['img_shape'] = img_shape

            # rotate masks，對mask進行旋轉變換
            for key in results.get('mask_fields', []):
                masks = results[key].masks
                mask_list = []
                for m in masks:
                    # 這裡與img旋轉相同
                    rotated_m = self.rotate_img(m, results['rotated_angle'])
                    mask_list.append(rotated_m)
                # 最後需要用BitmapMasks進行包裝
                results[key] = BitmapMasks(mask_list, *(img_shape[:2]))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class ColorJitter:
    """An interface for torch color jitter so that it can be invoked in
    mmdetection pipeline."""

    def __init__(self, **kwargs):
        # 已看過，對圖像進行增強，kwargs增強的參數
        # 這裡就是hue的數據增強，透過調整亮度以及飽和度以及色度以及對比度
        self.transform = transforms.ColorJitter(**kwargs)

    def __call__(self, results):
        # 已看過，調整圖像的色度等等資料
        # img is bgr，img的圖像排列順序依舊是bgr只是shape是[channel, width, height]
        img = results['img'][..., ::-1]
        # 透過Image將圖像讀取出來
        img = Image.fromarray(img)
        # 使用pytorch官方實現的轉換
        img = self.transform(img)
        # 將圖像從Image轉回成ndarray格式
        img = np.asarray(img)
        # 將通道排列順序調整回來[height, width, channel]
        img = img[..., ::-1]
        # 將圖像資料進行更新
        results['img'] = img
        # 回傳results
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class ScaleAspectJitter(Resize):
    """Resize image and segmentation mask encoded by coordinates.

    Allowed resize types are `around_min_img_scale`, `long_short_bound`, and
    `indep_sample_in_range`.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=False,
                 resize_type='around_min_img_scale',
                 aspect_ratio_range=None,
                 long_size_bound=None,
                 short_size_bound=None,
                 scale_range=None):
        """ 已看過，主要是對圖像以及mask進行resize操作
        Args:
            img_scale: 圖像最後調整到的大小，list[tuple(width, height)]
            multiscale_mode: 縮放模式，如果是range就會在img_scale給的範圍當中隨機挑一個數字作為輸出高寬
                如過是value就直接使用img_scale指定的大小
            ratio_range: 縮放比例的範圍
            keep_ratio: 是否需要保留縮放比例
            resize_type: resize的模式
            aspect_ratio_range: 高寬比的範圍
            long_size_bound: 長邊大小上限
            short_size_bound: 短邊大小下限
            scale_range: 縮放範圍
        """
        # 繼承自Resize，對繼承對象進行初始化
        super().__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=keep_ratio)
        assert not keep_ratio
        # 檢查使用的resize_type是否有支援
        assert resize_type in [
            'around_min_img_scale', 'long_short_bound', 'indep_sample_in_range'
        ]
        # 保存
        self.resize_type = resize_type

        # 下面就是做一些檢查
        if resize_type == 'indep_sample_in_range':
            assert ratio_range is None
            assert aspect_ratio_range is None
            assert short_size_bound is None
            assert long_size_bound is None
            assert scale_range is not None
        else:
            assert scale_range is None
            assert isinstance(ratio_range, tuple)
            assert isinstance(aspect_ratio_range, tuple)
            assert check_argument.equal_len(ratio_range, aspect_ratio_range)

            if resize_type in ['long_short_bound']:
                assert short_size_bound is not None
                assert long_size_bound is not None

        # 做一些保存動作
        self.aspect_ratio_range = aspect_ratio_range
        self.long_size_bound = long_size_bound
        self.short_size_bound = short_size_bound
        self.scale_range = scale_range

    @staticmethod
    def sample_from_range(range):
        # 已看過，獲取一個範圍內的隨機值
        # 檢查range長度會是2
        assert len(range) == 2
        # 取出最大值以及最小值
        min_value, max_value = min(range), max(range)
        # 透過numpy獲取隨機值
        value = np.random.random_sample() * (max_value - min_value) + min_value

        # 回傳範圍內的隨機值
        return value

    def _random_scale(self, results):
        # 已看過，隨機獲取縮放大小

        if self.resize_type == 'indep_sample_in_range':
            # 如果縮放方式是indep_sample_in_range就會到這裡
            w = self.sample_from_range(self.scale_range)
            h = self.sample_from_range(self.scale_range)
            results['scale'] = (int(w), int(h))  # (w,h)
            results['scale_idx'] = None
            return
        # 獲取高寬
        h, w = results['img'].shape[0:2]
        if self.resize_type == 'long_short_bound':
            # 如過是使用long_short_bound就會到這裡來
            # scale1 = 第一個縮放比例
            scale1 = 1
            if max(h, w) > self.long_size_bound:
                # 如果高寬其中一邊的最大值超過指定最大值，我們會透過設定scale1將最大邊變成指定的最大大小
                scale1 = self.long_size_bound / max(h, w)
            # 透過sample_from_range在給定的範圍取出值
            scale2 = self.sample_from_range(self.ratio_range)
            # 最後的縮放比例會是兩個相乘
            scale = scale1 * scale2
            if min(h, w) * scale <= self.short_size_bound:
                # 如果將最小邊乘上縮放比例後會小於設定的最小值我們就需要調整縮放比例
                scale = (self.short_size_bound + 10) * 1.0 / min(h, w)
        elif self.resize_type == 'around_min_img_scale':
            # 如果設定的resize_type是around_min_img_scale會到這裡
            # 找出短編長度
            short_size = min(self.img_scale[0])
            # 從指定的縮放範圍隨機挑出一個值
            ratio = self.sample_from_range(self.ratio_range)
            # 縮放比例就會是隨機縮放比例乘上最短邊最後在除以原始邊
            scale = (ratio * short_size) / min(h, w)
        else:
            raise NotImplementedError

        # 從指定的aspect_ratio範圍隨機選取一個值
        aspect = self.sample_from_range(self.aspect_ratio_range)
        # 獲取最終高寬的縮放比例
        h_scale = scale * math.sqrt(aspect)
        w_scale = scale / math.sqrt(aspect)
        # 更新scale，也就是透過resize後圖像的大小
        results['scale'] = (int(w * w_scale), int(h * h_scale))  # (w,h)
        results['scale_idx'] = None


@PIPELINES.register_module()
class AffineJitter:
    """An interface for torchvision random affine so that it can be invoked in
    mmdet pipeline."""

    def __init__(self,
                 degrees=4,
                 translate=(0.02, 0.04),
                 scale=(0.9, 1.1),
                 shear=None,
                 resample=False,
                 fillcolor=0):
        self.transform = transforms.RandomAffine(
            degrees=degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            resample=resample,
            fillcolor=fillcolor)

    def __call__(self, results):
        # img is bgr
        img = results['img'][..., ::-1]
        img = Image.fromarray(img)
        img = self.transform(img)
        img = np.asarray(img)
        img = img[..., ::-1]
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomCropPolyInstances:
    """Randomly crop images and make sure to contain at least one intact
    instance."""

    def __init__(self,
                 instance_key='gt_masks',
                 crop_ratio=5.0 / 8.0,
                 min_side_ratio=0.4):
        """ 已看過，隨機剪裁但是會確保剪裁後圖像當中至少包含一個標註匡
        Args:
            instance_key: 主要對象名稱
            crop_ratio: 裁切比例
            min_side_ratio: 最小某一邊的比例
        """
        super().__init__()
        # 將傳入的參數保存
        self.instance_key = instance_key
        self.crop_ratio = crop_ratio
        self.min_side_ratio = min_side_ratio

    def sample_valid_start_end(self, valid_array, min_len, max_start, min_end):
        """ 已看過
        Args:
            valid_array: 在圖像當中該行或列都沒有標註圖像會是1，否則就會是0，ndarray shape [w or h]
            min_len: 最小長度
            max_start: 隨機選取一個標註對象的最小x或是最小y
            min_end: 隨機選取一個標註對象的最大x或是最大y
        """

        # 檢查傳入的資料是否合法
        assert isinstance(min_len, int)
        assert len(valid_array) > min_len

        # 將valid_array拷貝一份
        start_array = valid_array.copy()
        # 獲取最大起點，這裡會是min(總長度-最小長度, 標註對象最小x或是最小y)，這樣最左端一定會小於等於標註圖像最左端
        max_start = min(len(start_array) - min_len, max_start)
        # 從max_start後將start_array的地方都設定成0
        start_array[max_start:] = 0
        # 將最頭的start_array直接設定成1
        start_array[0] = 1
        # 這裡的hstack沒有特別作用，就是在start_array前面多一個0與start_array後面多一個0在對應index進行相減，shape [h+1 or w+1]
        diff_array = np.hstack([0, start_array]) - np.hstack([start_array, 0])
        # 找到在diff_array小於0的index
        region_starts = np.where(diff_array < 0)[0]
        # 找到在diff_array大於0的index
        region_ends = np.where(diff_array > 0)[0]
        # 隨機選取region_starts當中的一個index
        region_ind = np.random.randint(0, len(region_starts))
        # start會是從region_starts的指定index到region_ends中隨機選一個值
        start = np.random.randint(region_starts[region_ind],
                                  region_ends[region_ind])

        # 與選則start類似
        end_array = valid_array.copy()
        min_end = max(start + min_len, min_end)
        end_array[:min_end] = 0
        end_array[-1] = 1
        diff_array = np.hstack([0, end_array]) - np.hstack([end_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        end = np.random.randint(region_starts[region_ind],
                                region_ends[region_ind])
        # 最後回傳start與end，透過以上方式一定會取到最一開始隨機選則的標註
        return start, end

    def sample_crop_box(self, img_size, results):
        """Generate crop box and make sure not to crop the polygon instances.

        Args:
            img_size (tuple(int)): The image size (h, w).
            results (dict): The results dict.
        """
        # 已看過，生成剪裁的匡，並且確保不會剪裁到標註匡資料
        # img_size = 輸入圖像的圖像大小
        # results = 輸入圖像詳細資料

        # 檢查img_size是否符合tuple
        assert isinstance(img_size, tuple)
        # 獲取圖像高寬
        h, w = img_size[:2]

        # 獲取標註匡資訊，這裡會是list[list]，第一個list會是該圖像有多少個標註匡，第二個list會是該標註匡的(x, y)資訊
        key_masks = results[self.instance_key].masks
        # 構建全為1且長度為w的ndarray
        x_valid_array = np.ones(w, dtype=np.int32)
        # 構建全為1且長度為h的ndarray
        y_valid_array = np.ones(h, dtype=np.int32)

        # 從標註匡當中隨機選取一個標註匡
        selected_mask = key_masks[np.random.randint(0, len(key_masks))]
        # 將資料排成ndarray shape [points, 2]
        selected_mask = selected_mask[0].reshape((-1, 2)).astype(np.int32)
        # 找到該標註匡最小x，這裡會在最小x後再減2
        max_x_start = max(np.min(selected_mask[:, 0]) - 2, 0)
        # 找到該標註匡最大x，這裡會在最大x後再加3
        min_x_end = min(np.max(selected_mask[:, 0]) + 3, w - 1)
        # 找到該標註匡最小y，這裡會在最小y後再減2
        max_y_start = max(np.min(selected_mask[:, 1]) - 2, 0)
        # 找到該標註匡最大y，這裡會在最大y後再加3
        min_y_end = min(np.max(selected_mask[:, 1]) + 3, h - 1)

        # 遍歷所有與mask相關的資料
        for key in results.get('mask_fields', []):
            if len(results[key].masks) == 0:
                # 如果當中沒有任何標註訊息就直接continue
                continue
            # 獲取當中mask資料
            masks = results[key].masks
            # 遍歷所有的mask
            for mask in masks:
                # 這裡長度需要為1
                assert len(mask) == 1
                # 將通道重新排列，ndarray shape [points, 2]
                mask = mask[0].reshape((-1, 2)).astype(np.int32)
                # 分別獲取標註匡的x與y座標並且限制不會超過高寬，clip_x與clip_y的shape [points]
                clip_x = np.clip(mask[:, 0], 0, w - 1)
                clip_y = np.clip(mask[:, 1], 0, h - 1)
                # 透過clip_x與clip_y獲取標註匡的最小x以及最大x以及最小y以及最大y
                min_x, max_x = np.min(clip_x), np.max(clip_x)
                min_y, max_y = np.min(clip_y), np.max(clip_y)

                # 將x_valid_array在xmin-2到xmax+2之間設定成0，表示這裡是我們需要的
                x_valid_array[min_x - 2:max_x + 3] = 0
                # 將y_valid_array在ymin-2到yma+2之間設定成0，表示這裡是我們需要的
                y_valid_array[min_y - 2:max_y + 3] = 0

        # 獲取最小高寬，會是原始高寬乘上min_side_ratio
        min_w = int(w * self.min_side_ratio)
        min_h = int(h * self.min_side_ratio)

        # 將x_valid_array以及min_w以及隨機挑的一個標註匡中最小x以及最大x傳入
        x1, x2 = self.sample_valid_start_end(x_valid_array, min_w, max_x_start,
                                             min_x_end)
        # 將y_valid_array以及min_h以及隨機挑的一個標註匡中最小y以及最大y傳入
        y1, y2 = self.sample_valid_start_end(y_valid_array, min_h, max_y_start,
                                             min_y_end)

        # 最後回傳x1,y1,x2,y2，圍成的矩形就是我們要的範圍
        return np.array([x1, y1, x2, y2])

    def crop_img(self, img, bbox):
        """ 已看過，對輸入圖像進行指定剪裁
        Args:
            img: 輸入圖像，ndarray shape [height, width, channel]
            bbox: 指定的剪裁範圍
        """
        assert img.ndim == 3
        h, w, _ = img.shape
        assert 0 <= bbox[1] < bbox[3] <= h
        assert 0 <= bbox[0] < bbox[2] <= w
        # 用切片方式進行剪裁
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def __call__(self, results):
        # 已看過，進行隨機剪裁，剪裁後的圖像當中一定會包含至少一個標註匡
        if len(results[self.instance_key].masks) < 1:
            # 如果本身圖像當中就沒有標註匡就會直接回傳
            return results
        if np.random.random_sample() < self.crop_ratio:
            # 如果進來就表示會進行剪裁，這裡進來是有機率的
            # 透過sample_crop_box獲取剪裁範圍，ndarray [x1, y1, x2, y2]
            crop_box = self.sample_crop_box(results['img'].shape, results)
            # 保存裁切範圍
            results['crop_region'] = crop_box
            # 使用crop_img對圖像進行剪裁
            img = self.crop_img(results['img'], crop_box)
            # 更新圖像資料
            results['img'] = img
            results['img_shape'] = img.shape

            # crop and filter masks
            # 將剪裁的(x1, y1, x2, y2)提取出來
            x1, y1, x2, y2 = crop_box
            # 獲取新的圖像高寬
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            # 獲取每個標註匡對應的labels資訊
            labels = results['gt_labels']
            valid_labels = []
            # 遍歷所有與mask相關的資訊
            for key in results.get('mask_fields', []):
                if len(results[key].masks) == 0:
                    # 如果當中沒有標註資料就直接跳過
                    continue
                # 透過crop進行剪裁，這裡會將標註訊息根據平移進行調整
                results[key] = results[key].crop(crop_box)
                # filter out polygons beyond crop box，獲取剪裁過後的多邊形標註資料
                masks = results[key].masks
                # 當中可能有些因為剪裁後變成不合法的，所以這裡會有保存合法的標註訊息的list
                valid_masks_list = []

                # 遍歷所有的標註訊息
                for ind, mask in enumerate(masks):
                    # mask的shape是list[list]
                    assert len(mask) == 1
                    # 將mask的shape調整一下變成[points, 2]
                    polygon = mask[0].reshape((-1, 2))
                    if (polygon[:, 0] >
                            -4).all() and (polygon[:, 0] < w + 4).all() and (
                                polygon[:, 1] > -4).all() and (polygon[:, 1] <
                                                               h + 4).all():
                        # 如果x座標的範圍在[-3, w + 3]且y座標的範圍在[-3, h + 3]之間就會進來
                        # 將x座標限定在[0, w)且y座標限定在[0, h)之間
                        mask[0][::2] = np.clip(mask[0][::2], 0, w)
                        mask[0][1::2] = np.clip(mask[0][1::2], 0, h)
                        if key == self.instance_key:
                            # 如果當前的key是gt_masks就會將該index的labels保存
                            valid_labels.append(labels[ind])
                        # 保存調整後的mask放到valid_masks_list當中
                        valid_masks_list.append(mask)

                # 最後將裁切後還在圖像當中的標註訊息放入PolygonMaks實例化對象當中
                results[key] = PolygonMasks(valid_masks_list, h, w)
            # 更新labels資訊
            results['gt_labels'] = np.array(valid_labels)

        # 最後回傳更新好的results
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomRotatePolyInstances:

    def __init__(self,
                 rotate_ratio=0.5,
                 max_angle=10,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0)):
        """Randomly rotate images and polygon masks.

        Args:
            rotate_ratio (float): The ratio of samples to operate rotation.
            max_angle (int): The maximum rotation angle.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rotated image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        # 已看過，主要是對圖像以及標註的多邊形mask進行旋轉
        # rotate_ratio = 隨機旋轉的機率
        # max_angle = 最大選轉角度
        # pad_with_fixed_color = 旋轉後是否需要填充一個固定值讓圖像大小不變
        # pad_value = 填充的值

        # 保存傳入的參數
        self.rotate_ratio = rotate_ratio
        self.max_angle = max_angle
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def rotate(self, center, points, theta, center_shift=(0, 0)):
        """ 已看過，將mask部分進行旋轉
        Args:
            center: 旋轉中心位置
            points: 標註點，ndarray shape [points, 2]
            theta: 旋轉角度
            center_shift: 中心偏移量
        """
        # rotate points.
        # 獲取旋轉中心點
        (center_x, center_y) = center
        center_y = -center_y
        # 將x與y分開
        x, y = points[::2], points[1::2]
        y = -y

        # 以下就開始將點進行相對應得旋轉
        theta = theta / 180 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        x = (x - center_x)
        y = (y - center_y)

        _x = center_x + x * cos - y * sin + center_shift[0]
        _y = -(center_y + x * sin + y * cos) + center_shift[1]

        points[::2], points[1::2] = _x, _y
        return points

    def cal_canvas_size(self, ori_size, degree):
        # 已看過，獲取經過旋轉後圖像的大小
        # ori_size = 原始圖像的高寬
        assert isinstance(ori_size, tuple)
        # 獲取旋轉角度
        angle = degree * math.pi / 180.0
        # 獲取圖像高寬
        h, w = ori_size[:2]

        cos = math.cos(angle)
        sin = math.sin(angle)
        # 計算最後旋轉後的高寬
        canvas_h = int(w * math.fabs(sin) + h * math.fabs(cos))
        canvas_w = int(w * math.fabs(cos) + h * math.fabs(sin))

        # 打包成tuple(height, width)
        canvas_size = (canvas_h, canvas_w)
        return canvas_size

    def sample_angle(self, max_angle):
        # 已看過，獲取隨機旋轉角度
        # max_angle = 最大旋轉角度
        angle = np.random.random_sample() * 2 * max_angle - max_angle
        # 回傳隨機選取的角度
        return angle

    def rotate_img(self, img, angle, canvas_size):
        """ 已看過，將圖像進行旋轉
        Args:
            img: 當前圖像資料，ndarray shape [height, width, channel]
            angle: 旋轉角度
            canvas_size: 旋轉後圖像的高寬，tuple(height, width)
        """
        # 獲取圖像高寬
        h, w = img.shape[:2]
        # 透過getRotationMatrix2D進行旋轉的實例對象，傳入的參數為(圖像中心, 旋轉角度, 縮放比例)
        # rotation_matrix = 仿射變換矩陣
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        # 在指定位置加上值
        rotation_matrix[0, 2] += int((canvas_size[1] - w) / 2)
        rotation_matrix[1, 2] += int((canvas_size[0] - h) / 2)

        if self.pad_with_fixed_color:
            # 如果有指定padding時要用哪個固定的顏色會到這裡
            # warpAffine傳入的參數為(要旋轉的圖像, 仿射變換矩陣, 最後輸出圖像大小)
            # flags = 差值方法的組合，borderValue = 邊界填充值
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                flags=cv2.INTER_NEAREST,
                borderValue=self.pad_value)
        else:
            # 沒有指定會到這裡
            # 構建一個shape與img相同且全為0的mask
            mask = np.zeros_like(img)
            # 隨機獲取h_ind與w_ind的值
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            # 擷取一部分的原始圖像
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            # 將擷取的資料透過resize將大小縮放到canvas_size
            img_cut = mmcv.imresize(img_cut, (canvas_size[1], canvas_size[0]))
            # 將mask進行旋轉並且padding部分填充為[1, 1, 1]
            mask = cv2.warpAffine(
                mask,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[1, 1, 1])
            # 將原始圖像進行旋轉，填充為[0, 0, 0]
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[0, 0, 0])
            # 最後將原圖加上經過mask的img_cut
            target_img = target_img + img_cut * mask

        return target_img

    def __call__(self, results):
        # 已看過，將圖像以及mask進行旋轉
        if np.random.random_sample() < self.rotate_ratio:
            # 這裡會是有機率的對圖像進行旋轉
            # 獲取當前圖像
            img = results['img']
            # 獲取圖像的高寬資料
            h, w = img.shape[:2]
            # 使用sample_angle獲取旋轉角度
            angle = self.sample_angle(self.max_angle)
            # 獲取旋轉後圖像大小，canvas_size = tuple(height, width)
            canvas_size = self.cal_canvas_size((h, w), angle)
            # 計算中心偏移量，會是減少的長度的一半
            center_shift = (int(
                (canvas_size[1] - w) / 2), int((canvas_size[0] - h) / 2))

            # rotate image
            # 保存旋轉的角度
            results['rotated_poly_angle'] = angle
            # 將圖像進行旋轉
            img = self.rotate_img(img, angle, canvas_size)
            # 更新圖像資訊
            results['img'] = img
            img_shape = img.shape
            results['img_shape'] = img_shape

            # rotate polygons，將與mask有關係部分進行旋轉
            for key in results.get('mask_fields', []):
                if len(results[key].masks) == 0:
                    # 如果標註當中沒有內容就直接continue
                    continue
                # 獲取mask資訊
                masks = results[key].masks
                # 旋轉後的點會暫時存放在這裡
                rotated_masks = []
                # 遍歷所有點
                for mask in masks:
                    # 使用RandomRotatePolyInstances當中的rotate方法
                    rotated_mask = self.rotate((w / 2, h / 2), mask[0], angle,
                                               center_shift)
                    # 將結果傳入
                    rotated_masks.append([rotated_mask])

                # 最後將結果傳入到PolygonMasks建構實例對象
                results[key] = PolygonMasks(rotated_masks, *(img_shape[:2]))

        # 回傳更新後的results
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class SquareResizePad:

    def __init__(self,
                 target_size,
                 pad_ratio=0.6,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0)):
        """Resize or pad images to be square shape.

        Args:
            target_size (int): The target size of square shaped image.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rescales image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        # 已看過，主要是透過resize或是padding將圖像變成正方形
        # target_size = 輸出時圖像的大小，會是一個int
        # pad_with_fixed_color = 在padding時是否用固定的顏色
        # pad_value = 在pad時填充的顏色

        # 檢查傳入參數是否正確
        assert isinstance(target_size, int)
        assert isinstance(pad_ratio, float)
        assert isinstance(pad_with_fixed_color, bool)
        assert isinstance(pad_value, tuple)

        # 保存傳入參數
        self.target_size = target_size
        self.pad_ratio = pad_ratio
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def resize_img(self, img, keep_ratio=True):
        """ 已看過，將圖像進行resize
        Args:
            img: 當前圖像資料，ndarray shape [height, width, channel]
            keep_ratio: 是否需要保持高寬比
        """
        # 獲取當前圖像高寬
        h, w, _ = img.shape
        if keep_ratio:
            # 如果需要保持高寬比會在這裡，我們會將較長邊設定成target_size
            t_h = self.target_size if h >= w else int(h * self.target_size / w)
            t_w = self.target_size if h <= w else int(w * self.target_size / h)
        else:
            # 否則就強制最後高寬都是target_size
            t_h = t_w = self.target_size
        # 使用imresize進行resize
        img = mmcv.imresize(img, (t_w, t_h))
        # 回傳最後圖像以及高寬
        return img, (t_h, t_w)

    def square_pad(self, img):
        # 已看過，主要是將圖像padding成正方形的
        # img = 當前圖像資訊，ndarray shape [height, width, channel]

        # 獲取當前的高寬
        h, w = img.shape[:2]
        if h == w:
            # 如果高寬已經相同就代表已經是正方形，直接回傳
            return img, (0, 0)
        # padding後的大小會是當前圖像高寬較大的一邊
        pad_size = max(h, w)
        if self.pad_with_fixed_color:
            # 如果有指定padding的數值就會到這裡
            # 構建expand_img的ndarray shape [pad_size, pad_size, 3]且全為1
            expand_img = np.ones((pad_size, pad_size, 3), dtype=np.uint8)
            # 將expand_img當中的值全部改成指定的padding值
            expand_img[:] = self.pad_value
        else:
            # 如果沒有指定padding的數值就會到這裡
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            # 切下原圖指定的範圍
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            # 將img_cut擴展到pad_size大小
            expand_img = mmcv.imresize(img_cut, (pad_size, pad_size))
        # 依據高寬哪個是長邊會是上下需要padding，或是左右需要padding
        if h > w:
            y0, x0 = 0, (h - w) // 2
        else:
            y0, x0 = (w - h) // 2, 0
        expand_img[y0:y0 + h, x0:x0 + w] = img
        # 回傳圖像偏移量，好讓mask以及bbox進行調整
        offset = (x0, y0)

        # 回傳
        return expand_img, offset

    def square_pad_mask(self, points, offset):
        # 已看過，將標註點根據偏移量調整
        # 獲取偏移量
        x0, y0 = offset
        pad_points = points.copy()
        # 將標註點進行偏移調整
        pad_points[::2] = pad_points[::2] + x0
        pad_points[1::2] = pad_points[1::2] + y0
        return pad_points

    def __call__(self, results):
        # 已看過，主要是將圖像經過resize或是padding變成正方形
        # 獲取圖像資料
        img = results['img']

        # 這裡是有機率使用padding將圖像變成正方形或是使用resize方式變成正方形
        if np.random.random_sample() < self.pad_ratio or True:
            # 這裡是透過padding變成正方形
            # img = resize後的圖像，out_size = 經過resize後圖像的高寬
            img, out_size = self.resize_img(img, keep_ratio=True)
            # img = 經過padding後的圖像，offset = padding的長寬
            img, offset = self.square_pad(img)
        else:
            # 如果直接透過resize，就將保持原始高寬比關閉強制resize
            img, out_size = self.resize_img(img, keep_ratio=False)
            # 這裡的offset就會是(0, 0)
            offset = (0, 0)

        # 更新圖像資訊
        results['img'] = img
        results['img_shape'] = img.shape

        # 遍歷所有與mask相關的資料
        for key in results.get('mask_fields', []):
            if len(results[key].masks) == 0:
                # 如果當中沒有資料就會直接continue
                continue
            # 進行resize
            results[key] = results[key].resize(out_size)
            # 獲取mask資料
            masks = results[key].masks
            processed_masks = []
            # 遍歷所有標註點
            for mask in masks:
                square_pad_mask = self.square_pad_mask(mask[0], offset)
                # 將調整過後的標註點保存
                processed_masks.append([square_pad_mask])

            # 將點放入到PolygonMasks當中
            results[key] = PolygonMasks(processed_masks, *(img.shape[:2]))

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomScaling:

    def __init__(self, size=800, scale=(3. / 4, 5. / 2)):
        """Random scale the image while keeping aspect.

        Args:
            size (int) : Base size before scaling.
            scale (tuple(float)) : The range of scaling.
        """
        assert isinstance(size, int)
        assert isinstance(scale, float) or isinstance(scale, tuple)
        self.size = size
        self.scale = scale if isinstance(scale, tuple) \
            else (1 - scale, 1 + scale)

    def __call__(self, results):
        image = results['img']
        h, w, _ = results['img_shape']

        aspect_ratio = np.random.uniform(min(self.scale), max(self.scale))
        scales = self.size * 1.0 / max(h, w) * aspect_ratio
        scales = np.array([scales, scales])
        out_size = (int(h * scales[1]), int(w * scales[0]))
        image = mmcv.imresize(image, out_size[::-1])

        results['img'] = image
        results['img_shape'] = image.shape

        for key in results.get('mask_fields', []):
            if len(results[key].masks) == 0:
                continue
            results[key] = results[key].resize(out_size)

        return results


@PIPELINES.register_module()
class RandomCropFlip:

    def __init__(self,
                 pad_ratio=0.1,
                 crop_ratio=0.5,
                 iter_num=1,
                 min_area_ratio=0.2):
        """Random crop and flip a patch of the image.

        Args:
            crop_ratio (float): The ratio of cropping.
            iter_num (int): Number of operations.
            min_area_ratio (float): Minimal area ratio between cropped patch
                and original image.
        """
        assert isinstance(crop_ratio, float)
        assert isinstance(iter_num, int)
        assert isinstance(min_area_ratio, float)

        self.pad_ratio = pad_ratio
        self.epsilon = 1e-2
        self.crop_ratio = crop_ratio
        self.iter_num = iter_num
        self.min_area_ratio = min_area_ratio

    def __call__(self, results):
        for i in range(self.iter_num):
            results = self.random_crop_flip(results)
        return results

    def random_crop_flip(self, results):
        image = results['img']
        polygons = results['gt_masks'].masks
        ignore_polygons = results['gt_masks_ignore'].masks
        all_polygons = polygons + ignore_polygons
        if len(polygons) == 0:
            return results

        if np.random.random() >= self.crop_ratio:
            return results

        h, w, _ = results['img_shape']
        area = h * w
        pad_h = int(h * self.pad_ratio)
        pad_w = int(w * self.pad_ratio)
        h_axis, w_axis = self.generate_crop_target(image, all_polygons, pad_h,
                                                   pad_w)
        if len(h_axis) == 0 or len(w_axis) == 0:
            return results

        attempt = 0
        while attempt < 10:
            attempt += 1
            polys_keep = []
            polys_new = []
            ign_polys_keep = []
            ign_polys_new = []
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            if (xmax - xmin) * (ymax - ymin) < area * self.min_area_ratio:
                # area too small
                continue

            pts = np.stack([[xmin, xmax, xmax, xmin],
                            [ymin, ymin, ymax, ymax]]).T.astype(np.int32)
            pp = plg(pts)
            fail_flag = False
            for polygon in polygons:
                ppi = plg(polygon[0].reshape(-1, 2))
                ppiou = eval_utils.poly_intersection(ppi, pp)
                if np.abs(ppiou - float(ppi.area)) > self.epsilon and \
                        np.abs(ppiou) > self.epsilon:
                    fail_flag = True
                    break
                elif np.abs(ppiou - float(ppi.area)) < self.epsilon:
                    polys_new.append(polygon)
                else:
                    polys_keep.append(polygon)

            for polygon in ignore_polygons:
                ppi = plg(polygon[0].reshape(-1, 2))
                ppiou = eval_utils.poly_intersection(ppi, pp)
                if np.abs(ppiou - float(ppi.area)) > self.epsilon and \
                        np.abs(ppiou) > self.epsilon:
                    fail_flag = True
                    break
                elif np.abs(ppiou - float(ppi.area)) < self.epsilon:
                    ign_polys_new.append(polygon)
                else:
                    ign_polys_keep.append(polygon)

            if fail_flag:
                continue
            else:
                break

        cropped = image[ymin:ymax, xmin:xmax, :]
        select_type = np.random.randint(3)
        if select_type == 0:
            img = np.ascontiguousarray(cropped[:, ::-1])
        elif select_type == 1:
            img = np.ascontiguousarray(cropped[::-1, :])
        else:
            img = np.ascontiguousarray(cropped[::-1, ::-1])
        image[ymin:ymax, xmin:xmax, :] = img
        results['img'] = image

        if len(polys_new) + len(ign_polys_new) != 0:
            height, width, _ = cropped.shape
            if select_type == 0:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon[0].reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    polys_new[idx] = [poly.reshape(-1, )]
                for idx, polygon in enumerate(ign_polys_new):
                    poly = polygon[0].reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    ign_polys_new[idx] = [poly.reshape(-1, )]
            elif select_type == 1:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon[0].reshape(-1, 2)
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    polys_new[idx] = [poly.reshape(-1, )]
                for idx, polygon in enumerate(ign_polys_new):
                    poly = polygon[0].reshape(-1, 2)
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    ign_polys_new[idx] = [poly.reshape(-1, )]
            else:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon[0].reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    polys_new[idx] = [poly.reshape(-1, )]
                for idx, polygon in enumerate(ign_polys_new):
                    poly = polygon[0].reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    ign_polys_new[idx] = [poly.reshape(-1, )]
            polygons = polys_keep + polys_new
            ignore_polygons = ign_polys_keep + ign_polys_new
            results['gt_masks'] = PolygonMasks(polygons, *(image.shape[:2]))
            results['gt_masks_ignore'] = PolygonMasks(ignore_polygons,
                                                      *(image.shape[:2]))

        return results

    def generate_crop_target(self, image, all_polys, pad_h, pad_w):
        """Generate crop target and make sure not to crop the polygon
        instances.

        Args:
            image (ndarray): The image waited to be crop.
            all_polys (list[list[ndarray]]): All polygons including ground
                truth polygons and ground truth ignored polygons.
            pad_h (int): Padding length of height.
            pad_w (int): Padding length of width.
        Returns:
            h_axis (ndarray): Vertical cropping range.
            w_axis (ndarray): Horizontal cropping range.
        """
        h, w, _ = image.shape
        h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
        w_array = np.zeros((w + pad_w * 2), dtype=np.int32)

        text_polys = []
        for polygon in all_polys:
            rect = cv2.minAreaRect(polygon[0].astype(np.int32).reshape(-1, 2))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            text_polys.append([box[0], box[1], box[2], box[3]])

        polys = np.array(text_polys, dtype=np.int32)
        for poly in polys:
            poly = np.round(poly, decimals=0).astype(np.int32)
            minx = np.min(poly[:, 0])
            maxx = np.max(poly[:, 0])
            w_array[minx + pad_w:maxx + pad_w] = 1
            miny = np.min(poly[:, 1])
            maxy = np.max(poly[:, 1])
            h_array[miny + pad_h:maxy + pad_h] = 1

        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        return h_axis, w_axis


@PIPELINES.register_module()
class PyramidRescale:
    """Resize the image to the base shape, downsample it with gaussian pyramid,
    and rescale it back to original size.

    Adapted from https://github.com/FangShancheng/ABINet.

    Args:
        factor (int): The decay factor from base size, or the number of
            downsampling operations from the base layer.
        base_shape (tuple(int)): The shape of the base layer of the pyramid.
        randomize_factor (bool): If True, the final factor would be a random
            integer in [0, factor].

    :Required Keys:
        - | ``img`` (ndarray): The input image.

    :Affected Keys:
        :Modified:
            - | ``img`` (ndarray): The modified image.
    """

    def __init__(self, factor=4, base_shape=(128, 512), randomize_factor=True):
        assert isinstance(factor, int)
        assert isinstance(base_shape, list) or isinstance(base_shape, tuple)
        assert len(base_shape) == 2
        assert isinstance(randomize_factor, bool)
        self.factor = factor if not randomize_factor else np.random.randint(
            0, factor + 1)
        self.base_w, self.base_h = base_shape

    def __call__(self, results):
        assert 'img' in results
        if self.factor == 0:
            return results
        img = results['img']
        src_h, src_w = img.shape[:2]
        scale_img = mmcv.imresize(img, (self.base_w, self.base_h))
        for _ in range(self.factor):
            scale_img = cv2.pyrDown(scale_img)
        scale_img = mmcv.imresize(scale_img, (src_w, src_h))
        results['img'] = scale_img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(factor={self.factor}, '
        repr_str += f'basew={self.basew}, baseh={self.baseh})'
        return repr_str
