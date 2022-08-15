# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings
from collections.abc import Sequence

import cv2
import mmcv
import numpy as np
from mmcv.utils import digit_version
from torch.nn.modules.utils import _pair

from ..builder import PIPELINES
from .formatting import to_tensor


def _combine_quadruple(a, b):
    return (a[0] + a[2] * b[0], a[1] + a[3] * b[1], a[2] * b[2], a[3] * b[3])


def _flip_quadruple(a):
    return (1 - a[0] - a[2], a[1], a[2], a[3])


def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    # 已看過，根據當前的資料進行合適的懶處理初始化

    if 'img_shape' not in results:
        # 如果在results2當中沒有img_shape參數就會到這裡補上，根據當前圖像的大小填上
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        # 如果要使用lazy標籤就會到這裡
        if 'lazy' not in results:
            # 如果results當中沒有lazy的key就會到這裡
            # 獲取當前圖像的高寬
            img_h, img_w = results['img_shape']
            # 構建lazy operation的字典
            lazyop = dict()
            # 在lazy當中添加上原始圖像的大小資訊
            lazyop['original_shape'] = results['img_shape']
            # 預設裁切的box就會是整張圖
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            # 將翻轉關閉
            lazyop['flip'] = False
            # 同時翻轉方向設定成None
            lazyop['flip_direction'] = None
            # 差值方式也是None
            lazyop['interpolation'] = None
            # 將lazy的字典放到result的lazy下
            results['lazy'] = lazyop
    else:
        # 沒有要使用lazy就會到這裡，檢查results當中有沒有lazy，正常來說不應該有
        assert 'lazy' not in results, 'Use Fuse after lazy operations'


@PIPELINES.register_module()
class TorchvisionTrans:
    """Torchvision Augmentations, under torchvision.transforms.

    Args:
        type (str): The name of the torchvision transformation.
    """

    def __init__(self, type, **kwargs):
        try:
            import torchvision
            import torchvision.transforms as tv_trans
        except ImportError:
            raise RuntimeError('Install torchvision to use TorchvisionTrans')
        if digit_version(torchvision.__version__) < digit_version('0.8.0'):
            raise RuntimeError('The version of torchvision should be at least '
                               '0.8.0')

        trans = getattr(tv_trans, type, None)
        assert trans, f'Transform {type} not in torchvision'
        self.trans = trans(**kwargs)

    def __call__(self, results):
        assert 'imgs' in results

        imgs = [x.transpose(2, 0, 1) for x in results['imgs']]
        imgs = to_tensor(np.stack(imgs))

        imgs = self.trans(imgs).data.numpy()
        imgs[imgs > 255] = 255
        imgs[imgs < 0] = 0
        imgs = imgs.astype(np.uint8)
        imgs = [x.transpose(1, 2, 0) for x in imgs]
        results['imgs'] = imgs
        return results


@PIPELINES.register_module()
class PytorchVideoTrans:
    """PytorchVideoTrans Augmentations, under pytorchvideo.transforms.

    Args:
        type (str): The name of the pytorchvideo transformation.
    """

    def __init__(self, type, **kwargs):
        try:
            import pytorchvideo.transforms as ptv_trans
            import torch
        except ImportError:
            raise RuntimeError('Install pytorchvideo to use PytorchVideoTrans')
        if digit_version(torch.__version__) < digit_version('1.8.0'):
            raise RuntimeError(
                'The version of PyTorch should be at least 1.8.0')

        trans = getattr(ptv_trans, type, None)
        assert trans, f'Transform {type} not in pytorchvideo'

        supported_pytorchvideo_trans = ('AugMix', 'RandAugment',
                                        'RandomResizedCrop', 'ShortSideScale',
                                        'RandomShortSideScale')
        assert type in supported_pytorchvideo_trans,\
            f'PytorchVideo Transform {type} is not supported in MMAction2'

        self.trans = trans(**kwargs)
        self.type = type

    def __call__(self, results):
        assert 'imgs' in results

        assert 'gt_bboxes' not in results,\
            f'PytorchVideo {self.type} doesn\'t support bboxes yet.'
        assert 'proposals' not in results,\
            f'PytorchVideo {self.type} doesn\'t support bboxes yet.'

        if self.type in ('AugMix', 'RandAugment'):
            # list[ndarray(h, w, 3)] -> torch.tensor(t, c, h, w)
            imgs = [x.transpose(2, 0, 1) for x in results['imgs']]
            imgs = to_tensor(np.stack(imgs))
        else:
            # list[ndarray(h, w, 3)] -> torch.tensor(c, t, h, w)
            # uint8 -> float32
            imgs = to_tensor((np.stack(results['imgs']).transpose(3, 0, 1, 2) /
                              255.).astype(np.float32))

        imgs = self.trans(imgs).data.numpy()

        if self.type in ('AugMix', 'RandAugment'):
            imgs[imgs > 255] = 255
            imgs[imgs < 0] = 0
            imgs = imgs.astype(np.uint8)

            # torch.tensor(t, c, h, w) -> list[ndarray(h, w, 3)]
            imgs = [x.transpose(1, 2, 0) for x in imgs]
        else:
            # float32 -> uint8
            imgs = imgs * 255
            imgs[imgs > 255] = 255
            imgs[imgs < 0] = 0
            imgs = imgs.astype(np.uint8)

            # torch.tensor(c, t, h, w) -> list[ndarray(h, w, 3)]
            imgs = [x for x in imgs.transpose(1, 2, 3, 0)]

        results['imgs'] = imgs

        return results


@PIPELINES.register_module()
class PoseCompact:
    """Convert the coordinates of keypoints to make it more compact.
    Specifically, it first find a tight bounding box that surrounds all joints
    in each frame, then we expand the tight box by a given padding ratio. For
    example, if 'padding == 0.25', then the expanded box has unchanged center,
    and 1.25x width and height.

    Required keys in results are "img_shape", "keypoint", add or modified keys
    are "img_shape", "keypoint", "crop_quadruple".

    Args:
        padding (float): The padding size. Default: 0.25.
        threshold (int): The threshold for the tight bounding box. If the width
            or height of the tight bounding box is smaller than the threshold,
            we do not perform the compact operation. Default: 10.
        hw_ratio (float | tuple[float] | None): The hw_ratio of the expanded
            box. Float indicates the specific ratio and tuple indicates a
            ratio range. If set as None, it means there is no requirement on
            hw_ratio. Default: None.
        allow_imgpad (bool): Whether to allow expanding the box outside the
            image to meet the hw_ratio requirement. Default: True.

    Returns:
        type: Description of returned object.
    """

    def __init__(self,
                 padding=0.25,
                 threshold=10,
                 hw_ratio=None,
                 allow_imgpad=True):

        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = _pair(hw_ratio)

        self.hw_ratio = hw_ratio

        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def __call__(self, results):
        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']

        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return results

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)

        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape

        # the order is x, y, w, h (in [0, 1]), a tuple
        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        new_crop_quadruple = (min_x / w, min_y / h, (max_x - min_x) / w,
                              (max_y - min_y) / h)
        crop_quadruple = _combine_quadruple(crop_quadruple, new_crop_quadruple)
        results['crop_quadruple'] = crop_quadruple
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(padding={self.padding}, '
                    f'threshold={self.threshold}, '
                    f'hw_ratio={self.hw_ratio}, '
                    f'allow_imgpad={self.allow_imgpad})')
        return repr_str


@PIPELINES.register_module()
class Imgaug:
    """Imgaug augmentation.

    Adds custom transformations from imgaug library.
    Please visit `https://imgaug.readthedocs.io/en/latest/index.html`
    to get more information. Two demo configs could be found in tsn and i3d
    config folder.

    It's better to use uint8 images as inputs since imgaug works best with
    numpy dtype uint8 and isn't well tested with other dtypes. It should be
    noted that not all of the augmenters have the same input and output dtype,
    which may cause unexpected results.

    Required keys are "imgs", "img_shape"(if "gt_bboxes" is not None) and
    "modality", added or modified keys are "imgs", "img_shape", "gt_bboxes"
    and "proposals".

    It is worth mentioning that `Imgaug` will NOT create custom keys like
    "interpolation", "crop_bbox", "flip_direction", etc. So when using
    `Imgaug` along with other mmaction2 pipelines, we should pay more attention
    to required keys.

    Two steps to use `Imgaug` pipeline:
    1. Create initialization parameter `transforms`. There are three ways
        to create `transforms`.
        1) string: only support `default` for now.
            e.g. `transforms='default'`
        2) list[dict]: create a list of augmenters by a list of dicts, each
            dict corresponds to one augmenter. Every dict MUST contain a key
            named `type`. `type` should be a string(iaa.Augmenter's name) or
            an iaa.Augmenter subclass.
            e.g. `transforms=[dict(type='Rotate', rotate=(-20, 20))]`
            e.g. `transforms=[dict(type=iaa.Rotate, rotate=(-20, 20))]`
        3) iaa.Augmenter: create an imgaug.Augmenter object.
            e.g. `transforms=iaa.Rotate(rotate=(-20, 20))`
    2. Add `Imgaug` in dataset pipeline. It is recommended to insert imgaug
        pipeline before `Normalize`. A demo pipeline is listed as follows.
        ```
        pipeline = [
            dict(
                type='SampleFrames',
                clip_len=1,
                frame_interval=1,
                num_clips=16,
            ),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(
                type='MultiScaleCrop',
                input_size=224,
                scales=(1, 0.875, 0.75, 0.66),
                random_crop=False,
                max_wh_scale_gap=1,
                num_fixed_crops=13),
            dict(type='Resize', scale=(224, 224), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5),
            dict(type='Imgaug', transforms='default'),
            # dict(type='Imgaug', transforms=[
            #     dict(type='Rotate', rotate=(-20, 20))
            # ]),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCHW'),
            dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'label'])
        ]
        ```

    Args:
        transforms (str | list[dict] | :obj:`iaa.Augmenter`): Three different
            ways to create imgaug augmenter.
    """

    def __init__(self, transforms):
        import imgaug.augmenters as iaa

        if transforms == 'default':
            self.transforms = self.default_transforms()
        elif isinstance(transforms, list):
            assert all(isinstance(trans, dict) for trans in transforms)
            self.transforms = transforms
        elif isinstance(transforms, iaa.Augmenter):
            self.aug = self.transforms = transforms
        else:
            raise ValueError('transforms must be `default` or a list of dicts'
                             ' or iaa.Augmenter object')

        if not isinstance(transforms, iaa.Augmenter):
            self.aug = iaa.Sequential(
                [self.imgaug_builder(t) for t in self.transforms])

    @staticmethod
    def default_transforms():
        """Default transforms for imgaug.

        Implement RandAugment by imgaug.
        Please visit `https://arxiv.org/abs/1909.13719` for more information.

        Augmenters and hyper parameters are borrowed from the following repo:
        https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py # noqa

        Miss one augmenter ``SolarizeAdd`` since imgaug doesn't support this.

        Returns:
            dict: The constructed RandAugment transforms.
        """
        # RandAugment hyper params
        num_augmenters = 2
        cur_magnitude, max_magnitude = 9, 10
        cur_level = 1.0 * cur_magnitude / max_magnitude

        return [
            dict(
                type='SomeOf',
                n=num_augmenters,
                children=[
                    dict(
                        type='ShearX',
                        shear=17.19 * cur_level * random.choice([-1, 1])),
                    dict(
                        type='ShearY',
                        shear=17.19 * cur_level * random.choice([-1, 1])),
                    dict(
                        type='TranslateX',
                        percent=.2 * cur_level * random.choice([-1, 1])),
                    dict(
                        type='TranslateY',
                        percent=.2 * cur_level * random.choice([-1, 1])),
                    dict(
                        type='Rotate',
                        rotate=30 * cur_level * random.choice([-1, 1])),
                    dict(type='Posterize', nb_bits=max(1, int(4 * cur_level))),
                    dict(type='Solarize', threshold=256 * cur_level),
                    dict(type='EnhanceColor', factor=1.8 * cur_level + .1),
                    dict(type='EnhanceContrast', factor=1.8 * cur_level + .1),
                    dict(
                        type='EnhanceBrightness', factor=1.8 * cur_level + .1),
                    dict(type='EnhanceSharpness', factor=1.8 * cur_level + .1),
                    dict(type='Autocontrast', cutoff=0),
                    dict(type='Equalize'),
                    dict(type='Invert', p=1.),
                    dict(
                        type='Cutout',
                        nb_iterations=1,
                        size=0.2 * cur_level,
                        squared=True)
                ])
        ]

    def imgaug_builder(self, cfg):
        """Import a module from imgaug.

        It follows the logic of :func:`build_from_cfg`. Use a dict object to
        create an iaa.Augmenter object.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj:`iaa.Augmenter`: The constructed imgaug augmenter.
        """
        import imgaug.augmenters as iaa

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            obj_cls = getattr(iaa, obj_type) if hasattr(iaa, obj_type) \
                else getattr(iaa.pillike, obj_type)
        elif issubclass(obj_type, iaa.Augmenter):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'children' in args:
            args['children'] = [
                self.imgaug_builder(child) for child in args['children']
            ]

        return obj_cls(**args)

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.aug})'
        return repr_str

    def __call__(self, results):
        assert results['modality'] == 'RGB', 'Imgaug only support RGB images.'
        in_type = results['imgs'][0].dtype.type

        cur_aug = self.aug.to_deterministic()

        results['imgs'] = [
            cur_aug.augment_image(frame) for frame in results['imgs']
        ]
        img_h, img_w, _ = results['imgs'][0].shape

        out_type = results['imgs'][0].dtype.type
        assert in_type == out_type, \
            ('Imgaug input dtype and output dtype are not the same. ',
             f'Convert from {in_type} to {out_type}')

        if 'gt_bboxes' in results:
            from imgaug.augmentables import bbs
            bbox_list = [
                bbs.BoundingBox(
                    x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
                for bbox in results['gt_bboxes']
            ]
            bboxes = bbs.BoundingBoxesOnImage(
                bbox_list, shape=results['img_shape'])
            bbox_aug, *_ = cur_aug.augment_bounding_boxes([bboxes])
            results['gt_bboxes'] = [[
                max(bbox.x1, 0),
                max(bbox.y1, 0),
                min(bbox.x2, img_w),
                min(bbox.y2, img_h)
            ] for bbox in bbox_aug.items]
            if 'proposals' in results:
                bbox_list = [
                    bbs.BoundingBox(
                        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
                    for bbox in results['proposals']
                ]
                bboxes = bbs.BoundingBoxesOnImage(
                    bbox_list, shape=results['img_shape'])
                bbox_aug, *_ = cur_aug.augment_bounding_boxes([bboxes])
                results['proposals'] = [[
                    max(bbox.x1, 0),
                    max(bbox.y1, 0),
                    min(bbox.x2, img_w),
                    min(bbox.y2, img_h)
                ] for bbox in bbox_aug.items]

        results['img_shape'] = (img_h, img_w)

        return results


@PIPELINES.register_module()
class Fuse:
    """Fuse lazy operations.

    Fusion order:
        crop -> resize -> flip

    Required keys are "imgs", "img_shape" and "lazy", added or modified keys
    are "imgs", "lazy".
    Required keys in "lazy" are "crop_bbox", "interpolation", "flip_direction".
    """

    def __call__(self, results):
        if 'lazy' not in results:
            raise ValueError('No lazy operation detected')
        lazyop = results['lazy']
        imgs = results['imgs']

        # crop
        left, top, right, bottom = lazyop['crop_bbox'].round().astype(int)
        imgs = [img[top:bottom, left:right] for img in imgs]

        # resize
        img_h, img_w = results['img_shape']
        if lazyop['interpolation'] is None:
            interpolation = 'bilinear'
        else:
            interpolation = lazyop['interpolation']
        imgs = [
            mmcv.imresize(img, (img_w, img_h), interpolation=interpolation)
            for img in imgs
        ]

        # flip
        if lazyop['flip']:
            for img in imgs:
                mmcv.imflip_(img, lazyop['flip_direction'])

        results['imgs'] = imgs
        del results['lazy']

        return results


@PIPELINES.register_module()
class RandomCrop:
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "img_shape", "keypoint" (optional), "imgs"
    (optional), added or modified keys are "keypoint", "imgs", "lazy"; Required
    keys in "lazy" are "flip", "crop_bbox", added or modified key is
    "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        return kps - crop_bbox[:2]

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        # 已看過，根據傳入的crop_bbox對imgs進行裁切
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox.

        Args:
            box (np.ndarray): The bounding boxes.
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """

        x1, y1, x2, y2 = crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        box_ = box.copy()
        box_[..., 0::2] = np.clip(box[..., 0::2] - x1, 0, img_w - 1)
        box_[..., 1::2] = np.clip(box[..., 1::2] - y1, 0, img_h - 1)
        return box_

    def _all_box_crop(self, results, crop_bbox):
        """Crop the gt_bboxes and proposals in results according to crop_bbox.

        Args:
            results (dict): All information about the sample, which contain
                'gt_bboxes' and 'proposals' (optional).
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """
        results['gt_bboxes'] = self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[1] == 4
            results['proposals'] = self._box_crop(results['proposals'],
                                                  crop_bbox)
        return results

    def __call__(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = self.size, self.size

        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox

        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        # Process entity boxes
        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class RandomResizedCrop(RandomCrop):
    """Random crop that specifics the area and height-weight ratio range.

    Required keys in results are "img_shape", "crop_bbox", "imgs" (optional),
    "keypoint" (optional), added or modified keys are "imgs", "keypoint",
    "crop_bbox" and "lazy"; Required keys in "lazy" are "flip", "crop_bbox",
    added or modified key is "crop_bbox".

    Args:
        area_range (Tuple[float]): The candidate area scales range of
            output cropped images. Default: (0.08, 1.0).
        aspect_ratio_range (Tuple[float]): The candidate aspect ratio range of
            output cropped images. Default: (3 / 4, 4 / 3).
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3),
                 lazy=False):
        self.area_range = area_range
        self.aspect_ratio_range = aspect_ratio_range
        self.lazy = lazy
        if not mmcv.is_tuple_of(self.area_range, float):
            raise TypeError(f'Area_range must be a tuple of float, '
                            f'but got {type(area_range)}')
        if not mmcv.is_tuple_of(self.aspect_ratio_range, float):
            raise TypeError(f'Aspect_ratio_range must be a tuple of float, '
                            f'but got {type(aspect_ratio_range)}')

    @staticmethod
    def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

    def __call__(self, results):
        """Performs the RandomResizeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']

        left, top, right, bottom = self.get_crop_bbox(
            (img_h, img_w), self.area_range, self.aspect_ratio_range)
        new_h, new_w = bottom - top, right - left

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'area_range={self.area_range}, '
                    f'aspect_ratio_range={self.aspect_ratio_range}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class MultiScaleCrop(RandomCrop):
    """Crop images with a list of randomly selected scales.

    Randomly select the w and h scales from a list of scales. Scale of 1 means
    the base size, which is the minimal of image width and height. The scale
    level of w and h is controlled to be smaller than a certain value to
    prevent too large or small aspect ratio.

    Required keys are "img_shape", "imgs" (optional), "keypoint" (optional),
    added or modified keys are "imgs", "crop_bbox", "img_shape", "lazy" and
    "scales". Required keys in "lazy" are "crop_bbox", added or modified key is
    "crop_bbox".

    Args:
        input_size (int | tuple[int]): (w, h) of network input.
        scales (tuple[float]): width and height scales to be selected.
        max_wh_scale_gap (int): Maximum gap of w and h scale levels.
            Default: 1.
        random_crop (bool): If set to True, the cropping bbox will be randomly
            sampled, otherwise it will be sampler from fixed regions.
            Default: False.
        num_fixed_crops (int): If set to 5, the cropping bbox will keep 5
            basic fixed regions: "upper left", "upper right", "lower left",
            "lower right", "center". If set to 13, the cropping bbox will
            append another 8 fix regions: "center left", "center right",
            "lower center", "upper center", "upper left quarter",
            "upper right quarter", "lower left quarter", "lower right quarter".
            Default: 5.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 input_size,
                 scales=(1, ),
                 max_wh_scale_gap=1,
                 random_crop=False,
                 num_fixed_crops=5,
                 lazy=False):
        """ 已看過，將圖像進行剪裁，根據傳入的scale
        Args:
            input_size: 傳入到網路的圖像大小
            scales: 可選擇的高寬比
            max_wh_scale_gap: 最大高寬比的差距
            random_crop: 如果啟用就會隨機裁切，如果不啟用就會在指定地方進行裁切
            num_fixed_crops: 須保留的地方，詳細看英文解說
            lazy: 是否啟用lazy操作
        """
        # 將input_size變成tuple型態 = tuple(int, int)
        self.input_size = _pair(input_size)
        if not mmcv.is_tuple_of(self.input_size, int):
            # 如果當前的input_size不是tuple(int, int)型態就會報錯
            raise TypeError(f'Input_size must be int or tuple of int, '
                            f'but got {type(input_size)}')

        if not isinstance(scales, tuple):
            # 如果scales不是tuple型態就會報錯
            raise TypeError(f'Scales must be tuple, but got {type(scales)}')

        if num_fixed_crops not in [5, 13]:
            # num_fixed_crops就只有設定成5或是13兩種模式，其他種就會報錯
            raise ValueError(f'Num_fix_crops must be in {[5, 13]}, '
                             f'but got {num_fixed_crops}')

        # 保存傳入資料
        self.scales = scales
        self.max_wh_scale_gap = max_wh_scale_gap
        self.random_crop = random_crop
        self.num_fixed_crops = num_fixed_crops
        self.lazy = lazy

    def __call__(self, results):
        """Performs the MultiScaleCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # 已看過，進行多種大小的剪裁
        # 設定lazy資訊，如果有使用lazy的話
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            # 如果有關節點檢測就不可以使用lazy operation，如果啟用就會報錯
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        # 獲取當前圖像的高寬資訊
        img_h, img_w = results['img_shape']
        # 基底的大小就會是當前高寬較短的地方
        base_size = min(img_h, img_w)
        # 這裡會給出裁切的大小，會是基礎大小乘上scales
        crop_sizes = [int(base_size * s) for s in self.scales]

        # 構建候選大小的list
        candidate_sizes = []
        # 遍歷crop_sizes
        for i, h in enumerate(crop_sizes):
            # 遍歷crop_sizes
            for j, w in enumerate(crop_sizes):
                if abs(i - j) <= self.max_wh_scale_gap:
                    # 當遍歷到的兩個index之間距離小於等於max_wh_scale_gap就會進來
                    # 且會設定成候選的高寬
                    candidate_sizes.append([w, h])

        # 從候選的高寬隨機選則作為需要裁切的大小，crop_size shape = [height, width]
        crop_size = random.choice(candidate_sizes)
        for i in range(2):
            # 如果crop_size與設定的input_size相差小於3就直接以input_size代替
            if abs(crop_size[i] - self.input_size[i]) < 3:
                crop_size[i] = self.input_size[i]

        # 獲取剪裁後的圖像大小
        crop_w, crop_h = crop_size

        if self.random_crop:
            # 如果是透過隨機剪裁就會到這裡
            # 選取一個[0, img_w - crop_w]的值作為x方向的偏移
            x_offset = random.randint(0, img_w - crop_w)
            # 選取一個[0, img_h - crop_h]的值作為y方向的偏移
            y_offset = random.randint(0, img_h - crop_h)
        else:
            # 如果不是隨機剪裁就會到這裡
            # 獲取可以偏移量的四分之一
            w_step = (img_w - crop_w) // 4
            h_step = (img_h - crop_h) // 4
            # 獲取剪裁的候選項
            candidate_offsets = [
                (0, 0),  # upper left，右上角
                (4 * w_step, 0),  # upper right，左上角
                (0, 4 * h_step),  # lower left，左下角
                (4 * w_step, 4 * h_step),  # lower right，右下角
                (2 * w_step, 2 * h_step),  # center，中心點
            ]
            if self.num_fixed_crops == 13:
                # 如果num_fixed_crops是13個點就會進來，有更詳細的額外候選偏移
                extra_candidate_offsets = [
                    (0, 2 * h_step),  # center left
                    (4 * w_step, 2 * h_step),  # center right
                    (2 * w_step, 4 * h_step),  # lower center
                    (2 * w_step, 0 * h_step),  # upper center
                    (1 * w_step, 1 * h_step),  # upper left quarter
                    (3 * w_step, 1 * h_step),  # upper right quarter
                    (1 * w_step, 3 * h_step),  # lower left quarter
                    (3 * w_step, 3 * h_step)  # lower right quarter
                ]
                # 將額外的點放到candidate_offsets當中
                candidate_offsets.extend(extra_candidate_offsets)
            # 從候選點當中隨機選取一組作為偏移量
            x_offset, y_offset = random.choice(candidate_offsets)

        # 新的高寬就會是crop_h與crop_w
        new_h, new_w = crop_h, crop_w

        # 構建選取範圍的矩形匡，這裏面的範圍就是我們需要裁切後留下的圖像面積
        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        # 將裁切匡保存到results當中
        results['crop_bbox'] = crop_bbox
        # 更新當前圖像大小為裁切後大小
        results['img_shape'] = (new_h, new_w)
        # 更新scales
        results['scales'] = self.scales

        if 'crop_quadruple' not in results:
            # 如果results當中沒有crop_quadruple就會到這裡，新增上去，預設為[0, 0, 1, 1]
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        # 獲取x與y的比例
        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        # 獲取高寬經過crop後與原先圖像的縮放比例
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        # 獲取crop_quadruple資訊，這裡如果一開始沒有預設會是[0, 0, 1, 1]
        old_crop_quadruple = results['crop_quadruple']
        # 獲取舊的x與y的比例
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        # 獲取舊的w與h的比例
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        # 更新quadruple資訊
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio, old_y_ratio + y_ratio * old_h_ratio,
            w_ratio * old_w_ratio, h_ratio * old_h_ratio
        ]
        # 將quadruple資訊更新到results當中
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        if not self.lazy:
            # 如果沒有使用lazy operation就會到這裡
            if 'keypoint' in results:
                # 將關節點部分透過_crop_kps與惡crop_bbox進行剪裁
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                # 將圖像透過_crop_imgs與指定的crop_bbox進行剪裁
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            # 如果有使用lazy operation就會到這裡
            # 獲取lazy的字典資訊
            lazyop = results['lazy']
            if lazyop['flip']:
                # 在lazy當中的flip需要是None
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            # 更新lazy當中的crop_bbox資訊
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            # 如果results當中有目標檢測的標註匡就會到這裡
            # 有目標檢測的標註匡就不支援lazy模式
            assert not self.lazy
            # 使用_all_box_crop進行標註匡剪裁
            results = self._all_box_crop(results, results['crop_bbox'])

        # 回傳更新後的results資訊
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'input_size={self.input_size}, scales={self.scales}, '
                    f'max_wh_scale_gap={self.max_wh_scale_gap}, '
                    f'random_crop={self.random_crop}, '
                    f'num_fixed_crops={self.num_fixed_crops}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Resize:
    """Resize images to a specific size.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "lazy", "resize_size". Required keys in "lazy" is None,
    added or modified key is "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False):
        """ 已看過，將圖像進行resize到指定的大小
        Args:
            scale: 如果keep_ratio是True，這裡的值表示的就會是縮放比例或是最大大小
                   如果是float型態，圖像就會縮放指定的float倍率
                   如果是tuple且是兩個int構成，圖像就會盡可能放大到給定的範圍內
                   否則就是直接指定最後圖像大小
            keep_ratio: 是否需要保留圖像長寬比
            interpolation: 差值方式
            lazy: 是否使用lazy操作
        """
        if isinstance(scale, float):
            # 如果傳入的scale是float格式就會到這裡
            if scale <= 0:
                # 如果scale的值小於0就會報錯
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            # 如果傳入的scale是tuple就會到這裡
            # 提取出最大的邊的長度
            max_long_edge = max(scale)
            # 提取出最小的邊的長度
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # 如果短編設定的是-1就會到這裡
                # assign np.inf to long edge for rescaling short edge later.
                # 將最大邊設定成無窮，短邊設定為傳入的最大邊
                scale = (np.inf, max_long_edge)
        else:
            # 其他的情況就會直接報錯
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        # 保存傳入的參數
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def _resize_imgs(self, imgs, new_w, new_h):
        """ 已看過，進行圖像的resize
        Args:
            imgs: 圖像資料，list[ndarray]，list的長度就是照片數量
            new_w: 新圖像的寬度
            new_h: 新圖像的高度
        """
        return [
            # 透過imresize進行高寬調整
            mmcv.imresize(
                img, (new_w, new_h), interpolation=self.interpolation)
            for img in imgs
        ]

    @staticmethod
    def _resize_kps(kps, scale_factor):
        return kps * scale_factor

    @staticmethod
    def _box_resize(box, scale_factor):
        """Rescale the bounding boxes according to the scale_factor.

        Args:
            box (np.ndarray): The bounding boxes.
            scale_factor (np.ndarray): The scale factor used for rescaling.
        """
        assert len(scale_factor) == 2
        scale_factor = np.concatenate([scale_factor, scale_factor])
        return box * scale_factor

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # 已看過，進行resize的數據增強

        # 進行lazy的初始化，如果適合的話就會初始化
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            # 如果當前有需要關節點檢測就不允許使用lazy，如果有使用到就會報錯
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        if 'scale_factor' not in results:
            # 如果results當中沒有scale_factor參數就會到這裡，將scale_factor設定成[1., 1.]
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        # 獲取當前圖像的高寬
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            # 如果有需要保持高寬比的resize就會到這裡，會將當前圖像的長邊控制在最長邊以內，將當前圖像短邊控制在最短邊以內
            new_w, new_h = mmcv.rescale_size((img_w, img_h), self.scale)
        else:
            # 否則scale就會是新的高寬
            new_w, new_h = self.scale

        # 獲取圖像在高寬上面縮放比例，這個對於關節點之類的標註會有效果，可以直接透過縮放比例調整新的標註位置
        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        # 更新當前圖像大小
        results['img_shape'] = (new_h, new_w)
        # 記錄下是否有保持高寬比
        results['keep_ratio'] = self.keep_ratio
        # 縮放比例會是原先的縮放比例乘上現在的縮放比例，因為縮放比例是可以連乘的，像是如果有經過兩次resize那麼最終圖像的縮放比例就是兩次的相乘
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            # 如果沒有使用lazy operation就會到這裡
            if 'imgs' in results:
                # 如果results當中有圖像資料就根據指定的大小進行resize
                results['imgs'] = self._resize_imgs(results['imgs'], new_w,
                                                    new_h)
            if 'keypoint' in results:
                # 如果results當中有關節點資訊就根據縮放比例進行調整
                results['keypoint'] = self._resize_kps(results['keypoint'],
                                                       self.scale_factor)
        else:
            # 如果有使用lazy operation就會到這裡
            # 將lazy的字典取出，這裡會有lazy的相關配置
            lazyop = results['lazy']
            if lazyop['flip']:
                # 這裡的flip應該要是None，如果不是None會報錯表示當前沒有實作
                raise NotImplementedError('Put Flip at last for now')
            # 將差值方式放到lazy的字典當中
            lazyop['interpolation'] = self.interpolation

        if 'gt_bboxes' in results:
            # 如果results當中有目標檢測的標註匡就會到這裡
            # 如果有目標檢測的標註匡時就不可以使用lazy operation
            assert not self.lazy
            # 將標註匡根據縮放比例進行調整
            results['gt_bboxes'] = self._box_resize(results['gt_bboxes'],
                                                    self.scale_factor)
            if 'proposals' in results and results['proposals'] is not None:
                # 如果有預選匡也需要將預選匡進行位置調整
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_resize(
                    results['proposals'], self.scale_factor)

        # 回傳更新後的results
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class RandomRescale:
    """Randomly resize images so that the short_edge is resized to a specific
    size in a given range. The scale ratio is unchanged after resizing.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs", "img_shape", "keep_ratio", "scale_factor", "resize_size",
    "short_edge".

    Args:
        scale_range (tuple[int]): The range of short edge length. A closed
            interval.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear". Default: "bilinear".
    """

    def __init__(self, scale_range, interpolation='bilinear'):
        self.scale_range = scale_range
        # make sure scale_range is legal, first make sure the type is OK
        assert mmcv.is_tuple_of(scale_range, int)
        assert len(scale_range) == 2
        assert scale_range[0] < scale_range[1]
        assert np.all([x > 0 for x in scale_range])

        self.keep_ratio = True
        self.interpolation = interpolation

    def __call__(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        short_edge = np.random.randint(self.scale_range[0],
                                       self.scale_range[1] + 1)
        resize = Resize((-1, short_edge),
                        keep_ratio=True,
                        interpolation=self.interpolation,
                        lazy=False)
        results = resize(results)

        results['short_edge'] = short_edge
        return results

    def __repr__(self):
        scale_range = self.scale_range
        repr_str = (f'{self.__class__.__name__}('
                    f'scale_range=({scale_range[0]}, {scale_range[1]}), '
                    f'interpolation={self.interpolation})')
        return repr_str


@PIPELINES.register_module()
class Flip:
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "keypoint", "lazy" and
    "flip_direction". Required keys in "lazy" is None, added or modified key
    are "flip" and "flip_direction". The Flip augmentation should be placed
    after any cropping / reshaping augmentations, to make sure crop_quadruple
    is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): Transform the label of the
            flipped image with the specific label. Default: None.
        left_kp (list[int]): Indexes of left keypoints, used to flip keypoints.
            Default: None.
        right_kp (list[ind]): Indexes of right keypoints, used to flip
            keypoints. Default: None.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None,
                 lazy=False):
        """ 已看過，有概率的將圖像進行翻轉
        Args:
            flip_ratio: 翻轉的概率
            direction: 翻轉的方向
            flip_label_map: 用特定標籤變換翻轉圖像的標籤
            left_kp: 左邊關鍵點的索引，用於翻轉關鍵點，如果有進行關鍵點檢測就會用到
            right_kp: 右邊關鍵點的索引，用於翻轉關鍵點，如果有進行關鍵點檢測就會用到
            lazy: 是否啟用lazy操作
        """
        if direction not in self._directions:
            # 如果指定的翻轉方向不在有實作的方式當中就會報錯
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        # 保存傳入的參數
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.lazy = lazy

    def _flip_imgs(self, imgs, modality):
        """ 已看過，將圖像進行翻轉
        Args:
            imgs: 圖像資料，list[ndarray]
            modality: 圖像的類別，會是RGB(一般影片圖像)或是Flow(光流圖像)
        """
        # 透過imflip_進行翻轉
        _ = [mmcv.imflip_(img, self.direction) for img in imgs]
        # 獲取圖像數量
        lt = len(imgs)
        if modality == 'Flow':
            # 如果是處理光流圖像就會到這裡
            # The 1st frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                # 透過iminvertr進行翻轉，等到有機會再來看
                imgs[i] = mmcv.iminvert(imgs[i])
        # 回傳翻轉後的imgs
        return imgs

    def _flip_kps(self, kps, kpscores, img_width):
        kp_x = kps[..., 0]
        kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
        new_order = list(range(kps.shape[2]))
        if self.left_kp is not None and self.right_kp is not None:
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left] = right
                new_order[right] = left
        kps = kps[:, :, new_order]
        if kpscores is not None:
            kpscores = kpscores[:, :, new_order]
        return kps, kpscores

    @staticmethod
    def _box_flip(box, img_width):
        """Flip the bounding boxes given the width of the image.

        Args:
            box (np.ndarray): The bounding boxes.
            img_width (int): The img width.
        """
        box_ = box.copy()
        box_[..., 0::4] = img_width - box[..., 2::4]
        box_[..., 2::4] = img_width - box[..., 0::4]
        return box_

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        # 已看過，進行圖像的翻轉
        # 初始化lazy字典，如果沒有設定lazy就不會有任何改變
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            # 如果results當中有關節點資訊就不可以使用lazy
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')
            # 如果有關節點資訊就只能使用水平翻轉，不可以有其他翻轉方式
            assert self.direction == 'horizontal', (
                'Only horizontal flips are'
                'supported for human keypoints')

        # 獲取modality資訊
        modality = results['modality']
        if modality == 'Flow':
            # 如果是Flow也就是光流資訊就會到這裡，光流資訊只支援水平翻轉
            assert self.direction == 'horizontal'

        # 獲取隨機值決定是否進行翻轉
        flip = np.random.rand() < self.flip_ratio

        # 將是否進行翻轉資料保存
        results['flip'] = flip
        # 保存翻轉方向
        results['flip_direction'] = self.direction
        # 獲取當前圖像寬度，水平翻轉只會需要用到寬度資訊
        img_width = results['img_shape'][1]

        if self.flip_label_map is not None and flip:
            # 如果有設定flip_label_map且當前需要進行flip就會到這裡
            results['label'] = self.flip_label_map.get(results['label'],
                                                       results['label'])

        if not self.lazy:
            # 如果沒有使用lazy operation就會到這裡
            if flip:
                # 如果有需要翻轉就會進來
                if 'imgs' in results:
                    # 將圖像資料進行翻轉
                    results['imgs'] = self._flip_imgs(results['imgs'],
                                                      modality)
                if 'keypoint' in results:
                    # 將關節點資料進行翻轉
                    kp = results['keypoint']
                    kpscore = results.get('keypoint_score', None)
                    kp, kpscore = self._flip_kps(kp, kpscore, img_width)
                    results['keypoint'] = kp
                    if 'keypoint_score' in results:
                        results['keypoint_score'] = kpscore
        else:
            # 如果有啟用lazy就會到這裡
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Use one Flip please')
            # 保存當前是否翻轉
            lazyop['flip'] = flip
            # 保存翻轉方向
            lazyop['flip_direction'] = self.direction

        if 'gt_bboxes' in results and flip:
            # 如果有目標檢測標註匡就會到這裡進行翻轉
            assert not self.lazy and self.direction == 'horizontal'
            width = results['img_shape'][1]
            results['gt_bboxes'] = self._box_flip(results['gt_bboxes'], width)
            if 'proposals' in results and results['proposals'] is not None:
                # 將預選匡也進行翻轉
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_flip(results['proposals'],
                                                      width)

        # 將更新後的results回傳
        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map}, lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class Normalize:
    """Normalize images with the given mean and std value.

    Required keys are "imgs", "img_shape", "modality", added or modified
    keys are "imgs" and "img_norm_cfg". If modality is 'Flow', additional
    keys "scale_factor" is required

    Args:
        mean (Sequence[float]): Mean values of different channels.
        std (Sequence[float]): Std values of different channels.
        to_bgr (bool): Whether to convert channels from RGB to BGR.
            Default: False.
        adjust_magnitude (bool): Indicate whether to adjust the flow magnitude
            on 'scale_factor' when modality is 'Flow'. Default: False.
    """

    def __init__(self, mean, std, to_bgr=False, adjust_magnitude=False):
        """ 已看過，將圖像進行均值標準化操作
        Args:
            mean: 均值
            std: 表準差
            to_bgr: 是否需要轉成RGB
            adjust_magnitude:
        """
        # 檢查傳入的參數
        if not isinstance(mean, Sequence):
            raise TypeError(
                f'Mean must be list, tuple or np.ndarray, but got {type(mean)}'
            )

        if not isinstance(std, Sequence):
            raise TypeError(
                f'Std must be list, tuple or np.ndarray, but got {type(std)}')

        # 將傳入的mean與std轉成ndarray格式
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        # 保存傳入參數
        self.to_bgr = to_bgr
        self.adjust_magnitude = adjust_magnitude

    def __call__(self, results):
        # 已看過，將圖像進行均值標準化調整

        # 獲取當前處理圖像的類型
        modality = results['modality']

        if modality == 'RGB':
            # 如果傳入的是RGB資料就會到這裡
            # 獲取圖像數量
            n = len(results['imgs'])
            # 獲取圖像的shape資訊
            h, w, c = results['imgs'][0].shape
            # 構建一個ndarray且shape是[clip_len, height, width, channel]
            imgs = np.empty((n, h, w, c), dtype=np.float32)
            # 遍歷傳入的圖像列表
            for i, img in enumerate(results['imgs']):
                # 將圖像放到imgs上
                imgs[i] = img

            # 遍歷imgs當中的圖像
            for img in imgs:
                # 進行均值標準化調整，這裡也會看是否需要將通道調整成RGB
                mmcv.imnormalize_(img, self.mean, self.std, self.to_bgr)

            # 更新results當中imgs的資訊
            results['imgs'] = imgs
            # 保存調整的均值以及標準差數值
            results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_bgr=self.to_bgr)
            return results
        if modality == 'Flow':
            # 如果處理的是光流資訊就會到這裡，等有機會處理光流資訊再到這裡查看
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            assert self.mean.shape[0] == 2
            assert self.std.shape[0] == 2
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            x_flow = (x_flow - self.mean[0]) / self.std[0]
            y_flow = (y_flow - self.mean[1]) / self.std[1]
            if self.adjust_magnitude:
                x_flow = x_flow * results['scale_factor'][0]
                y_flow = y_flow * results['scale_factor'][1]
            imgs = np.stack([x_flow, y_flow], axis=-1)
            results['imgs'] = imgs
            args = dict(
                mean=self.mean,
                std=self.std,
                to_bgr=self.to_bgr,
                adjust_magnitude=self.adjust_magnitude)
            results['img_norm_cfg'] = args
            return results
        raise NotImplementedError

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'mean={self.mean}, '
                    f'std={self.std}, '
                    f'to_bgr={self.to_bgr}, '
                    f'adjust_magnitude={self.adjust_magnitude})')
        return repr_str


@PIPELINES.register_module()
class ColorJitter:
    """Perform ColorJitter to each img.

    Required keys are "imgs", added or modified keys are "imgs".

    Args:
        brightness (float | tuple[float]): The jitter range for brightness, if
            set as a float, the range will be (1 - brightness, 1 + brightness).
            Default: 0.5.
        contrast (float | tuple[float]): The jitter range for contrast, if set
            as a float, the range will be (1 - contrast, 1 + contrast).
            Default: 0.5.
        saturation (float | tuple[float]): The jitter range for saturation, if
            set as a float, the range will be (1 - saturation, 1 + saturation).
            Default: 0.5.
        hue (float | tuple[float]): The jitter range for hue, if set as a
            float, the range will be (-hue, hue). Default: 0.1.
    """

    @staticmethod
    def check_input(val, max, base):
        if isinstance(val, tuple):
            assert base - max <= val[0] <= val[1] <= base + max
            return val
        assert val <= max
        return (base - val, base + val)

    @staticmethod
    def rgb_to_grayscale(img):
        return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    @staticmethod
    def adjust_contrast(img, factor):
        val = np.mean(ColorJitter.rgb_to_grayscale(img))
        return factor * img + (1 - factor) * val

    @staticmethod
    def adjust_saturation(img, factor):
        gray = np.stack([ColorJitter.rgb_to_grayscale(img)] * 3, axis=-1)
        return factor * img + (1 - factor) * gray

    @staticmethod
    def adjust_hue(img, factor):
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        offset = int(factor * 255)
        hsv[..., 0] = (hsv[..., 0] + offset) % 180
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img.astype(np.float32)

    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1):
        self.brightness = self.check_input(brightness, 1, 1)
        self.contrast = self.check_input(contrast, 1, 1)
        self.saturation = self.check_input(saturation, 1, 1)
        self.hue = self.check_input(hue, 0.5, 0)
        self.fn_idx = np.random.permutation(4)

    def __call__(self, results):
        imgs = results['imgs']
        num_clips, clip_len = 1, len(imgs)

        new_imgs = []
        for i in range(num_clips):
            b = np.random.uniform(
                low=self.brightness[0], high=self.brightness[1])
            c = np.random.uniform(low=self.contrast[0], high=self.contrast[1])
            s = np.random.uniform(
                low=self.saturation[0], high=self.saturation[1])
            h = np.random.uniform(low=self.hue[0], high=self.hue[1])
            start, end = i * clip_len, (i + 1) * clip_len

            for img in imgs[start:end]:
                img = img.astype(np.float32)
                for fn_id in self.fn_idx:
                    if fn_id == 0 and b != 1:
                        img *= b
                    if fn_id == 1 and c != 1:
                        img = self.adjust_contrast(img, c)
                    if fn_id == 2 and s != 1:
                        img = self.adjust_saturation(img, s)
                    if fn_id == 3 and h != 0:
                        img = self.adjust_hue(img, h)
                img = np.clip(img, 0, 255).astype(np.uint8)
                new_imgs.append(img)
        results['imgs'] = new_imgs
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'brightness={self.brightness}, '
                    f'contrast={self.contrast}, '
                    f'saturation={self.saturation}, '
                    f'hue={self.hue})')
        return repr_str


@PIPELINES.register_module()
class CenterCrop(RandomCrop):
    """Crop the center area from images.

    Required keys are "img_shape", "imgs" (optional), "keypoint" (optional),
    added or modified keys are "imgs", "keypoint", "crop_bbox", "lazy" and
    "img_shape". Required keys in "lazy" is "crop_bbox", added or modified key
    is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, crop_size, lazy=False):
        self.crop_size = _pair(crop_size)
        self.lazy = lazy
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str


@PIPELINES.register_module()
class ThreeCrop:
    """Crop images into three crops.

    Crop the images equally into three crops with equal intervals along the
    shorter side.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the ThreeCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)
        if 'gt_bboxes' in results or 'proposals' in results:
            warnings.warn('ThreeCrop cannot process bounding boxes')

        imgs = results['imgs']
        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size
        assert crop_h == img_h or crop_w == img_w

        if crop_h == img_h:
            w_step = (img_w - crop_w) // 2
            offsets = [
                (0, 0),  # left
                (2 * w_step, 0),  # right
                (w_step, 0),  # middle
            ]
        elif crop_w == img_w:
            h_step = (img_h - crop_h) // 2
            offsets = [
                (0, 0),  # top
                (0, 2 * h_step),  # down
                (0, h_step),  # middle
            ]

        cropped = []
        crop_bboxes = []
        for x_offset, y_offset in offsets:
            bbox = [x_offset, y_offset, x_offset + crop_w, y_offset + crop_h]
            crop = [
                img[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            cropped.extend(crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs))])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = cropped
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class TenCrop:
    """Crop the images into 10 crops (corner + center + flip).

    Crop the four corners and the center part of the image with the same
    given crop_size, and flip it horizontally.
    Required keys are "imgs", "img_shape", added or modified keys are "imgs",
    "crop_bbox" and "img_shape".

    Args:
        crop_size(int | tuple[int]): (w, h) of crop size.
    """

    def __init__(self, crop_size):
        self.crop_size = _pair(crop_size)
        if not mmcv.is_tuple_of(self.crop_size, int):
            raise TypeError(f'Crop_size must be int or tuple of int, '
                            f'but got {type(crop_size)}')

    def __call__(self, results):
        """Performs the TenCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, False)

        if 'gt_bboxes' in results or 'proposals' in results:
            warnings.warn('TenCrop cannot process bounding boxes')

        imgs = results['imgs']

        img_h, img_w = results['imgs'][0].shape[:2]
        crop_w, crop_h = self.crop_size

        w_step = (img_w - crop_w) // 4
        h_step = (img_h - crop_h) // 4

        offsets = [
            (0, 0),  # upper left
            (4 * w_step, 0),  # upper right
            (0, 4 * h_step),  # lower left
            (4 * w_step, 4 * h_step),  # lower right
            (2 * w_step, 2 * h_step),  # center
        ]

        img_crops = list()
        crop_bboxes = list()
        for x_offset, y_offsets in offsets:
            crop = [
                img[y_offsets:y_offsets + crop_h, x_offset:x_offset + crop_w]
                for img in imgs
            ]
            flip_crop = [np.flip(c, axis=1).copy() for c in crop]
            bbox = [x_offset, y_offsets, x_offset + crop_w, y_offsets + crop_h]
            img_crops.extend(crop)
            img_crops.extend(flip_crop)
            crop_bboxes.extend([bbox for _ in range(len(imgs) * 2)])

        crop_bboxes = np.array(crop_bboxes)
        results['imgs'] = img_crops
        results['crop_bbox'] = crop_bboxes
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(crop_size={self.crop_size})'
        return repr_str


@PIPELINES.register_module()
class AudioAmplify:
    """Amplify the waveform.

    Required keys are "audios", added or modified keys are "audios",
    "amplify_ratio".

    Args:
        ratio (float): The ratio used to amplify the audio waveform.
    """

    def __init__(self, ratio):
        if isinstance(ratio, float):
            self.ratio = ratio
        else:
            raise TypeError('Amplification ratio should be float.')

    def __call__(self, results):
        """Perform the audio amplification.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        assert 'audios' in results
        results['audios'] *= self.ratio
        results['amplify_ratio'] = self.ratio

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(ratio={self.ratio})'
        return repr_str


@PIPELINES.register_module()
class MelSpectrogram:
    """MelSpectrogram. Transfer an audio wave into a melspectogram figure.

    Required keys are "audios", "sample_rate", "num_clips", added or modified
    keys are "audios".

    Args:
        window_size (int): The window size in millisecond. Default: 32.
        step_size (int): The step size in millisecond. Default: 16.
        n_mels (int): Number of mels. Default: 80.
        fixed_length (int): The sample length of melspectrogram maybe not
            exactly as wished due to different fps, fix the length for batch
            collation by truncating or padding. Default: 128.
    """

    def __init__(self,
                 window_size=32,
                 step_size=16,
                 n_mels=80,
                 fixed_length=128):
        if all(
                isinstance(x, int)
                for x in [window_size, step_size, n_mels, fixed_length]):
            self.window_size = window_size
            self.step_size = step_size
            self.n_mels = n_mels
            self.fixed_length = fixed_length
        else:
            raise TypeError('All arguments should be int.')

    def __call__(self, results):
        """Perform MelSpectrogram transformation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import librosa
        except ImportError:
            raise ImportError('Install librosa first.')
        signals = results['audios']
        sample_rate = results['sample_rate']
        n_fft = int(round(sample_rate * self.window_size / 1000))
        hop_length = int(round(sample_rate * self.step_size / 1000))
        melspectrograms = list()
        for clip_idx in range(results['num_clips']):
            clip_signal = signals[clip_idx]
            mel = librosa.feature.melspectrogram(
                y=clip_signal,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=self.n_mels)
            if mel.shape[0] >= self.fixed_length:
                mel = mel[:self.fixed_length, :]
            else:
                mel = np.pad(
                    mel, ((0, mel.shape[-1] - self.fixed_length), (0, 0)),
                    mode='edge')
            melspectrograms.append(mel)

        results['audios'] = np.array(melspectrograms)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}'
                    f'(window_size={self.window_size}), '
                    f'step_size={self.step_size}, '
                    f'n_mels={self.n_mels}, '
                    f'fixed_length={self.fixed_length})')
        return repr_str
