# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.utils import build_from_cfg
from numpy import random
from torchvision.transforms import functional as F

from ..builder import PIPELINES

try:
    import albumentations
except ImportError:
    albumentations = None


@PIPELINES.register_module()
class ToTensor:
    """Transform image to Tensor.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        results (dict): contain all information about training.
    """

    def __init__(self, device='cpu'):
        # 轉換到tensor格式，保存當前訓練設備
        self.device = device

    def _to_tensor(self, x):
        return torch.from_numpy(x.astype('float32')).permute(2, 0, 1).to(
            self.device).div_(255.0)

    def __call__(self, results):
        # 將指定的資訊轉成tensor
        if isinstance(results['img'], (list, tuple)):
            # 將img資訊轉成tensor
            results['img'] = [self._to_tensor(img) for img in results['img']]
        else:
            results['img'] = self._to_tensor(results['img'])

        # 返回更新後的results
        return results


@PIPELINES.register_module()
class NormalizeTensor:
    """Normalize the Tensor image (CxHxW), with mean and std.

    Required key: 'img'. Modifies key: 'img'.

    Args:
        mean (list[float]): Mean values of 3 channels.
        std (list[float]): Std values of 3 channels.
    """

    def __init__(self, mean, std):
        # 進行圖像均值方差調整
        self.mean = mean
        self.std = std

    def __call__(self, results):
        # 將圖像進行標準化
        if isinstance(results['img'], (list, tuple)):
            results['img'] = [
                F.normalize(img, mean=self.mean, std=self.std, inplace=True)
                for img in results['img']
            ]
        else:
            # 進行表準化
            results['img'] = F.normalize(
                results['img'], mean=self.mean, std=self.std, inplace=True)

        return results


@PIPELINES.register_module()
class Compose:
    """Compose a data pipeline with a sequence of transforms.

    Args:
        transforms (list[dict | callable]): Either config
          dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        """ 構建圖像處理流
        Args:
            transforms: 處理流程詳細資訊
        """
        # 檢查transforms需要是Sequence
        assert isinstance(transforms, Sequence)
        # 保存處理實例對象
        self.transforms = []
        # 遍歷每層處理方式
        for transform in transforms:
            if isinstance(transform, dict):
                # 如果是dict格式就會到這裡
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                # 如果是可以直接call的就會到這裡
                self.transforms.append(transform)
            else:
                # 其他就會報錯
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
            dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


@PIPELINES.register_module()
class Collect:
    """Collect data from the loader relevant to the specific task.

    This keeps the items in `keys` as it is, and collect items in `meta_keys`
    into a meta item called `meta_name`.This is usually the last stage of the
    data loader pipeline.
    For example, when keys='imgs', meta_keys=('filename', 'label',
    'original_shape'), meta_name='img_metas', the results will be a dict with
    keys 'imgs' and 'img_metas', where 'img_metas' is a DataContainer of
    another dict with keys 'filename', 'label', 'original_shape'.

    Args:
        keys (Sequence[str|tuple]): Required keys to be collected. If a tuple
          (key, key_new) is given as an element, the item retrieved by key will
          be renamed as key_new in collected data.
        meta_name (str): The name of the key that contains meta information.
          This key is always populated. Default: "img_metas".
        meta_keys (Sequence[str|tuple]): Keys that are collected under
          meta_name. The contents of the `meta_name` dictionary depends
          on `meta_keys`.
    """

    def __init__(self, keys, meta_keys, meta_name='img_metas'):
        """ 蒐集results當中的指定資料
        Args:
            keys: 指定哪些資料我們需要保留
            meta_keys: 哪些要保存到meta當中
            meta_name: meta的名稱
        """
        # 保存傳入資料
        self.keys = keys
        self.meta_keys = meta_keys
        self.meta_name = meta_name

    def __call__(self, results):
        """Performs the Collect formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
              to the next transform in pipeline.
        """
        # 蒐集results當中感興趣資料
        if 'ann_info' in results:
            # 如果results當中有anno_info資料就會到這裡
            results.update(results['ann_info'])

        # 最終需要的資料保存
        data = {}
        # 遍歷指定需要保存的key
        for key in self.keys:
            if isinstance(key, tuple):
                # 如果key是tuple就會到這裡
                assert len(key) == 2
                # 分別提取出key_src與key_tgt
                key_src, key_tgt = key[:2]
            else:
                # 如果只有一個key就將key_src與key_tgt設定成一樣的
                key_src = key_tgt = key
            # 獲取results當中指定的key_src放到data當中，且key設定成key_tgt
            data[key_tgt] = results[key_src]

        # 最終需要保存的meta資訊
        meta = {}
        # 如果meta的key超過一個就會往下
        if len(self.meta_keys) != 0:
            # 遍歷所有的meta_keys
            for key in self.meta_keys:
                if isinstance(key, tuple):
                    assert len(key) == 2
                    key_src, key_tgt = key[:2]
                else:
                    key_src = key_tgt = key
                # 保存meta值
                meta[key_tgt] = results[key_src]
        if 'bbox_id' in results:
            # 如果有bbox_id就會進行保存
            meta['bbox_id'] = results['bbox_id']
        # 將meta透過DC進行包裝
        data[self.meta_name] = DC(meta, cpu_only=True)

        # 回傳整理後的data資訊
        return data

    def __repr__(self):
        """Compute the string representation."""
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, meta_keys={self.meta_keys})')


@PIPELINES.register_module()
class Albumentation:
    """Albumentation augmentation (pixel-level transforms only). Adds custom
    pixel-level transformations from Albumentations library. Please visit
    `https://albumentations.readthedocs.io` to get more information.

    Note: we only support pixel-level transforms.
    Please visit `https://github.com/albumentations-team/`
    `albumentations#pixel-level-transforms`
    to get more information about pixel-level transforms.

    An example of ``transforms`` is as followed:

    .. code-block:: python

        [
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
        transforms (list[dict]): A list of Albumentation transformations
        keymap (dict): Contains {'input key':'albumentation-style key'},
            e.g., {'img': 'image'}.
    """

    def __init__(self, transforms, keymap=None):
        if albumentations is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False

        self.aug = albumentations.Compose(
            [self.albu_builder(t) for t in self.transforms])

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.

        It resembles some of :func:`build_from_cfg` logic.

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
            if not hasattr(albumentations.augmentations.transforms, obj_type):
                warnings.warn('{obj_type} is not pixel-level transformations. '
                              'Please use with caution.')
            obj_cls = getattr(albumentations, obj_type)
        else:
            raise TypeError(f'type must be a str, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """Dictionary mapper.

        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}

        Returns:
            dict: new dict.
        """

        updated_dict = {keymap.get(k, k): v for k, v in d.items()}
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        results = self.aug(**results)
        # back to the original format
        results = self.mapper(results, self.keymap_back)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module()
class PhotometricDistortion:
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

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beta with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        # Apply saturation distortion to hsv-formatted img
        img[:, :, 1] = self.convert(
            img[:, :, 1],
            alpha=random.uniform(self.saturation_lower, self.saturation_upper))
        return img

    def hue(self, img):
        # Apply hue distortion to hsv-formatted img
        img[:, :, 0] = (img[:, :, 0].astype(int) +
                        random.randint(-self.hue_delta, self.hue_delta)) % 180
        return img

    def swap_channels(self, img):
        # Apply channel swap
        if random.randint(2):
            img = img[..., random.permutation(3)]
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        hsv_mode = random.randint(4)
        if hsv_mode:
            # random saturation/hue distortion
            img = mmcv.bgr2hsv(img)
            if hsv_mode == 1 or hsv_mode == 3:
                img = self.saturation(img)
            if hsv_mode == 2 or hsv_mode == 3:
                img = self.hue(img)
            img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        # randomly swap channels
        self.swap_channels(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@PIPELINES.register_module()
class MultiItemProcess:
    """Process each item and merge multi-item results to lists.

    Args:
        pipeline (dict): Dictionary to construct pipeline for a single item.
    """

    def __init__(self, pipeline):
        self.pipeline = Compose(pipeline)

    def __call__(self, results):
        results_ = {}
        for idx, result in results.items():
            single_result = self.pipeline(result)
            for k, v in single_result.items():
                if k in results_:
                    results_[k].append(v)
                else:
                    results_[k] = [v]

        return results_


@PIPELINES.register_module()
class DiscardDuplicatedItems:

    def __init__(self, keys_list):
        """Discard duplicated single-item results.

        Args:
            keys_list (list): List of keys that need to be deduplicate.
        """
        self.keys_list = keys_list

    def __call__(self, results):
        for k, v in results.items():
            if k in self.keys_list:
                assert isinstance(v, Sequence)
                results[k] = v[0]

        return results


@PIPELINES.register_module()
class MultitaskGatherTarget:
    """Gather the targets for multitask heads.

    Args:
        pipeline_list (list[list]): List of pipelines for all heads.
        pipeline_indices (list[int]): Pipeline index of each head.
    """

    def __init__(self,
                 pipeline_list,
                 pipeline_indices=None,
                 keys=('target', 'target_weight')):
        self.keys = keys
        self.pipelines = []
        for pipeline in pipeline_list:
            self.pipelines.append(Compose(pipeline))
        if pipeline_indices is None:
            self.pipeline_indices = list(range(len(pipeline_list)))
        else:
            self.pipeline_indices = pipeline_indices

    def __call__(self, results):
        # generate target and target weights using all pipelines
        pipeline_outputs = []
        for pipeline in self.pipelines:
            pipeline_output = pipeline(results)
            pipeline_outputs.append(pipeline_output.copy())

        for key in self.keys:
            result_key = []
            for ind in self.pipeline_indices:
                result_key.append(pipeline_outputs[ind].get(key, None))
            results[key] = result_key
        return results


@PIPELINES.register_module()
class RenameKeys:
    """Rename the keys.

    Args:
        key_pairs (Sequence[tuple]): Required keys to be renamed.
            If a tuple (key_src, key_tgt) is given as an element,
            the item retrieved by key_src will be renamed as key_tgt.
    """

    def __init__(self, key_pairs):
        self.key_pairs = key_pairs

    def __call__(self, results):
        """Rename keys."""
        for key_pair in self.key_pairs:
            assert len(key_pair) == 2
            key_src, key_tgt = key_pair
            results[key_tgt] = results.pop(key_src)
        return results
