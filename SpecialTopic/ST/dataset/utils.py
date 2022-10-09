import numpy as np
import cv2
import torch
from ..utils import get_cls_from_dict


class Compose:
    def __init__(self, pipelines_cfg):
        from .pipeline import LoadInfoFromAnno, Resize, ResizeAndAugmentation, Mosaic, Collect
        from .pipeline_recognizer3d import PyAVInit, SampleFrames, PyAVDecode, MultiScaleCrop, Flip, Normalize
        from .pipeline_recognizer3d import FormatShape, ToTensor, ThreeCrop
        from .pipeline_recognizer3d import Resize as Recognizer3dResize
        from .pipeline_classify import LoadRemainingAnnotation, ResizeSingle, NormalizeSingle
        support_pipeline = {
            'LoadInfoFromAnno': LoadInfoFromAnno, 'Resize': Resize, 'ResizeAndAugmentation': ResizeAndAugmentation,
            'Mosaic': Mosaic, 'Collect': Collect, 'PyAVInit': PyAVInit, 'SampleFrames': SampleFrames,
            'PyAVDecode': PyAVDecode, 'MultiScaleCrop': MultiScaleCrop, 'Flip': Flip, 'Normalize': Normalize,
            'FormatShape': FormatShape, 'ToTensor': ToTensor, 'Recognizer3dResize': Recognizer3dResize,
            'ThreeCrop': ThreeCrop, 'LoadRemainingAnnotation': LoadRemainingAnnotation, 'ResizeSingle': ResizeSingle,
            'NormalizeSingle': NormalizeSingle
        }
        self.pipelines = list()
        for pipeline_cfg in pipelines_cfg:
            pipeline_cls = get_cls_from_dict(support_pipeline, pipeline_cfg)
            pipeline = pipeline_cls(**pipeline_cfg)
            self.pipelines.append(pipeline)

    def __call__(self, data):
        for pipeline in self.pipelines:
            data = pipeline(data)
        return data


def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def rand(a=0., b=1.):
    return np.random.rand() * (b - a) + a


def _scale_size(size, scale):
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rescale_size(old_size, scale, return_scale=False):
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        raise TypeError(f'Scale must be a number or tuple of int')
    new_size = _scale_size((w, h), scale_factor)
    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imresize(img, size, return_scale=False, interpolation='bilinear', out=None):
    cv2_interp_codes = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }
    h, w = img.shape[:2]
    resized_img = cv2.resize(img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def imflip_(img, direction='horizontal'):
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return cv2.flip(img, 1, img)
    elif direction == 'vertical':
        return cv2.flip(img, 0, img)
    else:
        return cv2.flip(img, -1, img)


def imflip(img, direction='horizontal'):
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


def imnormalize_(img, mean, std, to_rgb=True):
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    std_inv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, std_inv, img)
    return img


def imnormalize(img, mean, std, to_rgb=True):
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, int):
        return torch.LongTensor([data])
    if isinstance(data, float):
        return torch.FloatTensor([data])
    raise TypeError('Can not convert to tensor')


def imrescale(img, scale, return_scale=False, interpolation='bilinear'):
    h, w = img.shae[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescale_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescale_img, scale_factor
    else:
        return rescale_img


def impad(img, *, shape=None, padding=None, pad_val=0, padding_mode='constant'):
    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        width = max(shape[1] - img.shape[1], 0)
        height = max(shape[0] - img.shape[0], 0)
        padding = (0, 0, width, height)
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, int):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('給定的padding需要長度為2或是4或是直接給一個數')
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(img, padding[1], padding[3], padding[0], padding[2], border_type[padding_mode],
                             value=pad_val)
    return img


def impad_to_multiple(img, divisor, pad_val=0):
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, shape=(pad_h, pad_w), pad_val=pad_val)
