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
        support_pipeline = {
            'LoadInfoFromAnno': LoadInfoFromAnno, 'Resize': Resize, 'ResizeAndAugmentation': ResizeAndAugmentation,
            'Mosaic': Mosaic, 'Collect': Collect, 'PyAVInit': PyAVInit, 'SampleFrames': SampleFrames,
            'PyAVDecode': PyAVDecode, 'MultiScaleCrop': MultiScaleCrop, 'Flip': Flip, 'Normalize': Normalize,
            'FormatShape': FormatShape, 'ToTensor': ToTensor, 'Recognizer3dResize': Recognizer3dResize,
            'ThreeCrop': ThreeCrop
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


def imnormalize_(img, mean, std, to_rgb=True):
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    std_inv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, std_inv, img)
    return img


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
