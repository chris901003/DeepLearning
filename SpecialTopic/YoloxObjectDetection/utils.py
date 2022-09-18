import numpy as np
import cv2
import numbers


def get_specified_option(support, target):
    # 根據傳入的字典以及指定的參數或是類名，獲取對應的值，如果途中出錯就會報錯
    if isinstance(target, dict):
        target = target.pop('type', None)
        assert target is not None, '在字典當中沒有獲取type資訊'
    assert target in support, f'在support當中沒有 {target} 的值'
    result = support[target]
    return result


def find_inside_bboxes(bboxes, img_h, img_w):
    inside_inds = (bboxes[:, 0] < img_w) & (bboxes[:, 2] > 0) & (bboxes[:, 1] < img_h) & (bboxes[:, 3] > 0)
    return inside_inds


def imgflip(img, direction='horizontal'):
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))


def bbox_flip(bboxes, img_shape, direction):
    assert bboxes.shape[-1] % 4 == 0
    flipped = bboxes.copy()
    if direction == 'horizontal':
        w = img_shape[1]
        flipped[..., 0::4] = w - bboxes[..., 2::4]
        flipped[..., 2::4] = w - bboxes[..., 0::4]
    elif direction == 'vertical':
        h = img_shape[0]
        flipped[..., 1::4] = h - bboxes[..., 3::4]
        flipped[..., 3::4] = h - bboxes[..., 1::4]
    elif direction == 'diagonal':
        w = img_shape[1]
        h = img_shape[0]
        flipped[..., 0::4] = w - bboxes[..., 2::4]
        flipped[..., 1::4] = h - bboxes[..., 3::4]
        flipped[..., 2::4] = w - bboxes[..., 0::4]
        flipped[..., 3::4] = h - bboxes[..., 1::4]
    else:
        raise ValueError(f"Invalid flipping direction '{direction}'")
    return flipped


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


def imrescale(img, scale, return_scale=False, interpolation='bilinear'):
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def _scale_size(size, scale):
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rescale_size(old_size, scale, return_scale=False):
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError('Scale有誤')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        raise TypeError
    new_size = _scale_size((w, h), scale_factor)
    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


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
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
    border_type = {
        'constant': cv2.BORDER_CONSTANT,
        'edge': cv2.BORDER_REPLICATE,
        'reflect': cv2.BORDER_REFLECT_101,
        'symmetric': cv2.BORDER_REFLECT
    }
    img = cv2.copyMakeBorder(img, padding[1], padding[3], padding[0], padding[2],
                             border_type[padding_mode], value=pad_val)
    return img


def impad_to_multiple(img, divisor, pad_val=0):
    pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
    pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
    return impad(img, shape=(pad_h, pad_w), pad_val=pad_val)
