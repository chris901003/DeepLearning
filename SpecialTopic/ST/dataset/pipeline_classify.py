import cv2
import os
import numpy as np
from .utils import imresize


class LoadRemainingAnnotation:
    def __init__(self, key):
        self.key = key
        support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
        self.support_image_format = support_image_format

    def __call__(self, result):
        image_path = result['image_path']
        label = result['label']
        assert os.path.exists(image_path) and os.path.splitext(image_path)[1] in self.support_image_format, \
            '提供的圖像不合法'
        image = cv2.imread(image_path)
        result[self.key[0]] = image
        result[self.key[1]] = int(label)
        return result


class ResizeSingle:
    def __init__(self, input_shape, save_info, keep_ratio=True):
        self.input_shape = input_shape
        self.save_info = save_info
        self.keep_ratio = keep_ratio

    def __call__(self, result):
        image = result['image']
        image_height, image_width = image.shape[:2]
        target_height, target_width = self.input_shape
        if self.keep_ratio:
            scale = min(target_height / image_height, target_width / image_width)
            new_height, new_width = int(image_height * scale), int(image_width * scale)
            new_image = imresize(image, (new_width, new_height))
        else:
            new_image = imresize(image, self.input_shape[::-1])
        new_height, new_width = new_image.shape[:2]
        x_offset = (self.input_shape[1] - new_width) // 2
        y_offset = (self.input_shape[0] - new_height) // 2
        picture = np.full((self.input_shape[0], self.input_shape[1], 3), fill_value=(128, 128, 128))
        picture[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = new_image
        result['image'] = picture
        if self.save_info:
            result['resize_image_shape'] = (new_height, new_width)
            result['offset'] = (x_offset, y_offset)
        return result


class NormalizeSingle:
    def __init__(self, mean, std, to_rgb=False):
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb

    def __call__(self, result):
        image = result['image']
        image = image.astype(np.float32)
        if self.to_rgb:
            # 這裡注意一下，如果使用ImageNet的標準化方式需要先調整成RGB在處理
            # 不然就是mean與std需要調整
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image /= 255.0
        image -= np.array(self.mean)
        image /= np.array(self.std)
        result['image'] = image
        return result
