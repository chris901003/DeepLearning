import numpy as np
import os
from PIL import Image
import cv2
from .utils import cvtColor, rand


class LoadInfoFromAnno:
    def __init__(self, key):
        support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
        self.key = key
        self.support_image_format = support_image_format

    def __call__(self, data):
        annotations_line = data.get(self.key, None)
        assert annotations_line is not None, f'指定的 {self.key} 不在data當中'
        if not isinstance(annotations_line, list):
            annotations_line = [annotations_line]
        images_path = list()
        bboxes = list()
        for annotation_line in annotations_line:
            line = annotation_line.split()
            images_path.append(line[0])
            assert os.path.exists(line[0]), f'指定 {line[0]} 圖像資料不存在'
            assert os.path.splitext(line[0])[1] in self.support_image_format, f'指定 {line[0]} 圖像格式不支援'
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
            bboxes.append(box)
        data['images_path'] = images_path
        data['bboxes'] = bboxes
        return data


class Resize:
    def __init__(self, input_shape, keep_ratio=True, save_info=False):
        self.input_shape = input_shape
        self.keep_ratio = keep_ratio
        self.save_info = save_info

    def __call__(self, data):
        images_path = data.get('images_path', None)
        bboxes = data.get('bboxes', None)
        assert images_path is not None
        assert bboxes is not None
        if not isinstance(images_path, list):
            images_path = [images_path]
        if not isinstance(bboxes, list):
            bboxes = [bboxes]
        new_images, new_bboxes, ori_size = list(), list(), list()
        for image_path, box in zip(images_path, bboxes):
            image = Image.open(image_path)
            image = cvtColor(image)
            img_width, img_height = image.size
            ori_size.append((img_height, img_width))
            h, w = self.input_shape
            if self.keep_ratio:
                scale = min(h / img_height, w / img_width)
                nw = int(img_width * scale)
                nh = int(img_height * scale)
                dx = (w - nw) // 2
                dy = (h - nh) // 2
                image = image.resize((nw, nh), Image.BICUBIC)
                new_image = Image.new('RGB', (w, h), (128, 128, 128))
                new_image.paste(image, (dx, dy))
                image_data = np.array(new_image, np.float32)
                if len(box) > 0:
                    np.random.shuffle(box)
                    box[:, [0, 2]] = box[:, [0, 2]] * nw / img_width + dx
                    box[:, [1, 3]] = box[:, [1, 3]] * nh / img_height + dy
            else:
                scale_w = w / img_width
                scale_h = h / img_height
                image_data = image.resize((w, h), Image.BICUBIC)
                if len(box) > 0:
                    np.random.shuffle(box)
                    box[:, [0, 2]] = box[:, [0, 2]] * scale_w
                    box[:, [1, 3]] = box[:, [1, 3]] * scale_h
            if len(box) > 0:
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
            new_images.append(image_data)
            new_bboxes.append(box)
        if self.save_info:
            data['ori_size'] = ori_size
            data['keep_ratio'] = self.keep_ratio
        data['image'] = new_images
        data['bboxes'] = new_bboxes
        return data


class ResizeAndAugmentation:
    def __init__(self, input_shape, jitter=0.3, hue=0.1, sat=0.7, val=0.4):
        self.input_shape = input_shape
        self.jitter = jitter
        self.hue = hue
        self.sat = sat
        self.val = val

    def __call__(self, data):
        images_path = data.get('images_path', None)
        bboxes = data.get('bboxes', None)
        assert images_path is not None
        assert bboxes is not None
        if not isinstance(images_path, list):
            images_path = [images_path]
        if not isinstance(bboxes, list):
            bboxes = [bboxes]
        new_images, new_bboxes = list(), list()
        for image_path, box in zip(images_path, bboxes):
            image = Image.open(image_path)
            image = cvtColor(image)
            iw, ih = image.size
            h, w = self.input_shape
            new_ar = iw / ih * rand(1 - self.jitter, 1 + self.jitter) / rand(1 - self.jitter, 1 + self.jitter)
            scale = rand(0.25, 2)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)
            dx = int(rand(0, w - nw))
            dy = int(rand(0, h - nh))
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image = new_image

            flip = rand() < 0.5
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image_data = np.array(image, np.uint8)
            r = np.random.uniform(-1, 1, 3) * [self.hue, self.sat, self.val] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
            dtype = image_data.dtype
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                if flip:
                    box[:, [0, 2]] = w - box[:, [2, 0]]
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
            new_images.append(image_data)
            new_bboxes.append(box)
        data['image'] = new_images
        data['bbox'] = new_bboxes
        return data


class Mosaic:
    def __init__(self, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        self.input_shape = input_shape
        self.jitter = jitter
        self.hue = hue
        self.sat = sat
        self.val = val

    @staticmethod
    def merge_bboxes(bboxes, cut_x, cut_y):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                if i == 0:
                    if y1 > cut_y or x1 > cut_x:
                        continue
                    if y2 >= cut_y >= y1:
                        y2 = cut_y
                    if x2 >= cut_x >= x1:
                        x2 = cut_x
                elif i == 1:
                    if y2 < cut_y or x1 > cut_x:
                        continue
                    if y2 >= cut_y >= y1:
                        y1 = cut_y
                    if x2 >= cut_x >= x1:
                        x2 = cut_x
                elif i == 2:
                    if y2 < cut_y or x2 < cut_x:
                        continue
                    if y2 >= cut_y >= y1:
                        y1 = cut_y
                    if x2 >= cut_x >= x1:
                        x1 = cut_x
                elif i == 3:
                    if y1 > cut_y or x2 < cut_x:
                        continue
                    if y2 >= cut_y >= y1:
                        y2 = cut_y
                    if x2 >= cut_x >= x1:
                        x1 = cut_x
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def __call__(self, data):
        h, w = self.input_shape
        min_offset_x = rand(0.3, 0.7)
        min_offset_y = rand(0.3, 0.7)
        images_path = data.get('images_path', None)
        bboxes = data.get('bboxes', None)
        assert images_path is not None
        assert bboxes is not None
        if not isinstance(images_path, list):
            images_path = [images_path]
        if not isinstance(bboxes, list):
            bboxes = [bboxes]
        image_datas = []
        box_datas = []
        index = 0
        for image_path, box in zip(images_path, bboxes):
            image = Image.open(image_path)
            image = cvtColor(image)
            iw, ih = image.size
            flip = rand() < 0.5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]
            new_ar = iw / ih * rand(1 - self.jitter, 1 + self.jitter) / rand(1 - self.jitter, 1 + self.jitter)
            scale = rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh
            else:
                raise ValueError
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)
            index = index + 1
            box_data = []
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box
            image_datas.append(image_data)
            box_datas.append(box_data)
        cut_x = int(w * min_offset_x)
        cut_y = int(h * min_offset_y)
        new_image = np.zeros([h, w, 3])
        new_image[:cut_y, : cut_x, :] = image_datas[0][:cut_y, :cut_x, :]
        new_image[cut_y:, :cut_x, :] = image_datas[1][cut_y:, :cut_x, :]
        new_image[cut_y:, cut_x:, :] = image_datas[2][cut_y:, cut_x:, :]
        new_image[:cut_y, cut_x:, :] = image_datas[3][:cut_y, cut_x:, :]
        new_image = np.array(new_image, np.uint8)
        r = np.random.uniform(-1, 1, 3) * [self.hue, self.sat, self.val] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
        new_boxes = self.merge_bboxes(box_datas, cut_x, cut_y)
        data['image'] = new_image
        data['boxes'] = new_boxes
        return data


class Collect:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        result = dict()
        for key in self.keys:
            info = data.get(key, None)
            assert info is not None, f'指定的 {key} 不在data當中'
            result[key] = info
        return result
