# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    # 已看過
    # region = (crop_top, crop_left, crop_height, crop_width)
    # (上面剪裁量, 左邊剪裁量, 最後高度, 最後寬度)
    # region就是官方要的格式，左上角座標以及高寬
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    # 更新圖像處理後的大小
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    # 更新gt_box
    if "boxes" in target:
        boxes = target["boxes"]
        # 圖像高寬最大值
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        # 所有gt_box移動到剪裁後圖像的對應位置
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        # 把[xmin, ymin, xmax, ymax] -> [[xmin, ymin], [xmax, ymax]]
        # 同時把超過[h, w]變成在範圍內
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        # 最小值不能小於0，剛剛有進行平移所以小於左上角點的gt_box座標會小於0
        cropped_boxes = cropped_boxes.clamp(min=0)
        # 計算每個gt_box的面積
        # 右下角點減去左上角點
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        # 重新合併
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        # 這裡還沒有過濾不在剪裁後圖片的gt_box
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        # 直接把要的放入新的array
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            # 再把左上角點與右下角點拆開
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            # 製造過濾用的mask，把已經不在剪裁後的圖片的gt_box拿掉
            # 要留下的在keep中是True否則是False
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            # 只保留True的標註
            target[field] = target[field][keep]

    # 返回剪裁好的圖像以及處理好的gt_box(做好偏移以及移除非法gt_box)
    return cropped_image, target


def hflip(image, target):
    # 已看過
    # 直接使用官方的hflip就可以對圖像翻轉
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    # 對圖像gt_box進行左右翻轉
    if "boxes" in target:
        boxes = target["boxes"]
        # 進行左右翻轉，注意y不會改變只有x會變而已
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    # masks就直接左右翻轉就可以了
    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # 已看過
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        # image_size = 原圖大小
        # size = 輸出大小
        # max_size = 輸出圖像最大大小
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            # 當我們把圖片小邊調整到size時大邊不能超過max_size
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            # 小邊等於size時直接回傳
            return (h, w)

        # 將小邊調整成size，同時放大大邊
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        # 回傳調整後大小
        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            # size會走這裡
            return get_size_with_aspect_ratio(image_size, size, max_size)

    # get_size(原始圖像大小, 輸出圖像大小, 輸出圖像最大大小)
    # size的最小值會等於size最大值一定大於等於size
    size = get_size(image.size, size, max_size)
    # 將圖片調整到對應大小
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    # 獲得縮放比例
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    # 調整gt_box
    if "boxes" in target:
        boxes = target["boxes"]
        # 直接縮放
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        # 原本大小乘上縮放比例
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        # masks就用雙線性插值處理
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # 已看過
    # assumes that we only pad on the bottom right corners
    # 在圖像右邊或是下面增加黑邊
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    # 更新target中圖片的大小，變成加上pad後的圖像大小
    # gt_box是不用動的，因為是加在右邊以及下面所以不會影響原始gt_box位置
    target["size"] = torch.tensor(padded_image.size[::-1])
    # 如果有masks也要增加pad
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    # 隨機剪裁
    # 這裡是給一個固定大小做剪裁
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        # 調用官方的隨機剪裁，不過我們這裡不會直接剪圖片，而已要他返回怎麼剪
        # 也就是會返回剪裁後圖片左上角對應上原圖的哪裡，以及高寬(當然返回的高寬就是給的高寬)
        # 這裡我們傳入的是一個int不是list，官方會直接剪裁成正方形的大小[size, size]
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    # 從給定的大小中隨機選擇大小進行剪裁
    def __init__(self, min_size: int, max_size: int):
        # 設定最大以及最小值
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        # 從範圍內隨機選出高寬
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        # 調用官方的隨機剪裁，不過我們這裡不會直接剪圖片，而已要他返回怎麼剪
        # 也就是會返回剪裁後圖片左上角對應上原圖的哪裡，以及高寬(當然返回的高寬就是給的高寬)
        region = T.RandomCrop.get_params(img, [h, w])
        # 調用crop，對圖像剪裁同時也對gt_box做對應的改動
        return crop(img, target, region)


class CenterCrop(object):
    # 中心裁剪
    def __init__(self, size):
        # 設定剪裁後圖像大小
        self.size = size

    def __call__(self, img, target):
        # 取出原始圖像大小
        image_width, image_height = img.size
        # 最後輸出大小
        crop_height, crop_width = self.size
        # 上方以及左邊剪裁大小，會是需剪裁的一半另一半要留給右邊以及下面
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    # 左右翻轉
    def __init__(self, p=0.5):
        # 發生翻轉的概率
        self.p = p

    def __call__(self, img, target):
        # 翻轉
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    # 已看過
    # 在coco.py實例化
    def __init__(self, sizes, max_size=None):
        # sizes是個list或是tuple
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        # 預設是1333
        self.max_size = max_size

    def __call__(self, img, target=None):
        # 從sizes中選出一個大小
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    # 已看過
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        # 隨機選取在圖片下方或右方增加黑匡
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    # 已看過
    # 隨機選取transforms方式
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    # 已看過
    # 就是把圖片轉成tensor格式
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        # 已看過
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        # 將圖片通過官方normalize方法進行正則化
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        # 再來將gt_box轉換成(center_x, center_y, w, h)且為相對座標的形式
        # 記得要用深度copy
        target = target.copy()
        h, w = image.shape[-2:]
        # 取出target中boxes的資訊
        if "boxes" in target:
            boxes = target["boxes"]
            # (xmin, ymin, xmax, ymax) -> (center_x, center_y, w, h)
            boxes = box_xyxy_to_cxcywh(boxes)
            # 轉成相對座標
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            # 把值丟回去
            target["boxes"] = boxes
        return image, target


class Compose(object):
    # 已看過
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
