import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    # 已看過
    # 大於size不會進行任何處理
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    # 對齊左上角有小於size的部分會在右側或是下方進行填充
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        # 已看過
        self.transforms = transforms

    def __call__(self, image, target):
        # 已看過
        # 將所有transforms裡面實例化好的變化方式跑一次__call__函數
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        # 已看過
        self.min_size = min_size
        # 沒有最大值就讓最小值與最大值相同
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        # 已看過
        # 從最小到最大之間隨機找一個大小
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小，大邊可能會大於size
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        # target也要跟著一起縮放
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        # 已看過
        # flip_prob = 水平翻轉概率
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        # 已看過
        # 有機率性的水平翻轉，記得target也要一起翻轉
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        # 已看過
        # size = 最終輸入到網路的圖片大小
        self.size = size

    def __call__(self, image, target):
        # 已看過
        # 將圖像長或寬不到size的部分進行填充，填充只會在右側或是下方
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        # 進行隨機裁減，這時圖片高寬一定大於或等於size，然後對圖片剪裁這裡我們只拿到左上角的座標位置
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        # 依據給定的crop_params會進行剪裁，這時image與target會依照相同方式剪裁同時大小都是size
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        # 回傳剪裁好的圖像
        return image, target


class CenterCrop(object):
    # 直接按照中心剪裁，這裡我們沒有用到
    def __init__(self, size):
        # 已看過
        self.size = size

    def __call__(self, image, target):
        # 已看過
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        # 已看過
        # to_tensor會將Image格式轉換成tensor格式同時也會將圖片的值從[0, 255]轉成[0, 1]
        image = F.to_tensor(image)
        # target只需要將格式轉成tensor就可以了，不需要做其他操作
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        # 將轉換成tensor格式的資料回傳
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        # 已看過
        # 記錄下均值與方差
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        # 已看過
        # 調整圖片均值與方差
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
