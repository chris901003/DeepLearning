# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

import mmcv

try:
    import torch
except ImportError:
    torch = None


def tensor2imgs(tensor, mean=None, std=None, to_rgb=True):
    """Convert tensor to 3-channel images or 1-channel gray images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W). :math:`C` can be either 3 or 1.
        mean (tuple[float], optional): Mean of images. If None,
            (0, 0, 0) will be used for tensor with 3-channel,
            while (0, ) for tensor with 1-channel. Defaults to None.
        std (tuple[float], optional): Standard deviation of images. If None,
            (1, 1, 1) will be used for tensor with 3-channel,
            while (1, ) for tensor with 1-channel. Defaults to None.
        to_rgb (bool, optional): Whether the tensor was converted to RGB
            format in the first place. If so, convert it back to BGR.
            For the tensor with 1 channel, it must be False. Defaults to True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """
    # 已看過，主要是將圖像的均值方差調整到原始的樣子
    # tensor = 原始的圖像shape [batch_size, channel, height, width]
    # mean, std = 對原始圖像進行的均值方差調整
    # to_rgb = 是否將通道順序調整成變成RGB

    if torch is None:
        # 如果沒有安裝pytorch就會報錯，沒有安裝也不知道你怎麼可以跑到這
        raise RuntimeError('pytorch is not installed')
    # 檢查通道數量，這裡通道數量會有四個
    assert torch.is_tensor(tensor) and tensor.ndim == 4
    # 獲取channel維度深度
    channels = tensor.size(1)
    # channel會是1或是3，一個會是灰階另一個就是RGB
    assert channels in [1, 3]
    if mean is None:
        # 如果沒有傳入mean就將均值設定成0
        mean = (0, ) * channels
    if std is None:
        # 如果沒有傳入std就將方差設定成1
        std = (1, ) * channels
    assert (channels == len(mean) == len(std) == 3) or \
        (channels == len(mean) == len(std) == 1 and not to_rgb)

    # num_imgs = batch_size
    num_imgs = tensor.size(0)
    # 將均值以及方差變成ndarray格式
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    # 之後要回傳的資料
    imgs = []
    # 遍歷所有圖像
    for img_id in range(num_imgs):
        # 先將原始圖像從tensor轉成ndarray，shape [channel, height, width] -> [height, width, channel]
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        # 透過imdenormalize將均值方差調整回最原始的樣子
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        # 記錄下來
        imgs.append(np.ascontiguousarray(img))
    # 最後回傳回去
    return imgs
