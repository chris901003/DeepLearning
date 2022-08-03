# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import numpy as np
import torch
from torch import Tensor

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['pixel_group'])


def pixel_group(
    score: Union[np.ndarray, Tensor],
    mask: Union[np.ndarray, Tensor],
    embedding: Union[np.ndarray, Tensor],
    kernel_label: Union[np.ndarray, Tensor],
    kernel_contour: Union[np.ndarray, Tensor],
    kernel_region_num: int,
    distance_threshold: float,
) -> List[List[float]]:
    """Group pixels into text instances, which is widely used text detection
    methods.

    Arguments:
        score (np.array or torch.Tensor): The foreground score with size hxw.
        mask (np.array or Tensor): The foreground mask with size hxw.
        embedding (np.array or torch.Tensor): The embedding with size hxwxc to
            distinguish instances.
        kernel_label (np.array or torch.Tensor): The instance kernel index with
            size hxw.
        kernel_contour (np.array or torch.Tensor): The kernel contour with
            size hxw.
        kernel_region_num (int): The instance kernel region number.
        distance_threshold (float): The embedding distance threshold between
            kernel and pixel in one instance.

    Returns:
        list[list[float]]: The instance coordinates and attributes list. Each
        element consists of averaged confidence, pixel number, and coordinates
        (x_i, y_i for all pixels) in order.
    """
    # 已看過，透過文字團中間線往外擴張，最終將預測為文字的地方分配一個文字團
    # score = 預測為文字的置信度，shape [height, width]
    # mask = 當預測置信度小於閾值的地方會是False，否則就會是True，shape [height, width]
    # embedding = 相似度向量，shape [height, width, channel=4]
    # kernel_label = 文字團中心團，每一個文字團會有自己的index，不在中心團當中的會是0，shape [height, width]
    # kernel_contour = 文字中心團的外圍邊匡，只有外圍邊匡會是255其他部分會是0，shape [height, width]
    # kernel_region_num = 總共有多少個文字團，int
    # distance_threshold = 距離的閾值，float

    # 檢查傳入的資料是否有型態上面的錯誤
    assert isinstance(score, (torch.Tensor, np.ndarray))
    assert isinstance(mask, (torch.Tensor, np.ndarray))
    assert isinstance(embedding, (torch.Tensor, np.ndarray))
    assert isinstance(kernel_label, (torch.Tensor, np.ndarray))
    assert isinstance(kernel_contour, (torch.Tensor, np.ndarray))
    assert isinstance(kernel_region_num, int)
    assert isinstance(distance_threshold, float)

    # 如果傳入的是ndarray就轉成tensor格式
    if isinstance(score, np.ndarray):
        score = torch.from_numpy(score)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    if isinstance(embedding, np.ndarray):
        embedding = torch.from_numpy(embedding)
    if isinstance(kernel_label, np.ndarray):
        kernel_label = torch.from_numpy(kernel_label)
    if isinstance(kernel_contour, np.ndarray):
        kernel_contour = torch.from_numpy(kernel_contour)

    if torch.__version__ == 'parrots':
        # 如果是用mmcv開發的底層就會到這裡
        label = ext_module.pixel_group(
            score,
            mask,
            embedding,
            kernel_label,
            kernel_contour,
            kernel_region_num=kernel_region_num,
            distance_threshold=distance_threshold)
        label = label.tolist()
        label = label[0]
        list_index = kernel_region_num
        pixel_assignment = []
        for x in range(kernel_region_num):
            pixel_assignment.append(
                np.array(
                    label[list_index:list_index + int(label[x])],
                    dtype=np.float))
            list_index = list_index + int(label[x])
    else:
        # 以pytorch為底層就會到這裡，這裡依舊使用了c++已經寫好的東西
        pixel_assignment = ext_module.pixel_group(score, mask, embedding,
                                                  kernel_label, kernel_contour,
                                                  kernel_region_num,
                                                  distance_threshold)
    # pixel_assignment = list[list]，第一個list會是總共有多少個文字團，第二個list會是一個文字團有哪些座標點
    return pixel_assignment
