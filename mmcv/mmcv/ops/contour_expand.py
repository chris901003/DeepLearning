# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import numpy as np
import torch

from ..utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['contour_expand'])


def contour_expand(kernel_mask: Union[np.array, torch.Tensor],
                   internal_kernel_label: Union[np.array, torch.Tensor],
                   min_kernel_area: int, kernel_num: int) -> list:
    """Expand kernel contours so that foreground pixels are assigned into
    instances.

    Args:
        kernel_mask (np.array or torch.Tensor): The instance kernel mask with
            size hxw.
        internal_kernel_label (np.array or torch.Tensor): The instance internal
            kernel label with size hxw.
        min_kernel_area (int): The minimum kernel area.
        kernel_num (int): The instance kernel number.

    Returns:
        list: The instance index map with size hxw.
    """
    # 已看過，擴展內核輪廓，將前景像素分配到實例
    # kernel_mask = 哪些地方的預測值有大於設定閾值，如果有大於就會是True，否則就會是False，shape [channel, height, width]
    # internal_kernel_label = 最一開始時的標註，用最小的kernel獲取的文字匡選處，不同匡選處會用不同index表示，沒有匡選到的會是0
    #                         shape [height, width]
    # min_kernel_ares = 最少kernel範圍
    # kernel_num = 一開始時總共有多少個連通塊，也就是有多少個不同組的文字匡

    # 檢查kernel_mask需要是tensor或是ndarray格式，kernel_mask shape [channel=7, height, width]
    assert isinstance(kernel_mask, (torch.Tensor, np.ndarray))
    # 檢查internal_kernel_label是否為tensor或是ndarray格式，internal_kernel_label shape [height, width]
    assert isinstance(internal_kernel_label, (torch.Tensor, np.ndarray))
    # 檢查min_kernel_area是否為int格式
    assert isinstance(min_kernel_area, int)
    # 檢查kernel_num數量
    assert isinstance(kernel_num, int)

    if isinstance(kernel_mask, np.ndarray):
        # 如果傳入的kernel_mask是ndarray就轉成tensor格式
        kernel_mask = torch.from_numpy(kernel_mask)
    if isinstance(internal_kernel_label, np.ndarray):
        # 如果傳入的internal_kernel_label是ndarray就轉成tensor格式
        internal_kernel_label = torch.from_numpy(internal_kernel_label)

    if torch.__version__ == 'parrots':
        # 如果底層使用的是parrots就會到這裡
        if kernel_mask.shape[0] == 0 or internal_kernel_label.shape[0] == 0:
            label = []
        else:
            label = ext_module.contour_expand(
                kernel_mask,
                internal_kernel_label,
                min_kernel_area=min_kernel_area,
                kernel_num=kernel_num)
            label = label.tolist()  # type: ignore
    else:
        # 如果底層使用的是pytorch就會到這裡，這裡會調用擴展的算法，但是因為該算法適用c++寫的並且包裝成python可以執行的樣子
        # 所以這裡我們無法進去看係項，依據論輪就是將kernel擴大時會有一些原先沒有被標記為文字的地方會被匡選到，此時依據最原始的
        # 文字匡選區域進行BFS依據誰先佔到該位置就會歸到哪個全組當中
        # label = list[list]，第一個list長度會是height，第二個list長度會是width
        label = ext_module.contour_expand(kernel_mask, internal_kernel_label,
                                          min_kernel_area, kernel_num)
    # 將label進行返回
    return label
