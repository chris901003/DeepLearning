# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import PIPELINES

from . import PANetTargets


@PIPELINES.register_module()
class PSENetTargets(PANetTargets):
    """Generate the ground truth targets of PSENet: Shape robust text detection
    with progressive scale expansion network.

    [https://arxiv.org/abs/1903.12473]. This code is partially adapted from
    https://github.com/whai362/PSENet.

    Args:
        shrink_ratio(tuple(float)): The ratios for shrinking text instances.
        max_shrink(int): The maximum shrinking distance.
    """

    def __init__(self,
                 shrink_ratio=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4),
                 max_shrink=20):
        """ 已看過，對於標註圖像會進行縮放，為了計算不同層的損失值
        Args:
            shrink_ratio: 不同層的標註縮放比例
            max_shrink: 最大縮放距離
        """
        # 繼承自PANetTargets，對繼承對象進行初始化
        super().__init__(shrink_ratio=shrink_ratio, max_shrink=max_shrink)
