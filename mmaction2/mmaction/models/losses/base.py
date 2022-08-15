# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):
    """Base class for loss.

    All subclass should overwrite the ``_forward()`` method which returns the
    normal loss without loss weights.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call.

        Args:
            *args: The positional arguments for the corresponding
                loss.
            **kwargs: The keyword arguments for the corresponding
                loss.

        Returns:
            torch.Tensor: The calculated loss.
        """
        # 已看過，使用交叉熵計算分類類別損失
        # ret = 計算出來的交叉熵損失
        ret = self._forward(*args, **kwargs)
        if isinstance(ret, dict):
            # 如果ret是dict格式就會到這裡
            for k in ret:
                if 'loss' in k:
                    ret[k] *= self.loss_weight
        else:
            # 如果只是單一值就乘上損失權重
            ret *= self.loss_weight
        # 最後將ret回傳
        return ret
