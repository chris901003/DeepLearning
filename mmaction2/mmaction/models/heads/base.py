# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

from ...core import top_k_accuracy
from ..builder import build_loss


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``forward``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
        topk (int | tuple): Top-k accuracy. Default: (1, 5).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0,
                 topk=(1, 5)):
        """ 已看過，分類頭的祖先類別初始化函數
        Args:
            num_classes: 分類類別數量
            in_channels: 輸入的channel深度
            loss_cls: 損失計算資訊
            multi_class: 是否為多類別的分類任務
            label_smooth_eps: 對標籤進行平滑處理時的超參數
            topk: 取出置信度前幾大的
        """
        # 繼承於nn.Module，將繼承對象進行初始化
        super().__init__()
        # 保存傳入參數
        self.num_classes = num_classes
        self.in_channels = in_channels
        # 構建損失計算實例對象
        self.loss_cls = build_loss(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        # 檢查topk要是int或是tuple格式
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            # 如果topk是int格式就在外面加上tuple
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        # 保存topk
        self.topk = topk

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.

        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.

        Returns:
            dict: A dict containing field 'loss_cls'(mandatory)
            and 'topk_acc'(optional).
        """
        # 已看過，計算損失值的
        # cls_score = 預測結果，tensor shape [batch_size * num_clips, num_classes]
        # labels = 標註訊息，tensor shape [batch_size]

        # 構建損失的dict
        losses = dict()
        if labels.shape == torch.Size([]):
            # 如果labels的shape是torch.Size([])就會在最前面加上一個維度
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size():
            # 如果不是一個影像有多個類別且預測的size與標註的size相同，就會到這裡
            # 將預測的結果以及標註訊息都轉成ndarray並且將topk資訊傳入到top_k_accuracy當中
            # topk_k_acc shape = [batch_size]
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            # 遍歷在不同topk底下計算的正確率
            for k, a in zip(self.topk, top_k_acc):
                # 將損失值轉成tensor格式
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        # 計算分類類別損失
        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            # 如果loss_cls是dict格式就透過update放到losses當中
            losses.update(loss_cls)
        else:
            # 如果是單一值就直接放到losses當中
            losses['loss_cls'] = loss_cls

        # 回傳losses
        return losses
