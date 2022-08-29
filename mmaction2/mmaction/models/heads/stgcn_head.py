# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class STGCNHead(BaseHead):
    """The classification head for STGCN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        num_person (int): Number of person. Default: 2.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 num_person=2,
                 init_std=0.01,
                 **kwargs):
        """ STGCN的分類頭
        Args:
            num_classes: 分類類別數
            in_channels: 輸入的channel深度
            loss_cls: 損失計算設定資料
            spatial_type: 獲取最終預測置信度方式
            num_person: 最多有多少人
            init_std: 初始化設定
            kwargs: 其他參數，通常為空
        """
        # 繼承自BaseHead，將繼承對象進行初始化
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        # 保存傳入參數
        self.spatial_type = spatial_type
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_person = num_person
        self.init_std = init_std

        # 先將pool設定成None
        self.pool = None
        if self.spatial_type == 'avg':
            # 如果是用avg這裡的pool就是用avg
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.spatial_type == 'max':
            # 如果是用max這裡的pool就是用max
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError

        # 最後接上全連接層將channel維度調整到分類類別數
        self.fc = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)

    def init_weights(self):
        normal_init(self.fc, std=self.init_std)

    def forward(self, x):
        # global pooling
        # 將STGCN的結果透過分類頭進行分類
        # x shape = [batch_size * people, channel, frames, num_node]

        # 檢查pool方式不可以為None
        assert self.pool is not None
        # 將x進行pool，x shape [batch_size * people, channel, 1, 1]
        x = self.pool(x)
        # 先進行通道調整 [batch_size, people, channel, 1, 1]，之後對第二個維度取均值，也就是平均兩個物體的均值
        # x shape [batch_size, channel, 1, 1]
        x = x.view(x.shape[0] // self.num_person, self.num_person, -1, 1, 1).mean(dim=1)

        # prediction
        # 透過全連接層將channel維度調整到分類類別數，shape [batch_size, num_classes, 1, 1]
        x = self.fc(x)
        # 最後調整通道，x shape [batch_size, num_classes]
        x = x.view(x.shape[0], -1)

        # 回傳結果
        return x
