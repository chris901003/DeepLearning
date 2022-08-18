# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class SlowFastHead(BaseHead):
    """The classification head for SlowFast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.8.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.8,
                 init_std=0.01,
                 **kwargs):
        """ 已看過，構建SlowFast分類頭
        Args:
            num_classes: 分類類別數
            in_channels: 輸入的channel深度
            loss_cls: 分類損失構建方式
            spatial_type: 融合空間資訊的方式
            dropout_ratio: dropout概率
            init_std: 初始化設定
        """

        # 繼承自BaseHead，將繼承對象進行初始化
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        # 保存傳入資料
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            # 如果dropout概率不是0就構建dropout實例化對象
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            # 否則就將dropout設定成None
            self.dropout = None
        # 構建全連接層，將channel深度調整到分類類別數
        self.fc_cls = nn.Linear(in_channels, num_classes)

        if self.spatial_type == 'avg':
            # 如果時間與空間維度上的統整方式是avg就會透過池化方式進行
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            # 其他就設定成None
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # 已看過，SlowFast的解碼頭
        # x = tuple(slow_tensor, fast_tensor)，這裡分別會有slow的輸出特徵圖，以及fast的輸出特徵圖
        # ([N, channel_fast, T, H, W], [(N, channel_slow, T, H, W)])
        # 將fast以及slow的特徵圖進行提取，tensor shape [batch_size, channel, num_clip, height, width]
        x_fast, x_slow = x
        # ([N, channel_fast, 1, 1, 1], [N, channel_slow, 1, 1, 1])
        # 將fast以及slow的特徵圖通過池化將時間以及空間維度壓縮，最後都會是1x1x1的狀態
        x_fast = self.avg_pool(x_fast)
        x_slow = self.avg_pool(x_slow)
        # [N, channel_fast + channel_slow, 1, 1, 1]
        # 使用concat將slow以及fast進行融合，x shape = [batch_size, channel_slow + channel_fast, 1, 1, 1]
        x = torch.cat((x_slow, x_fast), dim=1)

        if self.dropout is not None:
            # 如果有設定dropout就會到這裡
            x = self.dropout(x)

        # [N x C]
        # 最後調整通道，將時空維度進行壓縮，x shape [batch_size, channel]
        x = x.view(x.size(0), -1)
        # [N x num_classes]
        # 通過全連接層將channel深度調整到num_classes
        # cls_score shape = [batch_size, num_classes]
        cls_score = self.fc_cls(x)

        # 最終回傳
        return cls_score
