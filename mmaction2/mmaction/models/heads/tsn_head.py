# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import AvgConsensus, BaseHead


@HEADS.register_module()
class TSNHead(BaseHead):
    """Class head for TSN.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.4,
                 init_std=0.01,
                 **kwargs):
        """ 已看過，TSN專用的分類頭處理
        Args:
            num_classes: 分類類別數量
            in_channels: 輸入的channel深度
            loss_cls: 損失函數設定
            spatial_type: 空間方面處理的方式
            consensus: 共識設定資料
            dropout_ratio: dropout概率
            init_std: 初始化設定方式
            kwargs: 其他參數，通常為空
        """
        # 繼承自BaseHead，將繼承對象進行初始化
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        # 將傳入的資料進行保存
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        # 將consensus資料進行保存
        consensus_ = consensus.copy()

        # 將consensus的type提取出來
        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            # 如果consensus是AvgConsensus就會到這裡
            self.consensus = AvgConsensus(**consensus_)
        else:
            # 其他就會是這裡
            self.consensus = None

        if self.spatial_type == 'avg':
            # 如果spatial_type是選用avg就會將2D的特徵圖高寬透過池化變成1x1
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            # 否則avg_pool就設定成None
            self.avg_pool = None

        if self.dropout_ratio != 0:
            # 如果dropout率不是0，就會構建dropout實例化對象
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            # 否則就設定成None
            self.dropout = None
        # 透過全連接層將channel深度調整到分類類別數
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, num_segs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # 已看過，分類頭的forward函數
        # x = 特徵圖，tensor shape [batch_size * num_crops * num_clips * clip_len, channel, height, width]
        # num_segs = 一個影片總共有多少個片段

        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            # 如果avg_pool不是None就會到這裡，將空間維度壓縮成1x1大小
            if isinstance(x, tuple):
                # 如果x是tuple就會到這裡
                # 獲取每個特徵圖的shape資訊
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            # 將x通過avg_pool進行池化
            x = self.avg_pool(x)
            # [N * num_segs, in_channels, 1, 1]
        # 將x的通道進行改變 [batch_size, num_crops * num_clips * clip_len, channel, height, width]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        # 將x通過consensus將分類概率進行平均，x shape = [batch_size, 1, channel, height, width]
        x = self.consensus(x)
        # [N, 1, in_channels, 1, 1]
        # 調整x的通道，x shape = [batch_size, channel, height, width]
        x = x.squeeze(1)
        # [N, in_channels, 1, 1]
        if self.dropout is not None:
            # 如果有設定dropout就會到這裡
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        # 調整通道順序 [batch_size, channel]
        x = x.view(x.size(0), -1)
        # [N, in_channels]
        # 透過全連接層將channel深度調整成num_classes
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        # 最終回傳
        return cls_score
