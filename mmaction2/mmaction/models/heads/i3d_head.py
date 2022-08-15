# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class I3DHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        """ 已看過，I3D的分類頭初始化函數
        Args:
            num_classes: 分類類別數，這裡如果使用數據及為kinetics400就會是400
            in_channels: 輸入的channel深度
            loss_cls: 損失計算方式
            spatial_type: 在空間上的池化方式
            dropout_ratio: dropout概率
            init_std: 初始化資料
            kwargs: 其他參數
        """
        # 繼承自BaseHead，將繼承對象進行初始化
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        # 保存傳入的參數
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            # 如果有設定dropout的概率就會實例化一個dropout層
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            # 否則就設定成None
            self.dropout = None
        # 最後通過一個linear層將channel設定成最終分類數量
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            # 如果空間上面的池化方式是avg就會構建一個AdaptiveAvgPool層結構，通過後直接將時間與空間維度壓成1x1x1
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            # 否則就設定成None
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
        # 已看過，這裡是I3D的解碼頭，最終會將channel調整到與分類類別數相同
        # x = 經過特徵提取的圖像資料，tensor shape [batch_size * num_clips, channel=2048, clip_len=4, height=7, width=7]

        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            # 如果spatial_type是avg就會到這裡進行池化
            # x shape = [batch_size * num_clips, channel=2048, clip_len=1, height=1, width=1]
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            # 如果有設定dropout層就會到這裡進行dropout
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        # 將通道進行排列 [batch_size * num_clips, channel * clip_len * height * width = 2048]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        # 通過全連接層進行分類，cls_score shape = [batch_size * num_clips, num_classes]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        # 最後將結果回傳
        return cls_score
