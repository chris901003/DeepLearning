# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        """
        :param num_convs: 卷積層堆疊數量
        :param kernel_size: 卷積核大小
        :param concat_input: 是否會將輸入的特徵圖與卷積後的特徵圖進行融合，也就是是否有殘差結構
        :param dilation: 膨脹係數
        :param kwargs: 其他參數
        """
        # 已看過

        # 檢查一些簡單的東西，正常都不會有問題
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        # 保存一些資料
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size

        # 繼承自BaseDecodeHead，與ASPP相同都先對繼承對象進行初始化
        super(FCNHead, self).__init__(**kwargs)
        if num_convs == 0:
            # 如果num_convs==0那麼輸入的channel需要跟輸出的channel深度相同
            assert self.in_channels == self.channels

        # 計算出卷積時需要的padding大小，確保輸入的特徵圖高寬透過卷積後不會改變
        conv_padding = (kernel_size // 2) * dilation
        # 卷積列表
        convs = []
        # 添加第一層的卷積層
        convs.append(
            # 使用卷積加標準化加激活函數層，在這裡就會將channel調整到輸出的channel深度
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        # 創建剩下的卷積層結構，與上方只差在輸入channel深度
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            # 如果num_convs=0表示不需要通過任何的卷積層，這裡就用nn.Identity表示
            # nn.Identity裏面就是x=y
            self.convs = nn.Identity()
        else:
            # 否則就將convs用nn.Sequential包裝起來
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            # 如果有需要將輸入與輸出進行融合就會進來
            # 融合後的特徵圖channel會需要進行調整，所以這裡透過ConvModule進行融合後的channel調整
            # 因為前面有用padding所以特徵圖的高寬是不會改變的
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # 已看過，進行向前傳遞，傳入的參數看下面的forward就可以
        # 這裡會透過transform_inputs對inputs做某些操作
        x = self._transform_inputs(inputs)
        feats = self.convs(x)
        if self.concat_input:
            # 如果有需要進行concat會在這裡進行拼接
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        # 已看過，這裡會是FCN的forward函數
        # inputs = tuple[tensor]，tensor shape [batch_size, channel, width, height]，tuple長度就會是輸入的特徵圖數量
        output = self._forward_feature(inputs)
        # 透過cls_seg會將channel調整到與num_classes相同，shape [batch_size, channel=num_classes, height, width]
        output = self.cls_seg(output)
        return output
