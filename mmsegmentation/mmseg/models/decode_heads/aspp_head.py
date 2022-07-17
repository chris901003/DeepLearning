# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class ASPPModule(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        """
        :param dilations: 膨脹係數列表
        :param in_channels: 輸入的channel
        :param channels: 輸出的channel
        :param conv_cfg: 卷積層的相關設定
        :param norm_cfg: 標準化層的相關設定
        :param act_cfg: 激活函數的相關設定
        """
        # 已看過

        # 繼承自torch.nn.ModuleList
        super(ASPPModule, self).__init__()
        # 保留參數
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # 遍歷所有的膨脹係數
        for dilation in dilations:
            self.append(
                ConvModule(
                    self.in_channels,
                    self.channels,
                    # 對於卷積核有一些調整
                    1 if dilation == 1 else 3,
                    dilation=dilation,
                    # padding也會根據膨脹係數調整
                    padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        for aspp_module in self:
            aspp_outs.append(aspp_module(x))

        return aspp_outs


@HEADS.register_module()
class ASPPHead(BaseDecodeHead):
    """Rethinking Atrous Convolution for Semantic Image Segmentation.

    This head is the implementation of `DeepLabV3
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """

    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        """
        :param dilations: 膨脹卷積係數，這裡的ASPP是透過不同的膨脹係數組合成不同感受野的特徵圖
        :param kwargs: 其他的一些參數包括[輸入channel, 輸出channel, loss_decode損失計算方式, ...]
        """
        # 已看過

        # 繼承於BaseDecodeHead，解碼頭的最底層
        # BaseDecodeHead裏面決定將那些特徵圖進行融合同時可以設定融合模式，最後透過卷積將channel變成分類類別數，同時也會有損失計算在裡面
        # 總之BaseDecodeHead做了最後一步的處理
        super(ASPPHead, self).__init__(**kwargs)
        # dilations膨脹係數需要是list或是tuple格式
        assert isinstance(dilations, (list, tuple))
        # 保存dilations參數
        self.dilations = dilations
        # ASPP當中有一個分支就是將特徵圖變成1*1大小的，self.image_pool就是這個分支
        self.image_pool = nn.Sequential(
            # 透過nn.Sequential將一系列操作包在裡面
            # 透過nn.AdaptiveAvgPool2d可以將任何高寬的特徵圖透過平均計算變成1*1的特徵圖
            nn.AdaptiveAvgPool2d(1),
            # 將高寬為1*1的特徵圖通過一個卷積加標準化加激活層
            ConvModule(
                self.in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        # 其餘的ASPP結構部分，這裡預設的會是DeepLab_V3的ASPP的dilations參數
        self.aspp_modules = ASPPModule(
            dilations,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # self.bottleneck = 將ASPP的特徵層融合後透過一個卷積調整channel深度，同時也有融合的效果
        self.bottleneck = ConvModule(
            (len(dilations) + 1) * self.channels,
            self.channels,
            3,
            padding=1,
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
        # 已看過
        # inputs = 多層特徵層，有不同的維度以及尺度的特徵圖

        # x = 將inputs根據融合方式或是指定某一層inputs的結果
        x = self._transform_inputs(inputs)
        # aspp輸出
        aspp_outs = [
            # 這裡會先將x放入到aspp當中pool的層結構，之後透過resize將大小調整回x相同的高寬且調整的方式是使用雙線性差值
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        # 將剩下的aspp模塊進行向前傳遞，將結果放到aspp_out當中
        aspp_outs.extend(self.aspp_modules(x))
        # 將aspp當中的特徵圖在channel維度方向進行拼接
        aspp_outs = torch.cat(aspp_outs, dim=1)
        # 透過一個瓶頸結構的卷積層將aspp當中的特徵圖進行融合，同時調整channel深度
        feats = self.bottleneck(aspp_outs)
        # return = tensor格式shape [batch_size, channel, height, width]
        return feats

    def forward(self, inputs):
        """Forward function."""
        # 已看過
        # inputs = 多層特徵層，有不同的維度以及尺度的特徵圖

        # output = tensor格式shape [batch_size, channel, height, width]
        output = self._forward_feature(inputs)
        # 透過cls_seg將channel調整到與num_classes相同，作為最後的輸出
        # output shape = [batch_size, channel=num_classes, height, width]
        output = self.cls_seg(output)
        return output
