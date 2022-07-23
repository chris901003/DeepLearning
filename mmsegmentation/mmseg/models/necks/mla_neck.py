# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, build_norm_layer

from ..builder import NECKS


class MLAModule(nn.Module):

    def __init__(self,
                 in_channels=[1024, 1024, 1024, 1024],
                 out_channels=256,
                 norm_cfg=None,
                 act_cfg=None):
        """ 已看過，MLA模塊，會將多個輸出整理變成一個輸出，整合來自不同層的encoder輸出
        Args:
            in_channels: 輸入的channel深度，這裡會是list表示個特徵圖的深度
            out_channels: 輸出的channel深度
            norm_cfg: 標準化層的設定
            act_cfg: 激活函數的設定
        """

        # 這裡繼承於nn.Module
        super(MLAModule, self).__init__()
        # 紀錄一系列層結構
        self.channel_proj = nn.ModuleList()
        # 遍歷輸入的層結構數量
        for i in range(len(in_channels)):
            self.channel_proj.append(
                # 使用卷積模塊調整channel深度，將深度調整到out_channel
                ConvModule(
                    in_channels=in_channels[i],
                    out_channels=out_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        # 這裡還會將最後全部融合後提取特徵
        self.feat_extract = nn.ModuleList()
        # 遍歷所有輸入層
        for i in range(len(in_channels)):
            self.feat_extract.append(
                ConvModule(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        """ 已看過，mla的forward函數
        Args:
            inputs: list[tensor]，tensor shape [batch_size, channel, height, width]
                這裡的inputs將backbone的輸出進行標準化後的結果
        """

        # feat_list -> [p2, p3, p4, p5]
        # 將input透過conv後的結果存放位置
        feat_list = []
        # 遍歷輸入
        for x, conv in zip(inputs, self.channel_proj):
            # 透過一個conv層結構調整channel，channel從1024到256
            feat_list.append(conv(x))

        # feat_list -> [p5, p4, p3, p2]
        # mid_list -> [m5, m4, m3, m2]
        # 將feat_list倒序，將較高層次的特徵圖放在最前面
        feat_list = feat_list[::-1]
        mid_list = []
        for feat in feat_list:
            if len(mid_list) == 0:
                # 如果是第一個就直接放到mid_list當中
                mid_list.append(feat)
            else:
                # 如果不是第一個就會將上一個加上當前的特徵圖記錄到mid_list當中
                mid_list.append(mid_list[-1] + feat)

        # mid_list -> [m5, m4, m3, m2]
        # out_list -> [o2, o3, o4, o5]
        # 最後回傳的內容
        out_list = []
        # 遍歷中間層，這些中間層都有混合上層的結構
        for mid, conv in zip(mid_list, self.feat_extract):
            out_list.append(conv(mid))

        return tuple(out_list)


@NECKS.register_module()
class MLANeck(nn.Module):
    """Multi-level Feature Aggregation.

    This neck is `The Multi-level Feature Aggregation construction of
    SETR <https://arxiv.org/abs/2012.15840>`_.


    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        norm_layer (dict): Config dict for input normalization.
            Default: norm_layer=dict(type='LN', eps=1e-6, requires_grad=True).
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer=dict(type='LN', eps=1e-6, requires_grad=True),
                 norm_cfg=None,
                 act_cfg=None):
        """ 已看過，這裡就是setr的專屬neck模塊
        Args:
            in_channels: 輸入的channel深度，這裡會是list，因為總共會從backbone中取出4個transformer encoder的層輸出
            out_channels: 輸出的channel深度，這裡會將輸入的4個特徵圖整合成一個，所以這裡只會指定最後的輸出channel
            norm_layer: 標準化層的設定，會將一開始傳入的資料進行標準化
            norm_cfg: 標準化層的設定，給MLAModule使用的
            act_cfg: 激活函數的設定
        """

        # 這裡繼承於nn.Module
        super(MLANeck, self).__init__()
        # 在這裡輸入的in_channels一定會是list，因會選用MLANeck就是要將多個backbone輸出進行融合
        assert isinstance(in_channels, list)
        # 保存一些傳入的參數
        self.in_channels = in_channels
        self.out_channels = out_channels

        # In order to build general vision transformer backbone, we have to
        # move MLA to neck.
        # 構建一開始會需要的標準化層，這裡用的是norm_layer設定資料
        self.norm = nn.ModuleList([
            build_norm_layer(norm_layer, in_channels[i])[1]
            for i in range(len(in_channels))
        ])

        # 構建MLAModule實例對象
        self.mla = MLAModule(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, inputs):
        """ 已看過，這裡就是neck模塊
        Args:
            inputs: list[tensor]，list的長度就會是從backbone輸出多少層
        """

        # in_channels的長度需要與inputs的長度相同，這樣才會知道每一層輸入的channel深度
        assert len(inputs) == len(self.in_channels)

        # Convert from nchw to nlc，調整shape [batch_size, channel, height, width] -> [batch_size, height * width, channel]
        # outs記錄下結果
        outs = []
        # 遍歷輸入的層數
        for i in range(len(inputs)):
            # 將輸出提取出來
            x = inputs[i]
            # 獲取相關shape
            n, c, h, w = x.shape
            # 調整通道排序，shape [batch_size, channel, height, width] -> [batch_size, height * width, channel]
            x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
            # 透過標準化層，這裡不同層的輸出都有獨立的標準化層表示參數不共用
            x = self.norm[i](x)
            # 將通道排序調整回來
            x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
            # 將結果存下來
            outs.append(x)

        # 最後將結果傳入到mla層結構當中
        outs = self.mla(outs)
        # outs shape = list[tensor]，tensor shape [batch_size, channel, height, width]，list長度與backbone輸出的特徵層相同
        return tuple(outs)
