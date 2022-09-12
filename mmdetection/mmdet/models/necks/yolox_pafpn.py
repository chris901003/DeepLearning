# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS
from ..utils import CSPLayer


@NECKS.register_module()
class YOLOXPAFPN(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        """ PAFPN標註匡解碼頭初始化設定，這裡是專門給yolox使用
        Args:
            in_channels: 輸入的channel深度，這裡會是list因為會有多層backbone輸出的結果
            out_channels: 輸出的channel深度，這裡只會有一個值
            num_csp_blocks: 在csp當中瓶頸結構堆疊數量，會因為使用不同大小的模型會有不同堆疊數量
            use_depthwise: 使否使用dw卷積
            upsample_cfg: 上採樣方式設定
            conv_cfg: 卷積核設定資料
            norm_cfg: 標準化層設定
            act_cfg: 激活函數層設定
            init_cfg: 初始化設定
        """
        # 繼承自BaseModule，將繼承對象進行初始化
        super(YOLOXPAFPN, self).__init__(init_cfg)
        # 保存輸入channel深度以及輸出channel深度
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 根據是否使用dw卷積給定卷積類
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        # 構建上採樣方式，這裡會直接使用torch官方的上採樣方式
        self.upsample = nn.Upsample(**upsample_cfg)
        # 構建下採樣層結構
        self.reduce_layers = nn.ModuleList()
        # 構建由上往下的結構
        self.top_down_blocks = nn.ModuleList()
        # 構建從backbone輸出後須通過的卷積層，這裡不會包含最後一層的backbone輸出結果
        # 這裡構建的會是一路進行上採樣的結構
        for idx in range(len(in_channels) - 1, 0, -1):
            # 在reduce_layers添加卷積結構
            self.reduce_layers.append(
                ConvModule(
                    # 該輸出的channel深度
                    in_channels[idx],
                    # 將channel深度調整到與上層輸出channel深度相同
                    # 也就是將深層特徵圖的channel調整到與淺層特徵圖的channel深度相同
                    in_channels[idx - 1],
                    # 這裡使用的是1x1的卷積核
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            # 在top_down添加CSP層結構
            self.top_down_blocks.append(
                CSPLayer(
                    # 輸入的channel會是兩個特徵圖進行concat後的channel深度
                    in_channels[idx - 1] * 2,
                    # 輸出的channel
                    in_channels[idx - 1],
                    # 這裡瓶頸結構堆疊數量都會一致
                    num_blocks=num_csp_blocks,
                    # 不使用殘差結構
                    add_identity=False,
                    # 傳入是否使用dw卷積
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # build bottom-up blocks
        # 這裡構建的會是一路向下採樣的結構
        # 存放下採樣模塊
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        # 構建一系列往下層結構
        for idx in range(len(in_channels) - 1):
            # 在下採樣模塊當中添加卷積層，這裡就會根據是否使用dw卷積會有不同
            self.downsamples.append(
                conv(
                    # 輸入以及輸出channel不會發生變化
                    in_channels[idx],
                    in_channels[idx],
                    # 使用3x3卷積
                    3,
                    # 這裡步距會是2，高寬會變成原先的一半
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            # 構建CSP層結構
            self.bottom_up_blocks.append(
                CSPLayer(
                    # 這裡會有從左邊融合過來的特徵層，所以channel深度會翻倍
                    in_channels[idx] * 2,
                    # 輸出的channel深度
                    in_channels[idx + 1],
                    # 統一的瓶頸結構堆疊數量
                    num_blocks=num_csp_blocks,
                    # 不使用殘差邊
                    add_identity=False,
                    # 是否使用dw卷積
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # 輸出時會使用的卷積模塊
        self.out_convs = nn.ModuleList()
        # 這裡輸出的頭數就會是與輸入的特徵圖數量相同
        for i in range(len(in_channels)):
            self.out_convs.append(
                # 進行卷積
                ConvModule(
                    # 將輸出channel調整到指定輸出channel深度
                    in_channels[i],
                    out_channels,
                    # 使用1x1卷積核進行調整
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        # PAFPN解碼頭的forward函數部分
        # inputs = tuple[tensor]，tuple長度會是從backbone輸出的數量，tensor shape [batch_size, channel, height, width]
        # 檢查傳入的特徵圖數量與初始化設定時的in_channels的長度是否相同
        assert len(inputs) == len(self.in_channels)

        # top-down path
        # 這裡最後一層backbone輸出會直接使用
        inner_outs = [inputs[-1]]
        # 從底層開始往上融合以及提取特徵
        for idx in range(len(self.in_channels) - 1, 0, -1):
            # 獲取較高維的特徵圖
            feat_heigh = inner_outs[0]
            # 獲取較低維的特徵圖
            feat_low = inputs[idx - 1]
            # 先將高維特徵圖通過reduce_layers，透過1x1的卷積核將channel調整到與低維的特徵圖相同的channel深度
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_heigh)
            # 將調整過channel深度的特徵圖更新到inner_outs上
            inner_outs[0] = feat_heigh

            # 透過上採樣將高維度的特徵圖進行高寬放大
            upsample_feat = self.upsample(feat_heigh)

            # 將高維度特徵圖與低維度特徵圖在第二個維度進行拼接，之後再通過一系列卷積層
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], 1))
            # 將新的特徵圖放到inner_outs的最上面
            inner_outs.insert(0, inner_out)

        # bottom-up path
        # 接下來開始一路往下下採樣
        outs = [inner_outs[0]]
        # 遍歷下採樣層
        for idx in range(len(self.in_channels) - 1):
            # 獲取較低維度特徵圖，會保存在outs的最後一個
            feat_low = outs[-1]
            # 獲取較高維度的特徵圖，會放在inner_outs當中
            feat_height = inner_outs[idx + 1]
            # 透過down samples進行下採樣
            downsample_feat = self.downsamples[idx](feat_low)
            # 最後將高維度以及低維度特徵圖進行拼接後使用瓶頸結構卷積層
            out = self.bottom_up_blocks[idx](torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            # 最後會在輸出部分通過卷積層
            outs[idx] = conv(outs[idx])

        # 回傳結果，list[tensor]，list長度會是分類頭的數量，tensor shape [batch_size, channel, height, width]
        return tuple(outs)
