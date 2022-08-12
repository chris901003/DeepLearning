# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class Maxpool2d(nn.Module):
    """A wrapper around nn.Maxpool2d().

    Args:
        kernel_size (int or tuple(int)): Kernel size for max pooling layer
        stride (int or tuple(int)): Stride for max pooling layer
        padding (int or tuple(int)): Padding for pooling layer
    """

    def __init__(self, kernel_size, stride, padding=0, **kwargs):
        super(Maxpool2d, self).__init__()
        self.model = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input feature map

        Returns:
            Tensor: The tensor after Maxpooling layer.
        """
        return self.model(x)


@PLUGIN_LAYERS.register_module()
class GCAModule(nn.Module):
    """GCAModule in MASTER.

    Args:
        in_channels (int): Channels of input tensor.
        ratio (float): Scale ratio of in_channels.
        n_head (int): Numbers of attention head.
        pooling_type (str): Spatial pooling type. Options are [``avg``,
            ``att``].
        scale_attn (bool): Whether to scale the attention map. Defaults to
            False.
        fusion_type (str): Fusion type of input and context. Options are
            [``channel_add``, ``channel_mul``, ``channel_concat``].
    """

    def __init__(self,
                 in_channels,
                 ratio,
                 n_head,
                 pooling_type='att',
                 scale_attn=False,
                 fusion_type='channel_add',
                 **kwargs):
        """ 已看過，在MASTER當中用到的GCAModule初始化部分
        Args:
            in_channels: 輸入的channel深度
            ratio: 對於in_channels的縮放比例
            n_head: 多頭注意力當中的頭數
            pooling_type: pooling的方式
            scale_attn: 是否要將注意力模塊的特徵圖進行縮放
            fusion_type: 上下文的融合方式
            kwargs: 其他參數，包括out_channels
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super(GCAModule, self).__init__()

        # 檢查pooling的方式是否為[avg, att]其中一個
        assert pooling_type in ['avg', 'att']
        # 檢查上下文合併的方式是否為[channel_add, channel_mul, channel_concat]其中一個
        assert fusion_type in ['channel_add', 'channel_mul', 'channel_concat']

        # in_channels must be divided by headers evenly
        # 檢查輸入的channel深度需要可以被頭數整除同時channel深度需要大於8
        assert in_channels % n_head == 0 and in_channels >= 8

        # 保存傳入參數
        self.n_head = n_head
        self.in_channels = in_channels
        self.ratio = ratio
        # 在特徵圖融合時，中間卷積層的channel深度
        self.planes = int(in_channels * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.scale_attn = scale_attn
        # 計算每個注意力頭當中的channel深度
        self.single_header_inplanes = int(in_channels / n_head)

        if pooling_type == 'att':
            # 如果pooling_type是att就會到這裡
            # 會有一個卷積層將每個頭當中的channel變成1
            self.conv_mask = nn.Conv2d(
                self.single_header_inplanes, 1, kernel_size=1)
            # 還會有一層softmax層結構
            self.softmax = nn.Softmax(dim=2)
        else:
            # 否則如果pooling_type是avg，就會使用AdaptiveAvgPool2d
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if fusion_type == 'channel_add':
            # 如果上下文的融合方式是用相加就會到這裡
            self.channel_add_conv = nn.Sequential(
                # 透過conv將channel變成planes
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                # 進行LN標準化後通過激活函數
                nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True),
                # 最後通過conv將channel深度調整回來
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
        elif fusion_type == 'channel_concat':
            # 如果上下文的融合方式是用拼接就會到這裡
            # 這裡與上面相同
            self.channel_concat_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))
            # for concat
            # 這裡會是concat特有的，需要將拼接後的channel深度調整回來
            self.cat_conv = nn.Conv2d(
                2 * self.in_channels, self.in_channels, kernel_size=1)
        elif fusion_type == 'channel_mul':
            # 如果上下文的融合方式是透過矩陣乘法就會到這裡
            # 這裡與add相同
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]), nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1))

    def spatial_pool(self, x):
        """ 已看過，進行空間性的池化
        Args:
            x: 特徵圖，tensor shape [batch_size, channel, height, width]
        """
        # 獲取傳入的特徵圖資訊
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            # 如果設定的pooling方式是att就會到這裡
            # [N*headers, C', H , W] C = headers * C'
            # 調整傳入的特徵圖的通道 [batch_size, channel, height, width]
            # -> [batch_size * n_head, channel_pre_head = channel / n_head, height, width]
            x = x.view(batch * self.n_head, self.single_header_inplanes,
                       height, width)
            # 將x的值給到input_x當中
            input_x = x

            # [N*headers, C', H * W] C = headers * C'
            # 再將input_x進行通道調整 [batch_size * n_head, channel_pre_head, height, width]
            # -> [batch_size * n_head, channel_pre_head, height * width]
            input_x = input_x.view(batch * self.n_head,
                                   self.single_header_inplanes, height * width)

            # [N*headers, 1, C', H * W]
            # 添加一個維度在第一維度上，shape [batch_size * n_head, 1, channel_pre_head, height * width]
            input_x = input_x.unsqueeze(1)
            # [N*headers, 1, H, W]
            # 將x通過conv_mask層結構，這裡會將channel調整到1
            # context_mask shape = [batch_size * n_head, channel=1, height, width]
            context_mask = self.conv_mask(x)
            # [N*headers, 1, H * W]
            # 調整通道 [batch_size * n_head, channel=1, height, width] -> [batch_size * n_head, channel=1, height * width]
            context_mask = context_mask.view(batch * self.n_head, 1,
                                             height * width)

            # scale variance
            if self.scale_attn and self.n_head > 1:
                # 如果有設定scale_attn以及head數量大於1，就會到這裡將context_mask的值進行調整
                context_mask = context_mask / \
                               torch.sqrt(self.single_header_inplanes)

            # [N*headers, 1, H * W]
            # 將context_mask在[height * width]的維度上面進行softmax
            context_mask = self.softmax(context_mask)

            # [N*headers, 1, H * W, 1]
            # 添加最後一個維度 [batch_size * n_head, channel=1, height * width, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N*headers, 1, C', 1] =
            # [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
            # 將input_x與context_mask進行矩陣相乘，最後context shape = [batch_size * n_head, 1, channel_pre_head, 1]
            context = torch.matmul(input_x, context_mask)

            # [N, headers * C', 1, 1]
            # 最後調整通道，shape = [batch_size, channel, 1, 1]
            context = context.view(batch,
                                   self.n_head * self.single_header_inplanes,
                                   1, 1)
        else:
            # [N, C, 1, 1]
            # 如果是用avg就會到這裡，直接透過avg_pool變成1*1的特徵圖
            context = self.avg_pool(x)

        # 回傳結果
        return context

    def forward(self, x):
        """ 已看過，GCAModule的正向傳播部分，這主要是給MASTER模型使用的
        Args:
            x: 特徵圖，tensor shape [batch_size, channel, height, width]
        """
        # [N, C, 1, 1]
        # 將特徵圖通過一個空間性的池化，這裡對應到論文的模塊會是Multi-Aspect Context，在進行池化的過程中還有進行注意力機制
        context = self.spatial_pool(x)
        # 保存一份x，這裡是要作為殘差結構用的
        out = x

        if self.fusion_type == 'channel_mul':
            # 如果融合的方式是透過相乘就會到這裡
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            # 如果融合的方式是透過相加就會到這裡
            # [N, C, 1, 1]
            # 將context分別通過[卷積 -> LN標準化 -> 激活函數 -> 卷積]
            channel_add_term = self.channel_add_conv(context)
            # 最後將out與channel_add_term進行相加
            out = out + channel_add_term
        else:
            # 如果融合的方式是透過拼接就會到這裡
            # [N, C, 1, 1]
            channel_concat_term = self.channel_concat_conv(context)

            # use concat
            _, C1, _, _ = channel_concat_term.shape
            N, C2, H, W = out.shape

            out = torch.cat([out,
                             channel_concat_term.expand(-1, -1, H, W)],
                            dim=1)
            out = self.cat_conv(out)
            out = nn.functional.layer_norm(out, [self.in_channels, H, W])
            out = nn.functional.relu(out)

        # 最終將結果進行回傳
        return out
