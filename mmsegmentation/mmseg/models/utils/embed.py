# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Sequence

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner.base_module import BaseModule
from mmcv.utils import to_2tuple


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """
    # 主要就是在進行填充的，只是這裡的兩種填充方式官方沒有

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        """
        Args:
             kernel_size: 卷積核的大小，用來判斷padding應該要是多少
             stride: 步距，用來判斷padding應該要是多少
             dilation: 膨漲係數，用來判斷padding應該要是多少
             padding: padding的模式，這裡有兩種[same, corner]
                same就是在四周進行填充並且保證經過卷積後高寬與填充前相同
                corner就是只會在右側以及下方進行填充，同樣在經過卷積後高寬不會發生變化
        """
        # 已看過

        super(AdaptivePadding, self).__init__()

        # padding模式目前只有支援兩種
        assert padding in ('same', 'corner')

        # 將卷積核以及步距以及膨脹係數進行數據調整
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        # 將數據保存下來
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        # 已看過，計算需要的padding大小
        # input_shape = [height, width]

        # 將高寬取出
        input_h, input_w = input_shape
        # 拿出卷積核大小
        kernel_h, kernel_w = self.kernel_size
        # 拿出步距
        stride_h, stride_w = self.stride
        # 透過公式計算出輸出的大小，之後再計算出需要多少的padding才不會改變原先的高寬
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        # 將需要padding的大小進行返回
        return pad_h, pad_w

    def forward(self, x):
        # 已看過，在進行patch_embed的卷積前會先進來做padding
        # x = 原始訓練圖像，shape [batch_size, channel=3, height, width]

        # 透過get_pad_shape獲取需要的padding大小
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        # 如果回傳的pad_h或是pad_w大於0表示需要進行padding
        if pad_h > 0 or pad_w > 0:
            # 依據不同的padding方法進行padding
            if self.padding == 'corner':
                # 只會將在下方以及右方進行padding
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                # 在四周進行等寬度的paddding
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        # 回傳padding結果
        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The config dict for embedding
            conv layer type selection. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int, optional): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        """
        Args:
             in_channels: 輸入channel的深度
             embed_dims: 每一個特徵點會用多少維度的向量進行表示
             conv_type: 使用哪種的卷積，這裡預設會使用2d的卷積
             kernel_size: 卷積核的大小，也就是多大的高寬會是一個patch
             stride: 步距，基本上會與kernel_size相同大小，這樣才符合patch的意思
             padding: 填充的方式，這裡預設會是corner
             dilation: 膨脹係數，預設為1
             bias: 是否啟用偏置
             norm_cfg: 如果有需要進行標準化就會說明要用哪種標準化層
             input_size: 輸入的大小，預設為None
             init_cfg: 初始化參數
        """
        # 已看過，透過卷積的方式進行對原始圖像Patch

        # 初始化繼承對象
        super(PatchEmbed, self).__init__(init_cfg=init_cfg)

        # 記錄下參數
        self.embed_dims = embed_dims
        if stride is None:
            # 如果沒有指定步距，這裡會默認與kernel_size相同
            stride = kernel_size

        # 將卷積核以及步距以及膨漲係數進行調整， int -> (int, int)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            # 當padding傳入的是str格式，我們就會調用AdaptivePadding，並且將結果存在adap_padding
            # self.adap_padding會在進行卷積前先呼叫，所以會先進行padding才會做卷積
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            # 同時在卷積部分我們就不會使用padding
            padding = 0
        else:
            # 其他狀況adap_padding就會是None
            self.adap_padding = None
        # 調整padding的格式
        padding = to_2tuple(padding)

        # 構建卷積的實例化對象，之後將實例化對象傳到projection當中
        # 這裡如果是要Conv2d就會直接回傳torch.nn.Conv2d的實例對象
        self.projection = build_conv_layer(
            # 將我們需要的卷積類型傳入
            dict(type=conv_type),
            # 其他卷積相關參數
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            # 如果需要標準化層結構就會在這裡進行實例化
            #  build_norm_layer會回傳名稱以及實例化對象，這裡我們只需要實例化對象就可以了，所以取用[1]的資料
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            # 否則就將標準化層放空
            self.norm = None

        if input_size:
            # 如果有輸入input_size才會進來
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            # 這裡我們沒有輸入input_size
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """
        # 已看過，patch_embed的forward函數
        # x = 原始訓練圖像，shape [batch_size, channel=3, height, width]

        # 如果有設定特殊的padding方式會先進行padding
        if self.adap_padding:
            x = self.adap_padding(x)

        # 透過卷積操作進行patch_embed
        # x shape = [batch_size, channel=embed_dim, ori_height / kernel_size, ori_weight / kernel_size]
        x = self.projection(x)
        # out_size = [ori_height / kernel_size, ori_weight / kernel_size]
        out_size = (x.shape[2], x.shape[3])
        # [batch_size, channel, height, width] -> [batch_size, channel, height * width]
        # -> [batch_size, height * width, channel]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            # 如果有標準化層就進行標準化
            x = self.norm(x)
        # 回傳tensor以及經過patch後的高寬(用來到時可以還原成2d的圖像)
        return x, out_size


class PatchMerging(BaseModule):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map. Our implementation uses `nn.Unfold` to
    merge patch, which is about 25% faster than original implementation.
    Instead, we need to modify pretrained models for compatibility.

    Args:
        in_channels (int): The num of input channels.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adap_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size
        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility

        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size
