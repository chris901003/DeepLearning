# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import (Linear, build_activation_layer, build_conv_layer,
                      build_norm_layer)
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)
from .drop import build_dropout
from .registry import (ATTENTION, FEEDFORWARD_NETWORK, POSITIONAL_ENCODING,
                       TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE)

# Avoid BC-breaking of importing MultiScaleDeformableAttention from this file
try:
    from mmcv.ops.multi_scale_deform_attn import \
        MultiScaleDeformableAttention  # noqa F401
    warnings.warn(
        ImportWarning(
            '``MultiScaleDeformableAttention`` has been moved to '
            '``mmcv.ops.multi_scale_deform_attn``, please change original path '  # noqa E501
            '``from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention`` '  # noqa E501
            'to ``from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention`` '  # noqa E501
        ))

except ImportError:
    warnings.warn('Fail to import ``MultiScaleDeformableAttention`` from '
                  '``mmcv.ops.multi_scale_deform_attn``, '
                  'You should install ``mmcv-full`` if you need this module. ')


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    # 已看過，構建位置編碼實例對象
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    # 已看過，構建自注意力實例對象
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    # 已看過，構建FFN實例對象
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)


def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer layer."""
    # 已看過，構建自注意力層
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    # 已看過，構建detr的encoder與decoder部分
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)


class AdaptivePadding(nn.Module):
    """Applies padding adaptively to the input.

    This module can make input get fully covered by filter
    you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad
    zero around input. The "corner"  mode would pad zero
    to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel. Default: 1.
        stride (int | tuple): Stride of the filter. Default: 1.
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

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):
        super().__init__()
        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        """Calculate the padding size of input.

        Args:
            input_shape (:obj:`torch.Size`): arrange as (H, W).

        Returns:
            Tuple[int]: The padding size along the
            original H and W directions
        """
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        """Add padding to `x`

        Args:
            x (Tensor): Input tensor has shape (B, C, H, W).

        Returns:
            Tensor: The tensor with adaptive padding
        """
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        conv_type (str): The type of convolution
            to generate patch embedding. Default: "Conv2d".
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 conv_type='Conv2d',
                 kernel_size=16,
                 stride=16,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm_cfg=None,
                 input_size=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
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

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PatchMerging(BaseModule):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map ((used in Swin Transformer)).
    Our implementation uses `nn.Unfold` to
    merge patches, which is about 25% faster than the original
    implementation. However, we need to modify pretrained
    models for compatibility.

    Args:
        in_channels (int): The num of input channels.
            to gets fully covered by filter and stride you specified.
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
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adaptive_padding = None

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

        if self.adaptive_padding:
            x = self.adaptive_padding(x)
            H, W = x.shape[-2:]

        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)
        x = self.sampler(x)

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


@ATTENTION.register_module()
class MultiheadAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.

    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        """ 已看過，構建自注意力模塊
        Args:
             embed_dims: 每一個特徵點用多少維度的向量進行表示
             num_heads: 多頭注意力機制當中用多少頭
             attn_drop: 在attn當中的dropout_rate
             proj_drop: 經過proj後的dropout_rate
             dropout_layer: dropout的設定
             init_cfg: 初始化設定
             batch_first: 在使用pytorch官方的注意力模塊可以設定batch_size是否在最前面的維度
        """
        # 初始化繼承的class
        super().__init__(init_cfg)
        if 'dropout' in kwargs:
            # dropout參數已經被淘汰了，現在透過其他三個參數進行設定
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        # 保存一些參數
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        # 構建自注意力模塊，這裡我們直接使用pytorch官方提供的自注意力模塊
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        # 構建dropout層
        self.proj_drop = nn.Dropout(proj_drop)
        # 如果沒有指定dropout層就會直接用線性代替
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.

        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.

        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.

        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """
        # 已看過，自注意力機制的forward函數
        # query = 圖像的tensor，shape [batch_size, num_queries, channel=embed_dims]
        # identity = 用來給捷徑分支的
        # query與identity的shape會相同

        if key is None:
            # 當沒有指定key時就使用query
            key = query
        if value is None:
            # 當沒有指定value時就使用key
            value = key
        if identity is None:
            # 當沒有指定identity時就使用query
            identity = query
        if key_pos is None:
            # 當我們沒有給定key的位置編碼時會進來
            if query_pos is not None:
                # 如果有給query的位置編碼就會進來
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    # 如果query的位置編碼shape與key相同，我們就將query_pos給key_pos
                    key_pos = query_pos
                else:
                    # 否則會給警告
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        # 如果有位置編碼就將位置編碼加上去
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            # 如果有指定batch_size在前的話就需要調整一下通道順序
            # [batch_size, num_queries, channel] -> [num_queries, batch_size, channel]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # 將各種資料傳入進行自注意力機制，這裡就是官方實現的attn，我們只需要第一個回傳值就可以了
        # out shape [num_queries, batch_size, channel]
        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            # 將通道順序條回來
            # [num_queries, batch_size, channel] -> [batch_size, num_queries, channel]
            out = out.transpose(0, 1)

        # 最後通過dropout以及drop_path以及捷徑分支
        # 回傳的shape與輸入的shape相同
        return identity + self.dropout_layer(self.proj_drop(out))


@FEEDFORWARD_NETWORK.register_module()
class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    @deprecated_api_warning(
        {
            'dropout': 'ffn_drop',
            'add_residual': 'add_identity'
        },
        cls_name='FFN')
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 **kwargs):
        """ 已看過，FFN層結構
        Args:
            embed_dims: 每一個特徵點會用多少維度的向量進行表示
            feedforward_channels: 在FFN中間層channel深度
            num_fcs: FFN會用多少層全連接層
            act_cfg: 激活函數的設定
            ffn_drop: FFN的dropout率
            dropout_layer: 當有捷徑分支時的dropout
            add_identity: 是否使用捷徑分支
            init_cfg: 初始化設定
        """
        super().__init__(init_cfg)
        # FFN至少需要兩層以上的全連接層
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        # 保存一些參數
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        # 構建激活函數層
        self.activate = build_activation_layer(act_cfg)

        layers = []
        # 最一開始的channel就會是embed_dims
        in_channels = embed_dims
        # 遍歷num_fcs-1次
        for _ in range(num_fcs - 1):
            layers.append(
                # 添加全連接層以及激活函數以及失活層
                Sequential(
                    Linear(in_channels, feedforward_channels), self.activate,
                    nn.Dropout(ffn_drop)))
            # 更新輸入的channel
            in_channels = feedforward_channels
        # 最後一層需要將channel變回輸入時的channel深度
        layers.append(Linear(feedforward_channels, embed_dims))
        # 最後還有失活層
        layers.append(nn.Dropout(ffn_drop))
        # 透過Sequential全部包裝起來
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    @deprecated_api_warning({'residual': 'identity'}, cls_name='FFN')
    def forward(self, x, identity=None):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        # 已看過，FFN的forward函數
        # x = 圖像的tensor，shape [batch_size, num_queries, channel]
        # identity = 用來給捷徑分支的，這裡shape與x會相同

        # 通過一系列全連結層，最終shape不會改變
        out = self.layers(x)
        if not self.add_identity:
            # 不需要捷徑分支就會走這裡
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        # 添加上捷徑分支
        return identity + self.dropout_layer(out)


@TRANSFORMER_LAYER.register_module()
class BaseTransformerLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        """ 已看過，完整自注意力模塊(包含FFN)
        Args:
            attn_cfgs: 自注意力機制的配置
            ffn_cfgs: FFN的配置
            operation_order: 通過層結構的順序
            norm_cfg: 標準化層設定
            init_cfg: 初始化設定
            batch_first: 是否將batch_size放在第一個維度
            kwargs: 一些其他資訊
        """

        # 已棄用的參數
        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        # 遍歷已棄用的參數
        for ori_name, new_name in deprecated_args.items():
            # 遍歷kwargs裏面的值
            if ori_name in kwargs:
                # 如果有配對上的就會將舊的key值換成新的key值
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                # 並且將結果放到ffn_cfgs當中
                ffn_cfgs[new_name] = kwargs[ori_name]

        # 繼承於BaseModule，初始化繼承對象
        super().__init__(init_cfg)

        # 保存是否batch_size在第一個維度
        self.batch_first = batch_first

        # 檢查operation_order有沒有問題
        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        # 計算有多少個自注意力模塊
        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            # 將自注意力設定拷貝使用次數
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            # 檢查長度是否合法
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        # 保存傳入的參數
        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        # 是否先進行標準化
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        # 遍歷通過層結構的順序
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                # 如果是自注意力模塊
                if 'batch_first' in attn_cfgs[index]:
                    # 如果有設定batch_first就檢查是否與傳入的batch_first相同值
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    # 如果裡面沒有就添加上去
                    attn_cfgs[index]['batch_first'] = self.batch_first
                # 構建自注意力模塊實例對象
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                # 獲取名稱
                attention.operation_name = operation_name
                # 將實例化對象保存下來
                self.attentions.append(attention)
                index += 1

        # 獲取一個特徵點要用多少維度的向量進行表示
        self.embed_dims = self.attentions[0].embed_dims

        # FFN結構保存
        self.ffns = ModuleList()
        # 計算會通過多少個FFN
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            # 如果是dict格式就會包裝成ConfigDict格式
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            # 將config重複num_ffns次
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        # 檢查長度需要相等
        assert len(ffn_cfgs) == num_ffns
        # 遍歷FFN的堆疊次數
        for ffn_index in range(num_ffns):
            # 如果沒有設定embed_dims就會添加上去，否則就會檢查是否一致
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            # 構建FFN實例對象
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        # 構建標準化層
        self.norms = ModuleList()
        # 檢查需要多少個標準化層
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            # 根據需要的數量進行構建實例化對象
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.
        這裡的batch_size可以是在第一個維度會是在第二個維度，只需要調整一下或是設定batch first就可以

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
                自注意力當中的query，shape [batch_size, num_queries, channel=embed_dim]
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                自注意力當中的query，shape [batch_size, num_key, channel=embed_dim]
            value (Tensor): The value tensor with same shape as `key`.
                這裡會跟key的shape相同
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
                就是query的位置編碼，shape [batch_size, num_queries, channel=embed_dim]
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
                就是key的位置編碼，shape [batch_size, num_key, channel=embed_dim]
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
                如果是padding的部分就會是True其他部分就會是False
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        # 已看過，自注意力模塊的forward函數

        # 將一些東西設定成0，用來看要用第幾個標準化層或是自注意力層或是FFN
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        # 保留query之後用來做殘差邊的
        identity = query
        if attn_masks is None:
            # 如果傳入的attn_masks是None，就創建一個list長度為自注意力層數且都是None
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            # 否則就將attn_masks拷貝自注意力的層數並放到list當中
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            # 這裡會告知堆疊多層的attn都是用一樣的mask
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            # 其他就會到這裡來，如果想要堆疊多層的attn中每層的mask不同就可以給與堆疊層數相同數量的attn_mask
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        # 根據指定的通過層結構的順序進行向前傳遞
        for layer in self.operation_order:
            if layer == 'self_attn':
                # 如果是自注意力會到這裡
                temp_key = temp_value = query
                # 進行自注意力
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                # 標準化層
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                # 交叉注意力層
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TransformerLayerSequence(BaseModule):
    """Base class for TransformerEncoder and TransformerDecoder in vision
    transformer.

    As base-class of Encoder and Decoder in vision transformer.
    Support customization such as specifying different kind
    of `transformer_layer` in `transformer_coder`.

    Args:
        transformerlayer (list[obj:`mmcv.ConfigDict`] |
            obj:`mmcv.ConfigDict`): Config of transformerlayer
            in TransformerCoder. If it is obj:`mmcv.ConfigDict`,
             it would be repeated `num_layer` times to a
             list[`mmcv.ConfigDict`]. Default: None.
        num_layers (int): The number of `TransformerLayer`. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self, transformerlayers=None, num_layers=None, init_cfg=None):
        """ 已看過，transformer的encoder與decoder的基底
        Args:
            transformerlayers: transformer的設定
            num_layers: 堆疊層數
            init_cfg: 初始化設定方式
        """

        # 繼承於BaseModule，對繼承對象進行初始化
        super().__init__(init_cfg)
        if isinstance(transformerlayers, dict):
            # transformerlayers傳入的dict格式會進入到這裡
            # 將傳入的transformerlayers拷貝num_layers份並且保存下來
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            # 其他就要檢查長度是否符合堆疊層數
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        # 構建layers保存空間
        self.layers = ModuleList()
        # 遍歷總共需要堆疊多少層
        for i in range(num_layers):
            # 構建層結構實例對象
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        # 保存一個特徵點會用多少維度向量進行表示
        self.embed_dims = self.layers[0].embed_dims
        # 是否會先進行標準化
        self.pre_norm = self.layers[0].pre_norm

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerCoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_queries, bs, embed_dims)`.
            key (Tensor): The key tensor with shape
                `(num_keys, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_keys, bs, embed_dims)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor], optional): Each element is 2D Tensor
                which is used in calculation of corresponding attention in
                operation_order. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in self-attention
                Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor:  results with shape [num_queries, bs, embed_dims].
        """
        # 已看過，進行多層自注意力層的堆疊
        # query_key_padding_mask = 哪些部分是透過padding出來的，padding的部分會是True其他地方會是False
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        return query
