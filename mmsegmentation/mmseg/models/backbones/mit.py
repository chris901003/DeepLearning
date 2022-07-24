# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcv.runner import BaseModule, ModuleList, Sequential

from ..builder import BACKBONES
from ..utils import PatchEmbed, nchw_to_nlc, nlc_to_nchw


class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        """ 已看過，Segformer的FFN結構
        Args:
            embed_dims: 一個特徵點會用多少維度的向量進行表示
            feedforward_channels: FFN中間層的channel深度
            act_cfg: 激活函數的選擇
            ffn_drop: FFN當中的dropout率
            dropout_layer: 使用哪種dropout方法
            init_cfg: 初始化方式
        """

        # 對繼承對象進行初始化
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        # 構建激活函數的實例對象
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        # 這裡使用conv進行FFN而不是像傳統使用fc進行
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 最後還會有dropout層
        drop = nn.Dropout(ffn_drop)
        # 全部放入到self.layers當中且用Sequential進行包裝
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        # 如果沒有設定dropout就用一個顯性層代替
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        """ 已看過，Segformer的FFN結構
        Args:
            x: 圖像1d的特徵圖，shape [batch_size, tot_patch, channel]
            hw_shape: 特徵圖2d的高寬
            identity: 給捷徑分支使用的
        """

        # 調整通道，[batch_size, tot_patch, channel] -> [batch_size, channel, height, width]
        out = nlc_to_nchw(x, hw_shape)
        # 通過FFN層結構，這裡都是卷積層
        out = self.layers(out)
        # 調整通道，[batch_size, channel, height, width] -> [batch_size, tot_patch, channel]
        out = nchw_to_nlc(out)
        if identity is None:
            # 如果沒有傳入identity就直接是原始x
            identity = x
        # 將FFN結果加上捷徑分支
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    """An implementation of Efficient Multi-head Attention of Segformer.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=None,
                 init_cfg=None,
                 batch_first=True,
                 qkv_bias=False,
                 norm_cfg=dict(type='LN'),
                 sr_ratio=1):
        """ 已看過，這是為了Segformer做出的efficient multi-head transformer encoder
        Args:
            embed_dims: 一個特徵點要用多少維度的向量進行表示
            num_heads: 多頭注意力機制當中的頭數
            attn_drop: 在attn當中的dropout率
            proj_drop: 最後融合後的dropout率
            dropout_layer: 使用的dropout_layer模塊，這裡是DropPath
            init_cfg: 初始化方式
            batch_first: batch_size維度是否放在最前面
            qkv_bias: qkv的偏置是否開啟
            norm_cfg: 標準化層的選擇
            sr_ratio: 在Efficient attn當中的空間壓縮量，會透過一個conv進行壓縮，kernel_size=stride=sr_ratio
        """

        # 這裡繼承於MultiheadAttention，這裡將一些參數帶入進去
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias)

        # 保存sr_ratio數值
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # 如果sr_ratio大於一我們就使用conv進行壓縮，kernel_size=stride=sr_ratio
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            # 這裡會經過一個表準化層
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # handle the BC-breaking from https://github.com/open-mmlab/mmcv/pull/1418 # noqa
        from mmseg import digit_version, mmcv_version
        if mmcv_version < digit_version('1.3.17'):
            warnings.warn('The legacy version of forward function in'
                          'EfficientMultiheadAttention is deprecated in'
                          'mmcv>=1.3.17 and will no longer support in the'
                          'future. Please upgrade your mmcv.')
            # 這裡會根據mmcv的版本選擇forward函數，我們這裡會使用原始的forward函數
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None):
        """ 已看過，特別為Segformer設計的自注意力機制
        Args:
            x: 圖像的tensor，shape [batch_size, tot_path, channel]
            hw_shape: 特徵圖2d的高寬
            identity: 捷徑分支使用的
        """

        x_q = x
        if self.sr_ratio > 1:
            # 如果sr_ratio大於一這裡就會需要先通過sr層結構
            # 透過nlc_to_nchw將通道進行調整 [batch_size, tot_patch, channel] -> [batch_size, channel, height, width]
            x_kv = nlc_to_nchw(x, hw_shape)
            # 將特徵圖通過一個卷積層
            x_kv = self.sr(x_kv)
            # 再將通道進行調整，[batch_size, channel, height, width] -> [batch_size, tot_patch, channel]
            x_kv = nchw_to_nlc(x_kv)
            # 通過一層標準化層
            x_kv = self.norm(x_kv)
        else:
            # 如果沒有需要sr_ratio就直接給值
            x_kv = x

        if identity is None:
            # 如果沒有傳入identity這裡就使用傳入的x
            identity = x_q

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            # 如果batch_size是在第一個維度，這裡就會需要調整一下
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)

        # 將qkv傳入到自注意力模塊當中，這裡我們只需要第一個返回值就可以
        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            # 將通道順序調整回來
            out = out.transpose(0, 1)

        # 將結果經過DropPath後在標準化最後加上捷徑分支
        return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None):
        """multi head attention forward in mmcv version < 1.3.17."""

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # `need_weights=True` will let nn.MultiHeadAttention
        # `return attn_output, attn_output_weights.sum(dim=1) / num_heads`
        # The `attn_output_weights.sum(dim=1)` may cause cuda error. So, we set
        # `need_weights=False` to ignore `attn_output_weights.sum(dim=1)`.
        # This issue - `https://github.com/pytorch/pytorch/issues/37583` report
        # the error that large scale tensor sum operation may cause cuda error.
        out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False)[0]

        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 sr_ratio=1,
                 with_cp=False):
        """ 已看過，這裡就是MixViT的自注意力部分
        Args:
            embed_dims: 每一個特徵點要用多少維度的向量進行表示
            num_heads: 多頭注意力機制當中要用多少頭
            feedforward_channels: FFN中間全連接層channel的深度
            drop_rate: dropout率
            attn_drop_rate: 在attn當中dropout率
            drop_path_rate: drop_path的機率
            qkv_bias: 是否啟用qkv的偏置
            act_cfg: 激活函數的設置
            norm_cfg: 標準化層的設定
            batch_first: 在進行forward時傳入的資料batch_size是否放在第一個維度
            sr_ratio: 在Efficient attn當中的空間壓縮量，會透過一個conv進行壓縮，kernel_size=stride=sr_ratio
            with_cp: 是否用checkpoint
        """

        # 對繼承對象進行初始化
        super(TransformerEncoderLayer, self).__init__()

        # The ret[0] of build_norm_layer is norm name.
        # 構建標準化層，這裡我們只取出實例對象
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        # 實例化EfficientMultiheadAttention模塊
        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        # 會有經過一層標準化層，這裡我們也只需要實例化對象即可
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        self.with_cp = with_cp

    def forward(self, x, hw_shape):
        """ 已看過，自注意力機制模塊
        Args:
            x = 圖像的tensor，shape [batch_size, tot_patch, channel]
            hw_shape = 圖像2d的高寬
        """

        def _inner_forward(x):
            x = self.attn(self.norm1(x), hw_shape, identity=x)
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            # x shape = [batch_size, tot_patch, channel]
            x = _inner_forward(x)
        return x


@BACKBONES.register_module()
class MixVisionTransformer(BaseModule):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False):
        """ 已看過，這裡是ViT的變形版本，MixVisionTransformer
        Args:
            in_channels: 輸入的channel深度
            embed_dims: 每一個特徵點用多少維度的向量進行表示
            num_stages: 結構的層數
            num_layers: 每個transformer encoder要堆疊的數量
            num_heads: 每一個transformer encoder模塊當中多頭注意力的頭數
            patch_sizes: 每一個transformer encoder的patch大小
            strides: patch embedding的步距
            sr_ratios:
            out_indices: 哪些層結構的輸出要放到最終結果，這裡預設全部都會進行輸出
            mlp_ratio: 在FFN的中層channel會加深多少倍
            qkv_bias: qkv是否開啟偏置
            drop_rate: dropout率
            attn_drop_rate: 在attn當中的dropout率
            drop_path_rate: 在drop_path當中的dropout率
            act_cfg: 激活函數的選擇
            norm_cfg: 標準化層的選擇
            pretrained: 預訓練權重資料
            init_cfg: 初始化設定資料
            with_cp: 是否有需要checkpoint
        """

        # 將繼承的class進行初始化
        super(MixVisionTransformer, self).__init__(init_cfg=init_cfg)

        # init_cfg與pretrained只能選一個對模型進行初始化
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            # 如果傳入的pretrained是str格式就會跳出警告，這裡希望我們寫到init_cfg當中
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            # 其他非str格式的pretrained就會直接報錯
            raise TypeError('pretrained must be a str or None')

        # 保存一些傳入的參數
        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        # 檢查一些參數需要對應上
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        # 保存一些參數
        self.out_indices = out_indices
        # 輸出的層數量不能大於總層數
        assert max(out_indices) < self.num_stages

        # transformer encoder，dropout率是依照層的深度進行線性調整
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        # cur計算總層數
        cur = 0
        # 使用ModuleList將層結構記錄下來
        self.layers = ModuleList()
        # 遍歷所有層結構
        for i, num_layer in enumerate(num_layers):
            # num_layer = 當前層需要堆疊多少個encoder
            # 透過當前需要的頭數會先對當前的channel深度進行調整，embed_dims_i就會是當前的channel深度
            embed_dims_i = embed_dims * num_heads[i]
            # 將輸入透過PatchEmbed進行patch同時也會調整channel深度
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                # 這裡的patch_size與strides不相同，所以在進行patch時會有overlap的情況
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)
            # 這裡就是自注意力模塊部分，會進行堆疊num_layer次，使用的是TransformerEncoderLayer類
            layer = ModuleList([
                TransformerEncoderLayer(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            # 更新下層的輸入channel深度為本層的輸出channel深度
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            # 激活函數層
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            # 計算總層數使用的
            cur += num_layer

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(MixVisionTransformer, self).init_weights()

    def forward(self, x):
        """ 已看過，Segformer的第一個forward函數調用的地方
        Args:
            x = 圖像的tensor格式，shape [batch_size, channel=3, height, width]
        """

        # 最後回傳的資料
        outs = []

        # 遍歷各模塊
        for i, layer in enumerate(self.layers):
            # 先經過一個padding層，這裡是用conv進行實現
            # x shape = [batch_size, height * width, channel]
            # hw_shape = 記錄下高寬，這樣可以將特徵圖還原成2d的型態
            x, hw_shape = layer[0](x)
            # 經過一系列的自注意力層
            for block in layer[1]:
                x = block(x, hw_shape)
            # 最後是標準化層
            x = layer[2](x)
            # 調整通道，[batch_size, tot_patch, channel] -> [batch_size, channel, height, width]
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                # 如果是需要的輸出層就會保存下來
                outs.append(x)

        return outs
