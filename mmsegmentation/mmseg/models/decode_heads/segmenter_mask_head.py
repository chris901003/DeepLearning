# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmcv.runner import ModuleList

from mmseg.models.backbones.vit import TransformerEncoderLayer
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class SegmenterMaskTransformerHead(BaseDecodeHead):
    """Segmenter: Transformer for Semantic Segmentation.

    This head is the implementation of
    `Segmenter:　<https://arxiv.org/abs/2105.05633>`_.

    Args:
        backbone_cfg:(dict): Config of backbone of
            Context Path.
        in_channels (int): The number of channels of input image.
        num_layers (int): The depth of transformer.
        num_heads (int): The number of attention heads.
        embed_dims (int): The number of embedding dimension.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        drop_path_rate (float): stochastic depth rate. Default 0.1.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        init_std (float): The value of std in weight initialization.
            Default: 0.02.
    """

    def __init__(
            self,
            in_channels,
            num_layers,
            num_heads,
            embed_dims,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            num_fcs=2,
            qkv_bias=True,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            init_std=0.02,
            **kwargs,
    ):
        """ 已看過，這裡是對segmenter的解碼頭，這裡會接收來自ViT後的輸出，進行所謂的mask transformer
        Args:
            in_channels: 輸入進來的channel深度
            num_layers: 堆疊的深度
            num_heads: 多頭注意力當中的頭數
            embed_dims: 一個特徵點會用多少維度的向量進行表示
            mlp_ratio: 在FFN的中間層channel深度
            drop_path_rate: drop_path的概率
            drop_rate: 失活概率
            attn_drop_rate: attn當中的失活概率
            num_fcs: FFN當中堆疊多少層的全連接層
            qkv_bias: 是否啟用qkv偏置
            act_cfg: 激活函數相關配置
            norm_cfg: 標準化層的配置
            init_std:
        """
        # 初始化繼承對象，裏面包含最後通道數的調整以及損失函數構建
        super(SegmenterMaskTransformerHead, self).__init__(
            in_channels=in_channels, **kwargs)

        # 失活概率會從小到大，較淺層部分會用較小的失活率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        # 使用ModuleList進行保存層結構
        self.layers = ModuleList()
        # 遍歷需要構建的層數
        for i in range(num_layers):
            self.layers.append(
                # 這裡使用的是transformer encoder層
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True,
                ))

        # 這裡還會有一個全連接層
        self.dec_proj = nn.Linear(in_channels, embed_dims)

        # 這個是用來給decoder的原始輸入部分，先隨機生成一個可訓練的tensor且shape [1, num_classes, embed_dims]
        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_classes, embed_dims))
        # 這裡還有幾個全連接層，可以到forward當中看是在哪個部分使用到
        self.patch_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.classes_proj = nn.Linear(embed_dims, embed_dims, bias=False)

        # 構建標準化層
        self.decoder_norm = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)[1]
        self.mask_norm = build_norm_layer(
            norm_cfg, self.num_classes, postfix=2)[1]

        # 保存初始化標準值
        self.init_std = init_std

        # 將繼承的conv_seg卷積層移除，原先conv_seg是用來在最後時將通道調整到與num_classes相同
        delattr(self, 'conv_seg')

    def init_weights(self):
        trunc_normal_(self.cls_emb, std=self.init_std)
        trunc_normal_init(self.patch_proj, std=self.init_std)
        trunc_normal_init(self.classes_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, inputs):
        # 已看過，SegmenterMaskTransformerHead的forward函數，decoder的開始
        # inputs = tuple(Tensor)，shape [batch_size, channel, height, width]

        # 透過_transform_inputs將我們需要的特徵圖提取出來
        # x shape = [batch_size, channel, height, width]
        x = self._transform_inputs(inputs)
        # 獲取維度參數
        b, c, h, w = x.shape
        # [batch_size, channel, height, width] -> [batch_size, height, width, channel]
        # -> [batch_size, height * width, channel]
        x = x.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        # 通過一個全連接層，shape不產生變化
        x = self.dec_proj(x)
        # cls_emb shape = [1, num_classes, channel=embed_dim] -> [batch_size, num_classes, channel=embed_dim]
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        # 將cls_emb與x在第一維度進行拼接，x shape [batch_size, tot_patch + num_classes, channel=embed_dim]
        x = torch.cat((x, cls_emb), 1)
        # 遍歷層結構，x的shape不會產生變化
        for layer in self.layers:
            # 這裡就是自注意力機制的forward函數呼叫
            x = layer(x)
        # 通過一個標準化層
        x = self.decoder_norm(x)

        # 透過[:, :-self.num_classes]將非num_classes部分的tensor取出來之後再經過patch_proj(全連接層)
        # patches shape = [batch_size, tot_patch, channel=embed_dim]
        patches = self.patch_proj(x[:, :-self.num_classes])
        # 透過[:, -self.num_classes:]將num_classes部分的tensor取出來之後再經過class_proj(全連接層)
        # cls_seg_feat shape = [batch_size, num_classes, channel=embed_dim]
        cls_seg_feat = self.classes_proj(x[:, -self.num_classes:])

        # 對patches以及cls_seg_feat在channel維度上面進行標準化
        patches = F.normalize(patches, dim=2, p=2)
        cls_seg_feat = F.normalize(cls_seg_feat, dim=2, p=2)

        # 先調整cls_seg_feat的通道 [batch_size, channel, num_classes]，之後在與patches進行矩陣乘法
        # [batch_size, tot_patch, channel] @ [batch_size, channel, num_classes]
        # mask shape = [batch_size, tot_patch, num_classes]
        masks = patches @ cls_seg_feat.transpose(1, 2)
        # 將masks行標準化
        masks = self.mask_norm(masks)
        # [batch_size, tot_patch, num_classes] -> [batch_size, num_classes, tot_path]
        # -> [batch_size, num_classes, height, width]
        # mask成功變回2d特徵圖並且channel=num_classes
        masks = masks.permute(0, 2, 1).contiguous().view(b, -1, h, w)

        # 最後回傳shape [batch_size, channel=num_classes, height, width]
        return masks
