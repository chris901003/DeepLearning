# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch.nn as nn
from mmcv.runner import ModuleList

from mmocr.models.builder import ENCODERS
from mmocr.models.textrecog.layers import (Adaptive2DPositionalEncoding,
                                           SatrnEncoderLayer)
from .base_encoder import BaseEncoder


@ENCODERS.register_module()
class SatrnEncoder(BaseEncoder):
    """Implement encoder for SATRN, see `SATRN.

    <https://arxiv.org/abs/1910.04396>`_.

    Args:
        n_layers (int): Number of attention layers.
        n_head (int): Number of parallel attention heads.
        d_k (int): Dimension of the key vector.
        d_v (int): Dimension of the value vector.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
        n_position (int): Length of the positional encoding vector. Must be
            greater than ``max_seq_len``.
        d_inner (int): Hidden dimension of feedforward layers.
        dropout (float): Dropout rate.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 n_layers=12,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 n_position=100,
                 d_inner=256,
                 dropout=0.1,
                 init_cfg=None,
                 **kwargs):
        """ 已看過，SATRN的encoder的初始化部分，這裡會用transformer的encoder的基底進行針對性改進
        Args:
            n_layers: 堆疊的層數
            n_head: 多頭注意力機制的頭數
            d_k: 每個注意裡頭中key的深度
            d_v: 每個注意裡頭中value的深度
            d_model: 傳入的channel深度
            n_position: 位置編碼的長度，必須要比最大序列長度要長
            d_inner: FFN中間層的channel深度
            dropout: dropout概率
            init_cfg: 初始化設定
        """
        # 繼承自BaseEncoder，對繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)
        # 保存傳入的channel深度
        self.d_model = d_model
        # 構建自適應2D位置編碼實例對象
        self.position_enc = Adaptive2DPositionalEncoding(
            d_hid=d_model,
            n_height=n_position,
            n_width=n_position,
            dropout=dropout)
        # 構建多層自注意力機制的層結構，這裡會使用SatrnEncoderLayer類
        self.layer_stack = ModuleList([
            SatrnEncoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            # 這裡會進行重複堆疊
            for _ in range(n_layers)
        ])
        # 最後使用LN進行標準化
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, feat, img_metas=None):
        """
        Args:
            feat (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            Tensor: A tensor of shape :math:`(N, T, D_m)`.
        """
        # 已看過，SATRN的encoder的forward部分
        # feat = 從backbone出來的特徵圖，tensor shape [batch_size, channel=512, height, width]
        # img_metas = 圖像的詳細資料

        # 構建valid_ratios，這裡會都是1的list，且list長度會是batch_size
        valid_ratios = [1.0 for _ in range(feat.size(0))]
        if img_metas is not None:
            # 如果有傳入img_metas就會以img_metas當中的valid_ratios為主
            valid_ratios = [
                img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
            ]
        # 將傳入的特徵圖添加上位置編碼，這裡shape不會有改變
        feat += self.position_enc(feat)
        # 獲取特徵圖的shape資訊
        n, c, h, w = feat.size()
        # 獲取全為0的mask，這裡的shape會是[batch_size, height, width]
        mask = feat.new_zeros((n, h, w))
        # 遍歷整個valid_ratios，也就是batch_size
        for i, valid_ratio in enumerate(valid_ratios):
            # 獲取合法的寬度，這裡會是當前的w*valid_ratio與w取小，也就是當valid_ratio大於1就會是w，當valid_ratio小於1就會是另一個
            valid_width = min(w, math.ceil(w * valid_ratio))
            # 將合法的範圍內變成1，其他部分就會是0
            mask[i, :, :valid_width] = 1
        # 因為接下來要進入到transformer結構當中，所以這裡需要將高寬進行flatten
        # mask shape = [batch_size, height, width] -> [batch_size, height * width]
        mask = mask.view(n, h * w)
        # feat shape = [batch_size, channel=512, height, width] -> [batch_size, channel=512, height * width]
        feat = feat.view(n, c, h * w)

        # 調整特徵圖的通道排列順序 [batch_size, channel=512, height * width] -> [batch_size, height * width, channel=512]
        output = feat.permute(0, 2, 1).contiguous()
        # 遍歷多層的attn層結構
        for enc_layer in self.layer_stack:
            # 將特徵圖與原始高寬以及mask傳入進行正向傳播
            output = enc_layer(output, h, w, mask)
        # 最後通過LN標準化層結構
        output = self.layer_norm(output)

        # 最終進行輸出，shape = [batch_size, height * width = seq_len, channel]
        return output
