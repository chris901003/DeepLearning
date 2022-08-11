# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule

from mmocr.models.common.modules import (MultiHeadAttention,
                                         PositionwiseFeedForward)


class TFEncoderLayer(BaseModule):
    """Transformer Encoder Layer.

    Args:
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_inner (int): The dimension of the feedforward
            network model (default=256).
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
        act_cfg (dict): Activation cfg for feedforward module.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm')
            or ('norm', 'self_attn', 'norm', 'ffn').
            Default：None.
    """

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 act_cfg=dict(type='mmcv.GELU'),
                 operation_order=None):
        super().__init__()
        self.attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_cfg=act_cfg)
        self.norm2 = nn.LayerNorm(d_model)

        self.operation_order = operation_order
        if self.operation_order is None:
            self.operation_order = ('norm', 'self_attn', 'norm', 'ffn')

        assert self.operation_order in [('norm', 'self_attn', 'norm', 'ffn'),
                                        ('self_attn', 'norm', 'ffn', 'norm')]

    def forward(self, x, mask=None):
        if self.operation_order == ('self_attn', 'norm', 'ffn', 'norm'):
            residual = x
            x = residual + self.attn(x, x, x, mask)
            x = self.norm1(x)

            residual = x
            x = residual + self.mlp(x)
            x = self.norm2(x)
        elif self.operation_order == ('norm', 'self_attn', 'norm', 'ffn'):
            residual = x
            x = self.norm1(x)
            x = residual + self.attn(x, x, x, mask)

            residual = x
            x = self.norm2(x)
            x = residual + self.mlp(x)

        return x


class TFDecoderLayer(nn.Module):
    """Transformer Decoder Layer.

    Args:
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_inner (int): The dimension of the feedforward
            network model (default=256).
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
        act_cfg (dict): Activation cfg for feedforward module.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'enc_dec_attn',
            'norm', 'ffn', 'norm') or ('norm', 'self_attn', 'norm',
            'enc_dec_attn', 'norm', 'ffn').
            Default：None.
    """

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 act_cfg=dict(type='mmcv.GELU'),
                 operation_order=None):
        """ 已看過，transformer decoder部分
        Args:
            d_model: 輸入的channel深度
            d_inner: 在FFN中間層的channel深度
            n_head: 多頭注意力機制當中的頭數
            d_k: 每一個頭當中key的channel深度
            d_v: 每一個頭當中value的channel深度
            dropout: dropout概率
            qkv_bias: 是否啟用qkv的偏置
            act_cfg: 激活函數的設定
            operation_order: 層結構的順序
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()

        # 構建三層LN表準化層結構
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # 構建自注意力層結構
        self.self_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)

        # 這裡是encoder與decoder混合注意力結構
        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)

        # FFN層結構部分
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_cfg=act_cfg)

        self.operation_order = operation_order
        if self.operation_order is None:
            # 如果沒有特別設定層結構的順序就會到這裡，使用默認的順序
            self.operation_order = ('norm', 'self_attn', 'norm',
                                    'enc_dec_attn', 'norm', 'ffn')
        # 會檢查順序是否合法
        assert self.operation_order in [
            ('norm', 'self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn'),
            ('self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn', 'norm')
        ]

    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                dec_enc_attn_mask=None):
        """ 已看過，NLP當中transformer decoder的forward函數
        Args:
            dec_input: decoder的輸入，tensor shape = [batch_size, seq_len, channel]
            enc_output: encoder的輸出，這裡會作為decoder的輸入，tensor shape = [batch_size, height * width, channel]
            self_attn_mask: 屬於dec_input的mask資訊，也可以說是自注意時的mask，shape = [batch_size, seq_len, seq_len]
            dec_enc_attn_mask: 屬於enc_output的mask資訊，也可以說是交叉注意力時的mask，shape = [batch_size, height * width]
        """
        # 有兩種不同的層結構順序
        if self.operation_order == ('self_attn', 'norm', 'enc_dec_attn',
                                    'norm', 'ffn', 'norm'):
            dec_attn_out = self.self_attn(dec_input, dec_input, dec_input,
                                          self_attn_mask)
            dec_attn_out += dec_input
            dec_attn_out = self.norm1(dec_attn_out)

            enc_dec_attn_out = self.enc_attn(dec_attn_out, enc_output,
                                             enc_output, dec_enc_attn_mask)
            enc_dec_attn_out += dec_attn_out
            enc_dec_attn_out = self.norm2(enc_dec_attn_out)

            mlp_out = self.mlp(enc_dec_attn_out)
            mlp_out += enc_dec_attn_out
            mlp_out = self.norm3(mlp_out)
        elif self.operation_order == ('norm', 'self_attn', 'norm',
                                      'enc_dec_attn', 'norm', 'ffn'):
            # 將dec_input通過LN標準化層
            dec_input_norm = self.norm1(dec_input)
            # 進行dec_input的自注意，通過自注意模塊進行並且將mask傳入，這裡shape不會產生變化
            dec_attn_out = self.self_attn(dec_input_norm, dec_input_norm,
                                          dec_input_norm, self_attn_mask)
            # 這裡會有一個殘差邊
            dec_attn_out += dec_input

            # 通過LN標準化層
            enc_dec_attn_in = self.norm2(dec_attn_out)
            # 通過交叉注意力層結構，這裡需要將encoder的mask傳入
            # 這裡將decoder的自注意力結果當作query，encoder的輸出作為key以及value
            enc_dec_attn_out = self.enc_attn(enc_dec_attn_in, enc_output,
                                             enc_output, dec_enc_attn_mask)
            # 這裡會有一個殘差邊
            enc_dec_attn_out += dec_attn_out

            # 將結果通過LN標準化後再通過FFN層
            mlp_out = self.mlp(self.norm3(enc_dec_attn_out))
            # 再通過一個殘差邊
            mlp_out += enc_dec_attn_out

        # 最後回傳
        return mlp_out
