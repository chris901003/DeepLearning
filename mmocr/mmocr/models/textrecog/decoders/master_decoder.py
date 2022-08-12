# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.runner import ModuleList

from mmocr.models.builder import DECODERS
from mmocr.models.common.modules import PositionalEncoding
from .base_decoder import BaseDecoder


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        """ 已看過，構建文字的embedding，將文字對應的index轉成一個向量
        Args:
            d_model: 轉換成向量的向量維度
            vocab: 總共有多少種文字
        """
        # 繼承自nn.Module，對繼承對象進行初始化
        super(Embeddings, self).__init__()
        # 透過nn.Embedding構建
        self.lut = nn.Embedding(vocab, d_model)
        # 保存向量的維度
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


@DECODERS.register_module()
class MasterDecoder(BaseDecoder):
    """Decoder module in `MASTER <https://arxiv.org/abs/1910.02562>`_.

    Code is partially modified from https://github.com/wenwenyu/MASTER-pytorch.

    Args:
        start_idx (int): The index of `<SOS>`.
        padding_idx (int): The index of `<PAD>`.
        num_classes (int): Number of text characters :math:`C`.
        n_layers (int): Number of attention layers.
        n_head (int): Number of parallel attention heads.
        d_model (int): Dimension :math:`E` of the input from previous model.
        feat_size (int): The size of the input feature from previous model,
            usually :math:`H * W`.
        d_inner (int): Hidden dimension of feedforward layers.
        attn_drop (float): Dropout rate of the attention layer.
        ffn_drop (float): Dropout rate of the feedforward layer.
        feat_pe_drop (float): Dropout rate of the feature positional encoding
            layer.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
        self,
        start_idx,
        padding_idx,
        num_classes=93,
        n_layers=3,
        n_head=8,
        d_model=512,
        feat_size=6 * 40,
        d_inner=2048,
        attn_drop=0.,
        ffn_drop=0.,
        feat_pe_drop=0.2,
        max_seq_len=30,
        init_cfg=None,
    ):
        """ 已看過，MASTER的decoder部分
        Args:
            start_idx: 標籤為<SOS>的index值
            padding_idx: 標前為<PAD>的index值
            num_classes: 總分類數
            n_layers: 注意力機制的層數
            n_head: 多頭注意力機制當中的頭數
            d_model: 輸入的channel深度
            feat_size: 輸入的特徵圖高寬相乘後的大小，這裡因為是transformer結構，所以需要將高寬壓平
            d_inner: FFN中間層的channel深度
            attn_drop: 在注意力模塊當中的dropout概率
            ffn_drop: 在FFN當中的dropout概率
            feat_pe_drop: 在位置編碼部分的dropout概率
            max_seq_len: 最長序列長度
            init_cfg: 初始化方式
        """
        # 繼承自BaseDecoder，將繼承對象進行初始化
        super(MasterDecoder, self).__init__(init_cfg=init_cfg)

        # 構建通過層結構的順序
        operation_order = ('norm', 'self_attn', 'norm', 'cross_attn', 'norm',
                           'ffn')
        # 構建decoder_layer，這裡會實例化BaseTransformerLayer
        decoder_layer = BaseTransformerLayer(
            # 將通過層結構的順序傳入
            operation_order=operation_order,
            # 構建attn部分的config資料
            attn_cfgs=dict(
                # 使用多頭注意力機制模塊
                type='MultiheadAttention',
                # 輸入的channel深度
                embed_dims=d_model,
                # 多頭注意力當中的頭數
                num_heads=n_head,
                # 在attn當中的dropout概率
                attn_drop=attn_drop,
                # 使用的dropout設定資料
                dropout_layer=dict(type='Dropout', drop_prob=attn_drop),
            ),
            # 構建FFN的config資料
            ffn_cfgs=dict(
                # 使用FFN模塊
                type='FFN',
                # 輸入的channel深度
                embed_dims=d_model,
                # 中間層的channel深度
                feedforward_channels=d_inner,
                # 在FFN當中的dropout概率
                ffn_drop=ffn_drop,
                # 使用的dropout設定資料
                dropout_layer=dict(type='Dropout', drop_prob=ffn_drop),
            ),
            # 標準化層使用的是LN標準化
            norm_cfg=dict(type='LN'),
            # forward時輸入的tensor會將batch_size放在第0維度
            batch_first=True,
        )
        # 將decoder_layer總共重複n_layers，也就是會堆疊n_layers層
        self.decoder_layers = ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(n_layers)])

        # 最後通過一層全連接層將channel調整到num_classes
        self.cls = nn.Linear(d_model, num_classes)

        # 保存SOS對應到的index
        self.SOS = start_idx
        # 保存PAD對應到的index
        self.PAD = padding_idx
        # 保存最大序列長度
        self.max_seq_len = max_seq_len
        # 保存輸入特徵圖的高寬乘積
        self.feat_size = feat_size
        # 保存多頭的頭數
        self.n_head = n_head

        # 構建每個類別對應的向量，這裡向量的維度會是d_model
        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        # 獲取decoder的輸入位置編碼
        self.positional_encoding = PositionalEncoding(
            d_hid=d_model, n_position=self.max_seq_len + 1)
        # 獲取特徵圖的位置編碼
        self.feat_positional_encoding = PositionalEncoding(
            d_hid=d_model, n_position=self.feat_size, dropout=feat_pe_drop)
        # 構建LN標準化層結構
        self.norm = nn.LayerNorm(d_model)

    def make_mask(self, tgt, device):
        """Make mask for self attention.

        Args:
            tgt (Tensor): Shape [N, l_tgt]
            device (torch.Device): Mask device.

        Returns:
            Tensor: Mask of shape [N * self.n_head, l_tgt, l_tgt]
        """
        # 已看過，構建自注意力模塊當中需要用到的mask
        # tgt = 要輸入到decoder的target tensor，shape [batch_size, seq_len]

        # 將tgt當中不是PAD值的地方標示為True，其他地方是False，之後進行擴維，trg_pad_mask shape = [batch_size, 1, seq_len, 1]
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).bool()
        # 獲取序列長度
        tgt_len = tgt.size(1)
        # 構建子序列mask，這裡會創建tensor shape [seq_len, seq_len]且下三角部分為True，其他地方為False，這裡是從[0, 0]就是True
        trg_sub_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device))
        # 需要trg_pad_mask與trg_sub_mask都是True的地方最終才會是True
        tgt_mask = trg_pad_mask & trg_sub_mask

        # inverse for mmcv's BaseTransformerLayer
        # 拷貝一份tgt_mask
        tril_mask = tgt_mask.clone()
        # 將tril_mask是0的地方變成-1e9
        tgt_mask = tgt_mask.float().masked_fill_(tril_mask == 0, -1e9)
        # 會將tril_mask為True的地方變成0，也就是剛才tgt_mask沒有變成-1e9的地方會變成0
        tgt_mask = tgt_mask.masked_fill_(tril_mask, 0)
        # 進行擴維 [batch_size, 1, seq_len, seq_len] -> [batch_size, n_head, seq_len, seq_len]
        tgt_mask = tgt_mask.repeat(1, self.n_head, 1, 1)
        # 進行通道調整 [batch_size, n_head, seq_len, seq_len] -> [batch_size * n_head, seq_len, seq_len]
        tgt_mask = tgt_mask.view(-1, tgt_len, tgt_len)
        return tgt_mask

    def decode(self, input, feature, src_mask, tgt_mask):
        """ 已看過，MASTER的decode部分
        Args:
            input: 輸入到decoder的資料，從下方輸入到decoder的資料
            feature: 從encoder的輸出，不過因為MASTER沒有使用encoder，所以這裡的資料是將backbone輸出通過位置編碼的結果
            src_mask: 這裡會是None
            tgt_mask: 可以說是input在使用的mask資料
        """
        # 將輸入的input透過embedding將index轉換成對應的向量
        # x shape = [batch_size, max_seq_len, channel]
        x = self.embedding(input)
        # 將通過embedding後的結果再加上位置編碼，x shape = [batch_size, max_seq_len, channel]
        x = self.positional_encoding(x)
        # 將tgt_mask與src_mask放到attn_mask當中
        attn_masks = [tgt_mask, src_mask]
        # 遍歷多層decoder層結構
        for layer in self.decoder_layers:
            # 進行層結構的向前傳遞
            x = layer(
                query=x, key=feature, value=feature, attn_masks=attn_masks)
        # 將結果通過標準化層
        x = self.norm(x)
        # 最後通過分類層將channel深度進行調整
        return self.cls(x)

    def greedy_forward(self, SOS, feature):
        input = SOS
        output = None
        for _ in range(self.max_seq_len):
            target_mask = self.make_mask(input, device=feature.device)
            out = self.decode(input, feature, None, target_mask)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        return output

    def forward_train(self, feat, out_enc, targets_dict, img_metas=None):
        """
        Args:
            feat (Tensor): The feature map from backbone of shape
                :math:`(N, E, H, W)`.
            out_enc (Tensor): Encoder output.
            targets_dict (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            img_metas: Unused.

        Returns:
            Tensor: Raw logit tensor of shape :math:`(N, T, C)`.
        """
        # 已看過，MASTER部分decoder的forward函數
        # feat = 從backbone輸出的特徵圖，tensor shape [batch_size, channel, height, width]
        # out_enc = 從encoder輸出的特徵圖，在MASTER當中沒有使用到encoder，所以這裡會是None
        # target_dict = 標註訊息也就是正確結果
        # img_metas = 圖像的詳細資訊

        # flatten 2D feature map
        if len(feat.shape) > 3:
            # 如果傳入的特徵圖是2D的這裡就會需要壓平，因為接下來要進入transformer結構
            # 保存特徵圖shape資訊
            b, c, h, w = feat.shape
            # 調整通道 [batch_size, channel, height, width] -> [batch_size, channel, height * width]
            feat = feat.view(b, c, h * w)
            # 將通道順序進行調整 [batch_size, channel, height * width] -> [batch_size, channel, height * width]
            feat = feat.permute((0, 2, 1))
        # 如果沒有encoder的輸入就用backbone輸出的特徵圖加上位置編碼代替encoder的輸出
        # out_enc shape = [batch_size, height * width = seq_len, channel]
        out_enc = self.feat_positional_encoding(feat) \
            if out_enc is None else out_enc

        # 獲取當前訓練設備
        device = feat.device
        if isinstance(targets_dict, dict):
            # 如果targets_dict是dict格式就會到這裡，將padded_targets取出，並且放到訓練設備上
            padded_targets = targets_dict['padded_targets'].to(device)
        else:
            # 直接將targets_dict作為padded_targets並且放到訓練設備上
            padded_targets = targets_dict.to(device)
        # 先將mask設定成None
        src_mask = None
        # 構建target部分的mask，tgt_mask shape = [batch_size * n_head, seq_len, seq_len]
        tgt_mask = self.make_mask(padded_targets, device=out_enc.device)
        # 進入decode的forward
        return self.decode(padded_targets, out_enc, src_mask, tgt_mask)

    def forward_test(self, feat, out_enc, img_metas):
        """
        Args:
            feat (Tensor): The feature map from backbone of shape
                :math:`(N, E, H, W)`.
            out_enc (Tensor): Encoder output.
            img_metas: Unused.

        Returns:
            Tensor: Raw logit tensor of shape :math:`(N, T, C)`.
        """

        # flatten 2D feature map
        if len(feat.shape) > 3:
            b, c, h, w = feat.shape
            feat = feat.view(b, c, h * w)
            feat = feat.permute((0, 2, 1))
        out_enc = self.feat_positional_encoding(feat) \
            if out_enc is None else out_enc

        batch_size = out_enc.shape[0]
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        output = self.greedy_forward(SOS, out_enc)
        return output
