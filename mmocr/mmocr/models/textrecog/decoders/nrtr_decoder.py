# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import ModuleList

from mmocr.models.builder import DECODERS
from mmocr.models.common import PositionalEncoding, TFDecoderLayer
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class NRTRDecoder(BaseDecoder):
    """Transformer Decoder block with self attention mechanism.

    Args:
        n_layers (int): Number of attention layers.
        d_embedding (int): Language embedding dimension.
        n_head (int): Number of parallel attention heads.
        d_k (int): Dimension of the key vector.
        d_v (int): Dimension of the value vector.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
        d_inner (int): Hidden dimension of feedforward layers.
        n_position (int): Length of the positional encoding vector. Must be
            greater than ``max_seq_len``.
        dropout (float): Dropout rate.
        num_classes (int): Number of output classes :math:`C`.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        start_idx (int): The index of `<SOS>`.
        padding_idx (int): The index of `<PAD>`.
        init_cfg (dict or list[dict], optional): Initialization configs.

    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss as specified in
        :obj:`mmocr.models.textrecog.recognizer.EncodeDecodeRecognizer`.
    """

    def __init__(self,
                 n_layers=6,
                 d_embedding=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=256,
                 n_position=200,
                 dropout=0.1,
                 num_classes=93,
                 max_seq_len=40,
                 start_idx=1,
                 padding_idx=92,
                 init_cfg=None,
                 **kwargs):
        """ 已看過，transformer decoder部分且這裡包含自注意力機制
        Args:
            n_layers: 總共堆疊多少層自注意力結構
            d_embedding: 將一個序列的文字用多少維度的相相進行表達
            n_head: 多頭自注意力當中的頭數
            d_k: 每個頭當中key的channel深度
            d_v: 每個頭當中value的channel深度
            d_model: 輸入的channel深度
            d_inner: 在FFN中間層的channel深度
            n_position: 最長支援多長的位置編碼，這裡一定會大於設定的最長序列長度
            dropout: dropout概率
            num_classes: 總分類數
            max_seq_len: 最長序列長度
            start_idx: <SOS>的index
            padding_idx: <PAD>的index
            init_cfg: 初始化方式
        """
        # 繼承自BaseDecoder，將繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)

        # 保存傳入的資料
        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.max_seq_len = max_seq_len

        # 這裡創建Embedding，是為了將標註訊息對應上的index轉換成對應到一個向量，也就是原先一個文字是對應到一個int變成對應到一個
        # 深度為d_embedding的向量
        self.trg_word_emb = nn.Embedding(
            num_classes, d_embedding, padding_idx=padding_idx)

        # 獲取位置編碼實例對象，這是要給標註資料用的
        self.position_enc = PositionalEncoding(
            d_embedding, n_position=n_position)
        # 構建dropout層結構
        self.dropout = nn.Dropout(p=dropout)

        # 構建多層decoder層結構
        self.layer_stack = ModuleList([
            # 這裡使用的是TFDecoderLayer類
            TFDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout, **kwargs)
            for _ in range(n_layers)
        ])
        # 最終透過LN進行標準化
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        pred_num_class = num_classes - 1  # ignore padding_idx
        # 最後通過全連接層獲取最後channel深度
        self.classifier = nn.Linear(d_model, pred_num_class)

    @staticmethod
    def get_pad_mask(seq, pad_idx):
        """ 已看過，獲取pad的mask部分
        Args:
            seq: 標註資訊，tensor shape [batch_size, seq_len]
            pad_idx: padding部分的index
        """

        # 將seq當中不是padding index的地方，不是padding index的地方會是True否則就會是False
        # return shape = [batch_size, 1, seq_len]
        return (seq != pad_idx).unsqueeze(-2)

    @staticmethod
    def get_subsequent_mask(seq):
        """For masking out the subsequent info."""
        # 已看過，獲取子序列的mask資訊，這裡主要是因為transformer在decode時會是有先後順序的關係，前向資料無法看到後項資料
        # seq = 標註訊息，tensor shape = [batch_size, max_seq_len]

        # 獲取序列長度
        len_s = seq.size(1)
        # torch.triu = 獲取矩陣上三角的資料
        # 這裡會先構建一個全為1且shape是[len_s, len_s]的tensor，之後取上三角部分最後用1去剪，所以結果是獲取下三角都為1其他都為0
        # [[1, 0, 0],
        #  [1, 1, 0],
        #  [1, 1, 1]]
        subsequent_mask = 1 - torch.triu(
            torch.ones((len_s, len_s), device=seq.device), diagonal=1)
        # 多添加一個通道並且將數據形態改成bool，subsequent_mask shape = [1, seq_len, seq_len]
        subsequent_mask = subsequent_mask.unsqueeze(0).bool()

        # 回傳結果
        return subsequent_mask

    def _attention(self, trg_seq, src, src_mask=None):
        """ 已看過，自注意力層結構
        Args:
            trg_seq: 標註訊息，tensor shape [batch_size, max_seq_len]，如果當前圖像長度不到max_seq_len就會用padding的index代替
            src: 特徵向量，tensor shape [batch_size, height * width, channel]
            src_mask: mask部分，會將需要忽略的地方為0，其他地方會是1
        """
        # 將trg_seq通過trg_word_emb，將標註訊息的資料轉成向量型態，原先一個文字對應上的是一個index，這裡會將index映射到對應的向量
        # trg_embedding shape = [batch_size, max_seq_len, channel]
        trg_embedding = self.trg_word_emb(trg_seq)
        # 將trg_embedding加上位置編碼
        trg_pos_encoded = self.position_enc(trg_embedding)
        # 再通過一層dropout層結構
        tgt = self.dropout(trg_pos_encoded)

        # 獲取mask的部分
        # get_subsequent_mask = 獲取下三角型的mask，在矩陣下三角型的地方會是1其他地方會是0，shape [1, seq_len, seq_len]
        # get_pad_mask = 將trg_seq當中是padding部分設定成False，其他地方會是True，shape [batch_size, 1, seq_len]
        # 最後將兩個結果用and進行連接，trg_mask shape = [batch_size, seq_len, seq_len]，需要被忽略的地方會是False，否則就會是True
        trg_mask = self.get_pad_mask(
            trg_seq,
            pad_idx=self.padding_idx) & self.get_subsequent_mask(trg_seq)
        # 將tgt資料放到output上
        output = tgt
        # 開始進行注意力層結構的正向傳播
        for dec_layer in self.layer_stack:
            # 將資料傳入到transformer decoder的正向傳播當中，同時這裡會需要mask資訊
            output = dec_layer(
                output,
                src,
                self_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
        # 最後將output通過LN標準化層
        output = self.layer_norm(output)

        # 回傳，shape = [batch_size, max_seq_len, channel]
        return output

    def _get_mask(self, logit, img_metas):
        """ 已看過，獲取mask資訊
        Args:
            logit: 從encoder輸出的資料，shape [batch_size, seq_len, channel]
            img_metas: 圖像的詳細資料
        """
        # 先將valid_ratios設定成None
        valid_ratios = None
        if img_metas is not None:
            # 如果有傳入img_metas就以當中的valid_ratios作為標準
            valid_ratios = [
                img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
            ]
        # 獲取logit得shape資料，這裡會拿batch_size以及seq_len
        N, T, _ = logit.size()
        # 先將mask設定成None
        mask = None
        if valid_ratios is not None:
            # 如果有找到valid_ratios資訊就會進來
            # 構建全為0且shape為[batch_size, seq_len]的tensor
            mask = logit.new_zeros((N, T))
            # 遍歷整個batch的valid_ratios資料
            for i, valid_ratio in enumerate(valid_ratios):
                # 獲取合法的寬度部分
                valid_width = min(T, math.ceil(T * valid_ratio))
                # 將合法的地方設定成1，其他需要忽略的地方就維持是0
                mask[i, :valid_width] = 1

        # 最後將mask回傳
        return mask

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        r"""
        Args:
            feat (None): Unused.
            out_enc (Tensor): Encoder output of shape :math:`(N, T, D_m)`
                where :math:`D_m` is ``d_model``.
            targets_dict (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, T, C)`.
        """
        # 已看過，NRTR的decoder部分
        # feat = 特徵圖資訊，tensor shape [batch_size, channel, height, width]，這裡不會用到
        # out_enc = 從encoder輸出的資訊，tensor shape [batch_size, seq_len, channel]
        # targets_dict = 標註的資料
        # img_metas = 圖像的詳細資料

        # 獲取mask資訊，src_mask shape = [batch_size, seq_len]，會在需要忽略的地方是0，需要的地方會是1
        src_mask = self._get_mask(out_enc, img_metas)
        # 將targets_dict當中的padded_targets取出並且放到當前運行設備上面
        targets = targets_dict['padded_targets'].to(out_enc.device)
        # 通過注意力機制層結構，attn_output shape = [batch_size, seq_len, channel]
        attn_output = self._attention(targets, out_enc, src_mask=src_mask)
        # 最後通過分類層，簡單使用一個全連接層將channel調整到預測分類數量，outputs shape = [batch_size, seq_len, num_classes]
        outputs = self.classifier(attn_output)

        return outputs

    def forward_test(self, feat, out_enc, img_metas):
        src_mask = self._get_mask(out_enc, img_metas)
        N = out_enc.size(0)
        init_target_seq = torch.full((N, self.max_seq_len + 1),
                                     self.padding_idx,
                                     device=out_enc.device,
                                     dtype=torch.long)
        # bsz * seq_len
        init_target_seq[:, 0] = self.start_idx

        outputs = []
        for step in range(0, self.max_seq_len):
            decoder_output = self._attention(
                init_target_seq, out_enc, src_mask=src_mask)
            # bsz * seq_len * C
            step_result = F.softmax(
                self.classifier(decoder_output[:, step, :]), dim=-1)
            # bsz * num_classes
            outputs.append(step_result)
            _, step_max_index = torch.max(step_result, dim=-1)
            init_target_seq[:, step + 1] = step_max_index

        outputs = torch.stack(outputs, dim=1)

        return outputs
