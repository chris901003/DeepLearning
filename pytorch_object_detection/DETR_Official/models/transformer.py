# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        # 已看過
        # d_model預設為256
        # return_intermediate_dec預設為True，估計是為了每個decoder出來的都會拿去預測，可以讓每層都學得好
        super().__init__()

        # 構建一個encoder_layer後面由TransformerEncoder來把很多層疊起來
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        # 構建完整的一個TransformerEncoder
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # 構建一個decoder_layer後面由TransformerDecoder來把很多層疊起來
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        # 已看過
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # ----------------------------------------------------------------------------
        # src = 降維後的特徵圖 shape [batch_size, channel, w, h]
        # mask = 就是mask，還是一樣True的地方表示填充，False表示非填充 shape [batch_size, w, h]
        # query_embed = decoder的query的embedding，加上.weight後把embed的值拿出來(torch.nn.parameter.Parameter)
        # pos_embed = 在detr中已經把第一層重複的batch_size拿掉了，所以現在的shape [batch_size, channel, w, h]
        # ----------------------------------------------------------------------------
        # 已看過
        # flatten NxCxHxW to HWxNxC
        # 記錄下原始shape
        bs, c, h, w = src.shape
        # src shape [batch_size, channel, w, h] -> [w * h, batch_size, channel]
        src = src.flatten(2).permute(2, 0, 1)
        # pos_embed shape
        # [batch_size, channel, w, h] -> [batch_size, channel, w * h] -> [w * h, batch_size, channel]
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # [num_queries, channel] -> [num_queries, 1, channel] -> [num_queries, batch_size, channel]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask shape [channel, w, h] -> [channel, w * h]
        mask = mask.flatten(1)

        # 這裡我還是會叫預測匡為anchor
        # 構建出全為0的tensor且shape跟query_embed一樣[num_queries, batch_size, channel]
        tgt = torch.zeros_like(query_embed)
        # ----------------------------------------------------------------------------
        # 丟入encoder層
        # memory shape [w * h, batch_size, channel]
        # memory就是通過多層encoder層的輸出
        # ----------------------------------------------------------------------------
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # ----------------------------------------------------------------------------
        # 丟入decoder層
        # ----------------------------------------------------------------------------
        # hs shape [layers, num_queries, batch_size, channel]
        # layers表示輸出層數，如果要保留每層decoder輸出就會是decoder的層數
        # 如果只保留最後輸出就會是1
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # ----------------------------------------------------------------------------
        # hs = [layers, num_queries, batch_size, channel] -> [layers, batch_size, num_queries, channel]
        # => decoder輸出
        # memory = [batch_size, channel, w * h] -> [batch_size, channel, w, h] => encoder輸出
        # ----------------------------------------------------------------------------
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        """
        :param encoder_layer: 每個基礎的encoder_layer
        :param num_layers: 要重複幾個encoder_layer
        :param norm: normalize_before控制，如果是True那就會有LN，否則會是None
        """
        super().__init__()
        # 已看過
        # 傳入一個module class以及需要幾個，就會產生一個module list裡面就會有對應數量的module
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        # 預設會有
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        :param src: CNN最後一層的輸出且channel已經經過調整shape [w * h, batch_size, channel]
        :param mask: segmentation才會用到，這裡我們沒有傳入，預設為None
        :param src_key_padding_mask: 判斷該位置是否填充，如果是填充的就會是True，否則就是False
        :param pos: 位置編碼，這裡有點奇怪目前的shape是[h, batch_size, w]，後面看看怎麼用
        :return: shape [w * h, batch_size, channel]
        """
        # 已看過
        # 先將輸入給出書
        output = src

        # 經過多層encoder
        for layer in self.layers:
            # 這邊就去看TransformerEncoderLayer的forward函數
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        # output shape [w * h, batch_size, channel]
        # 看需不需要標準化
        if self.norm is not None:
            # 預設會有
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        # 已看過
        # return_intermediate預設為True，估計是為了要讓每層的decoder都可以進行預測學習，讓效果更好
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
        :param tgt: 最開始輸入到decoder的值，預設全為0，shape [num_queries, batch_size, channel]
        :param memory: encoder的輸出shape [w * h, batch_size, channel]
        :param tgt_mask: 這裡我們沒有傳入，預設為None
        :param memory_mask: 這裡我們沒有傳入，預設為None
        :param tgt_key_padding_mask: 這裡我們沒有傳入，預設為None
        :param memory_key_padding_mask: 判斷是否為padding，padding的部分會是True，否則會是False，shape [channel, w * h]
        :param pos: 位置編碼在encoder與decoder融合的時候會用到shape [w * h, batch_size, channel]
        :param query_pos: 給query用的位置編碼shape [num_queries, batch_size, channel]
        :return:
        """
        # 已看過
        # 將tgt附值給output
        output = tgt

        # decoder每層的輸出保留
        intermediate = []

        # 通過每一層decoder層
        for layer in self.layers:
            # output shape [num_queries, batch_size, channel]
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            # return_intermediate預設為True
            if self.return_intermediate:
                # 保存每一層decoder的輸出，這樣每層都可以計算loss
                intermediate.append(self.norm(output))

        # 看要不要進行標準化，如果有標準化的話要記得紀錄的也要更新
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            # 在維度0進行擴充，然後在維度0進行疊合
            # return shape [layers, num_queries, batch_size, channel]
            # layers就是我們decoder有多少層
            return torch.stack(intermediate)

        # 在多加一個維度，配合上面的shape
        # return shape [layers, num_queries, batch_size, channel]
        # layers在這裡只會有1，就是最後一層的輸出
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """
        :param d_model: 每個特徵點可以被多少維度的向量表示，這裡預設為256
        :param nhead: 多頭注意力要用幾個頭，這裡預設為8
        :param dim_feedforward: FFN中間層用到的channel深度
        :param dropout: dropout
        :param activation: 激活函數
        :param normalize_before: 預設為True
        """
        # 已看過
        super().__init__()
        # 實例化pytorch官方給的Multi-head Attention，實例化時需要給一個特徵點用多少維度的向量表示以及要用多少頭，剩下的都使選配
        # forward時最少需要給q,k,v剩下的mask之類的就是選配，輸出就是跟輸入一樣(shape)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # FFN，先升維再降維
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 暫時不知道用在哪裡
        # LayerNorm預設希望的輸入會是channel維度放在最後一個(batch_size, height, width, channels)
        # 可以透過data_format="channels_first"，使傳統(batch_size, channels, height, width)
        # 能夠正常進入LayerNorm當中進行運算
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # _get_activation_fn = 根據傳入的string會給出對應的激活函數
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 已看過
        # 如果有position_embedding就直接相加
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # 上面帶入的參數就到下面的forward看就可以了基本就是搬過來而已
        # 已看過
        # 基本上與下面的一樣，只是在最一開始的時候沒有經過標準化而已
        # 加上位置編碼
        q = k = self.with_pos_embed(src, pos)
        # 注意力機制
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # dropout
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # 上面帶入的參數就到下面的forward看就可以了基本就是搬過來而已
        # 已看過
        # 官方原始LayerNorm會需要將channel放在最後一個維度，輸入的src符合
        src2 = self.norm1(src)
        # 如果有位置編碼就直接相加，不然就返回原值
        q = k = self.with_pos_embed(src2, pos)
        # ---------------------------------------------------------------------------
        # 丟入官方實現的multi_head_self_attention
        # 需要輸入q,k,v三個參數以及mask
        # 這裡q,k基本上數值是一樣的，與v的差別只有加上的位置編碼
        # attn_mask = 作用是可以讓注意力機制不去注意被mask的部分，如果放在nlp中就是不讓注意力機制去看到後面的答案
        # 在這裡會是訓練segmentation才會用到，所以我們是給None
        # key_padding_mask = 官方解釋為標記出哪些地方是padding的也就是哪些地方不是原圖有的
        # 輸出會有兩個，第二個要在need_weights=True才會輸出
        # attn_output = 根據有沒有設定batch_first會有點不同，這裡我們不是batch_first
        # attn_output shape [w * h, batch_size, channel] (batch_first=False) << 我們是用這個
        # attn_output shape [batch_size, w * h, channel] (batch_first=True)
        # ---------------------------------------------------------------------------
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # 經過一層dropout層
        # 這裡有一個殘差
        src = src + self.dropout1(src2)
        # 通過一層標準化層
        src2 = self.norm2(src)
        # FFN的forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # 在經過一層dropout層
        # 這裡有一個殘差
        src = src + self.dropout2(src2)
        # 最後輸出 shape [w * h, batch_size, channel]
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        :param src: 輸入shape [w * h, batch_size, channel]
        :param src_mask: segmentation才會有東西，這裡我們傳入的是None
        :param src_key_padding_mask: 判斷該像素是否為填充的，填充的話會是True，否則為False
        :param pos: 位置編碼 shape [w * h, batch_size, channel]，與src有相同的shape
        :return:
        """
        # 已看過
        # 注意一下這裡只是一層的encoder而已
        if self.normalize_before:
            # normalize_before預設為True，所以會進來這裡
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        # 已看過
        # 參數傳入的與EncoderLayer一樣
        super().__init__()
        # 實例化multi-head attention，這個是用在self attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 實例化multi-head attention，這個是用在後面與encoder結合時的attention
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # FFN，先升維再降維
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 目前暫時不知道
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # _get_activation_fn = 根據傳入的string會給出對應的激活函數
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 已看過
        # 有位置編碼就加上去，沒有就直接回傳
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # 已看過
        # 這裡只負責一層的DecoderLayer
        # 帶入的參數下面forward有寫
        # q,k添加上位置編碼
        q = k = self.with_pos_embed(tgt, query_pos)
        # 經過自注意力機制，這部分細節可以去看TransformerEncoderLayer的forward就可以了
        # tgt2 shape [num_queries, batch_size, channel]
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # dropout層
        tgt = tgt + self.dropout1(tgt2)
        # 標準化層，注意一下channel的位置
        tgt = self.norm1(tgt)
        # ----------------------------------------------------------------------------
        # 與encoder的輸出進行融合，得到一個新的注意力
        # query = 拿decoder的來用，要加上位置編碼，這裡用到的是query用的編碼，與輸入時不同
        # key = 拿encoder的來用，要加上位置編碼，這裡用到的是非訓練的位置編碼
        # value = 拿encoder的來用，這裡不用加上位置編碼
        # attn_mask = None
        # key_padding_mask = 判斷是不是padding
        # tgt2 shape [num_queries, batch_size, channel]
        # ----------------------------------------------------------------------------
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # dropout層
        # 這裡有一個殘差
        tgt = tgt + self.dropout2(tgt2)
        # 標準化
        tgt = self.norm2(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # dropout + 標準化
        # 這裡有一殘差
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        # tgt shape [num_queries, batch_size, channel]
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        # 已看過
        # 基本上與上面相同，我們通常會進這裡，標註的時候標到上面那裡去了
        # 帶入的參數下面forward有寫
        # 只差在這行而已
        tgt2 = self.norm1(tgt)
        # 位置編碼
        q = k = self.with_pos_embed(tgt2, query_pos)
        # 自注意力機制
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        # 與encoder融合進行注意力機制
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        # 殘差
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        # 殘差
        tgt = tgt + self.dropout3(tgt2)
        # tgt shape [num_queries, batch_size, channel]
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        """
            :param tgt: 最開始輸入到decoder的值shape [num_queries, batch_size, channel]
            :param memory: encoder的輸出shape [w * h, batch_size, channel]
            :param tgt_mask: 這裡我們沒有傳入，預設為None
            :param memory_mask: 這裡我們沒有傳入，預設為None
            :param tgt_key_padding_mask: 這裡我們沒有傳入，預設為None
            :param memory_key_padding_mask: 判斷是否為padding，padding的部分會是True，否則會是False，shape [channel, w * h]
            :param pos: 位置編碼在encoder與decoder融合的時候會用到shape [w * h, batch_size, channel]
            :param query_pos: 給query用的位置編碼shape [num_queries, batch_size, channel]
            :return:
        """
        # 已看過
        # 看在動作前需不需要標準化
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    # 已看過
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    # ----------------------------------------------------------------------------
    # d_model = 輸入的每個點都會有hidden_dim維度向量，對於encoder來說就是每個feature_map上
    # 的特徵點的channel會變成的大小，這裡預設是256
    # dropout = 就是dropout，預設是0.1
    # nhead = 多頭注意力機制的頭數，預設是8
    # dim_feedforward = FFN的中間channel大小，預設為2048
    # num_encoder_layers = encoder模塊重複次數，預設為6
    # num_decoder_layers = decoder模塊重複次數，預設為6
    # normalize_before = 目前還不知道，預設是False
    # return_intermediate_dec = 目前不知道是啥，預設是True
    # ----------------------------------------------------------------------------
    # 已看過
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    # 已看過
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
