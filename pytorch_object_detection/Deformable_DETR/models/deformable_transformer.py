# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        """
        :param d_model: 每一個特徵點會用多少深度的特徵向量表示
        :param nhead: 多頭注意力機制中要用多少頭
        :param num_encoder_layers: encoder的堆疊層數
        :param num_decoder_layers: decoder的堆疊層數
        :param dim_feedforward: FFN中間層的channel深度
        :param dropout: dropout率
        :param activation: 激活函數的選擇
        :param return_intermediate_dec: 是否將decoder的中間層輸出保留，這裡會是True
        :param num_feature_levels:
        :param dec_n_points:
        :param enc_n_points:
        :param two_stage: 預設為False
        :param two_stage_num_proposals: 如果是使用2-stage模式，就會在encoder就會有proposal，我們會取前top-k個
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        # 構建一層的encoder層
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        # 透過輸入一層的encoder以及需要的層數構建出多層encoder層
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        # 構建一層的decoder層
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        # 構建多層decoder層結構，同時需要傳入是否要保留中間層輸出
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            # 2-stage會在這個地方
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            # 1-stage預設會往這個地方
            self.reference_points = nn.Linear(d_model, 2)

        # 初始化權重
        self._reset_parameters()

    def _reset_parameters(self):
        # 將層結構當中的權重進行初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        # 獲取proposal的位置編碼
        # proposals shape [batch_size, topk, 4]
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        # dim_t shape [128]
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # proposals shape [batch_size, top_k, 4]
        proposals = proposals.sigmoid() * scale
        # proposals shape [batch_size, top_k, 4, 128]
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        # pos shape [batch_size, top_k, 4, 64, 2] -> [batch_size, top_k, 512]
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        # return shape [batch_size, top_k, 512]
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """
        :param memory: shape [batch_size, total_pixel, channel=256]
        :param memory_padding_mask: [batch_size, total_pixel]
        :param spatial_shapes: [lvl, 2]
        :return:
        """
        # 這裡只有在2-stage中通過encoder時會進來
        # 獲取encoder的輸出維度資料
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        # 遍歷每一層特徵層
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # mask_flatten shape [batch_size, height, width, 1]
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            # 統計總共有多少個點是非padding的
            # valid_H shape [batch_size], valid_W shape [batch_size]
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            # grid_y, grid_x shape [height, width]
            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            # grid shape [height, width, 2]
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            # scale shape [batch_size, 1, 1, 2]
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            # grid shape [height, width, 2] -> [1, height, width, 2] -> [batch_size, height, width, 2]
            # 最後除以了scale會將座標縮放到相對空間當中
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            # wh shape [batch_size, height, width, 2]
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            # 將gird與wh合併變成一個預測匡，前面的是(x, y)座標後面的是(w, h)高寬
            # proposal shape [batch_size, height, width, 4] -> [batch_size, height * width, 4]
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            # 添加到proposals當中記錄下來
            proposals.append(proposal)
            # 這樣下一層才會知道mask要從哪裡開始
            _cur += (H_ * W_)
        # output_proposals shape [batch_size, total_pixel, 4]
        output_proposals = torch.cat(proposals, 1)
        # all就是檢查是否全為True，keepdim就是會不會保留最後一個維度
        # output_proposals_valid shape [batch_size, total_pixel, 1] (True, False)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        # output_proposals shape [batch_size, total_pixel, 4]
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        # 將padding的部分全部設定為無窮，表示沒有作用
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        # 將不在范為內的值設定成無窮，表示沒有作用
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        # 將padding的部分設定成0
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        # 將無效區域設定成0
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        # 透過一層全連接層再透過一次標準化層
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        # output_memory shape [batch_size, total_pixel, channel=256]
        # output_proposals shape [batch_size, total_pixel, channel=4]
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        # mask shape [batch_size, height, width]
        _, H, W = mask.shape
        # valid_H, valid_W shape [batch_size]
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        # 將值控制在[0, 1]，也就轉換成比例
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        # valid_ratio shape [batch_size, 2]
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """
        :param srcs: 特徵圖，List[tensor (batch_size, channel, height, width)]
        :param masks: 特徵圖對應上的mask，List[tensor (batch_size, height, width)]
        :param pos_embeds: 對於特徵圖的位置編碼，List[tensor (batch_size, channel, height, width)]
        :param query_embed: query的位置編碼
        :return:
        """
        # 如果使用2-stage就不會傳入query_embed，如果是用1-stage就會傳入query_embed
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        # 將準備放入encoder資料處理
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        # 遍歷所有特徵圖
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            # 獲取特徵圖shape
            bs, c, h, w = src.shape
            # 空間上的shape，之後還原會需要用到
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            # src shape [batch_size, channel, height, width] -> [batch_size, channel, height * width]
            # -> [batch_size, height * width, channel]
            src = src.flatten(2).transpose(1, 2)
            # mask shape [batch_size, height, width] -> [batch_size, height * width]
            mask = mask.flatten(1)
            # pos_embed shape變化與src相同，shape [batch_size, height * width, channel]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            # level_embed[lvl] shape [1, 1, hidden_dim]
            # 這裡的作用是每一層的特徵層又會透過一個層位置編碼進行分別
            # lvl_pos_embed shape [batch_size, height * width, hidden_dim=256]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # 將結果保存下來
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        # 將height*width維度全部concat在一起
        # src_flatten shape [batch_size, total_pixel, hidden_dim]
        src_flatten = torch.cat(src_flatten, 1)
        # mask_flatten shape [batch_size, total_pixel]
        mask_flatten = torch.cat(mask_flatten, 1)
        # lvl_pos_embed_flatten shape [batch_size, total_pixel, hidden_dim]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # 將spatial_shapes轉成tensor格式，shape [lvl, 2]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # level_start_index shape [lvl]
        # 裏面的值表示第i個層是從哪個index開始，因為我們把height*width全部concat所以需要這個東西記錄下來
        # Ex: spatial_shapes = [[5, 5], [7, 7], [9, 9]]
        # level_start_index = [0, 25, 74]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # valid_ratios shape = [batch_size, lvl, 2]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        # 進入encoder開始向前傳播
        # memory shape [batch_size, total_pixel, channel=256]
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                              lvl_pos_embed_flatten, mask_flatten)

        # prepare input for decoder
        # 獲取encoder輸出資料
        bs, _, c = memory.shape
        if self.two_stage:
            # 2-stage會走這裡
            # output_memory shape [batch_size, total_pixel, channel=256]
            # output_proposals shape [batch_size, total_pixel, channel=4]
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            # self.decoder.num_layers = decoder總共會重複堆疊多少層
            # self.decoder.class_embed = 應該要是一個全連結層，將channel深度調整到與num_classes相同
            # 也就是通過class_embed就會是預測出來的分類類別，只不過這裡好像沒有實作完成
            # enc_outputs_class shape [batch_size, total_pixel, num_classes]
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            # enc_outputs_coord_unact shape [batch_size, total_pixel, channel=4]
            # 這裡就可以獲得預測的邊界匡
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            # 獲取我們會取的前k個proposals
            topk = self.two_stage_num_proposals
            # 這裡有一點問題，我們只去用第一種類別的預測概率去選則保留的匡，如果這裡是二分類那就沒有問題
            # 同時這裡的class_embed是使用decoder中的，可能會導致decoder訓練過程中傾向對第一類別的分類
            # 這邊應該可以改成先找到每個預測類別的最大概率類別，之後再依據這個類別的分數進行挑選前k大的，這樣就可以顧及到所有類別
            # topk_proposals shape [batch_size, topk=300]
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            # topk_coords_unact shape [batch_size, topk, 4]，獲取到我們需要的標註訊息
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            # 取消梯度
            topk_coords_unact = topk_coords_unact.detach()
            # 使用sigmoid將值控制在[0, 1]之間
            # reference_points shape [batch_size, top_k, 4]
            reference_points = topk_coords_unact.sigmoid()
            # init_reference_out最終會被放到decoder中作為初始bbox估計
            init_reference_out = reference_points
            # get_proposal_pos_embed shape [batch_size, top_k, 512]
            # pos_trans = 將剛剛輸出的部分再通過一個全連接層，這裡的channel不會改變
            # 做後再通過pos_trans_norm進行標準化層
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            # query_embed, tgt shape [batch_size, top_k, 256]
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            # 1-stage預設會是走這裡
            # query_embed, tgt shape [num_queries, channel=256]
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            # shape [num_queries, channel] -> [1, num_queries, channel] -> [batch_size, num_queries, channel]
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            # shape [num_queries, channel] -> [1, num_queries, channel] -> [batch_size, num_queries, channel]
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            # reference_points shape [batch_size, num_queries, channel=2]
            reference_points = self.reference_points(query_embed).sigmoid()
            # init_reference_out shape [batch_size, num_queries, channel=2]
            init_reference_out = reference_points

        # decoder
        # hs shape [decoder_layers, batch_size, num_queries, channel]
        # inter_references [decoder_layers, batch_size, num_queries=300, levels=4, 2]
        # 如果沒有開啟輔助訓練，最前面的decoder_layers就不會有，也就是會直接少掉第0個維度
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            # 2-stage會回傳這兩個東西，最後可以用來計算損失
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        # 我們會回傳下面這個
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        """
        :param d_model: 一個特徵點用多少維度的向量表示
        :param d_ffn: FFN中間層的channel深度
        :param dropout: dropout率
        :param activation: 激活函數
        :param n_levels:
        :param n_heads: 多頭注意力機制中的頭數
        :param n_points:
        """
        super().__init__()

        # self attention
        # 構建multi_head self deformable attention
        # MSDeformAttn class在models.ops.modules.ms_deform_attn.py當中
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        # 將一些層結構失效
        self.dropout1 = nn.Dropout(dropout)
        # 標準化層
        self.norm1 = nn.LayerNorm(d_model)

        # ffn，多層感知機結構
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        # 如果有位置編碼就將位置編碼添加上去，否則就直接返回
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        # FFN
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        """
        :param src: [batch_size, total_pixel, channel]
        :param pos: [batch_size, total_pixel, channel]
        :param reference_points: [batch_size, total_pixel, lvl, 2]
        :param spatial_shapes: [lvl, 2]
        :param level_start_index: [lvl]
        :param padding_mask: [batch_size, total_pixel]
        :return:
        """
        # self attention
        # src2 shape [batch_size, total_pixel, channel]
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes,
                              level_start_index, padding_mask)
        # 這邊是一個殘差結構
        src = src + self.dropout1(src2)
        # 標準化層
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        # src shape [batch_size, total_pixel, channel=256]
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        # 構建多層encoder層
        super().__init__()
        # 獲取多層encoder層
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        :param spatial_shapes: shape [lvl, 2]
        :param valid_ratios: shape [batch_size, lvl, 2]
        :param device: 訓練設備
        :return:
        """
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            # ref_y, ref_x shape [H_, W_]
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            # ref_y, ref_x shape [2, H_ * W_]
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            # ref shape [batch_size, H_ * W_, 2]
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # reference_points shape [batch_size, total_pixel, 2]
        reference_points = torch.cat(reference_points_list, 1)
        # reference_points shape [batch_size, total_pixel, lvl, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        """
        :param src: [batch_size, total_pixel, hidden_dim]
        :param spatial_shapes: [lvl, 2]
        :param level_start_index: [lvl]
        :param valid_ratios: [batch_size, lvl, 2]
        :param pos: [batch_size, total_pixel, hidden_dim]
        :param padding_mask: [batch_size, total_pixel]
        :return:
        """
        output = src
        # reference_points shape [batch_size, total_pixel, lvl, 2]
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            # output shape [batch_size, total_pixel, channel=256]
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        # output shape [batch_size, total_pixel, channel=256]
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        """
        :param d_model: 一個特徵點由多少維度向量來表示
        :param d_ffn: FFN中間層結構的channel深度
        :param dropout: dropout率
        :param activation: 激活函數
        :param n_levels:
        :param n_heads: 多頭注意力機制中的頭數
        :param n_points:
        """
        super().__init__()

        # cross attention
        # 這個與encoder相同
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        # 表準化層結構
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        # 這裡就是用pytorch官方的自注意力模塊，因為這裡與圖像沒有關係所以不會需要用到deformable
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        # 如果有位置編碼就將位置編碼加上去
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        # FFN
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index,
                src_padding_mask=None):
        """
        :param tgt: shape [batch_size, num_queries=300, channel=256]
        :param query_pos: shape [batch_size, num_queries=300, channel=256]
        :param reference_points: 1-stage shape [batch_size, num_queries=300, levels=4, 2]
        2-stage shape [batch_size, num_queries, levels=4, channel=4]
        :param src: shape [batch_size, total_pixel, channel=256]
        :param src_spatial_shapes: shape [levels, 2]
        :param level_start_index: shape [levels]
        :param src_padding_mask: shape [batch_size, total_pixel]
        :return:
        """
        # self attention
        # 將query以及key添加上position embedding
        # q, k shape [batch_size, num_queries, channel]
        q = k = self.with_pos_embed(tgt, query_pos)
        # 將query以及key以及value傳入，這裡用的是pytorch官方實現的多頭注意力機制
        # 預設會是要將batch_size放在第1維度，所以這裡都會先進行transpose
        # 官方的多頭注意力會有兩個回傳，第一個是我們要的，第二個會是attn_output_weights
        # tgt2 shape [batch_size, num_queries, channel]
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        # 透過殘差結構進行相加
        tgt = tgt + self.dropout2(tgt2)
        # 標準化層結構
        tgt = self.norm2(tgt)

        # cross attention
        # 透過與encoder輸出的值進行融合
        # tgt2 shape [batch_size, num_queries=300, channel=256]
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        # 殘差結構
        tgt = tgt + self.dropout1(tgt2)
        # 標準化層
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        # tgt shape [batch_size, num_queries=300, channel=256]
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        # 獲取多層decoder層
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        """
        :param tgt: 在1-stage下shape [batch_size, num_queries=300, channel=256]
        :param reference_points: 在1-stage下shape [batch_size, num_queries=300, channel=2]
        在2-stage下shape [batch_size, num_queries, channel=4]
        :param src: shape [batch_size, total_pixel, channel=256]
        :param src_spatial_shapes: shape [levels, 2]
        :param src_level_start_index: [levels]
        :param src_valid_ratios: [batch_size, levels, 2]
        :param query_pos: shape [batch_size, num_queries, channel]
        :param src_padding_mask: shape [batch_size, total_pixel]
        :return:
        """
        # 先將output設定為tgt
        output = tgt

        # 一些保存的東西
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                # 2-stage會走這裡
                # [batch_size, num_queries, 1, channel=4] \ [batch_size, 1, levels, 4]
                # reference_points_input shape [batch_size, num_queries, levels, channel=4]
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                # 1-stage會走這裡
                assert reference_points.shape[-1] == 2
                # [batch_size, num_queries=300, 1, channel=2] * [batch_size, 1, levels=4, 2]
                # reference_points_input shape [batch_size, num_queries=300, levels=4, 2]
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            # output shape [batch_size, num_queries=300, channel=300]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index,
                           src_padding_mask)

            # hack implementation for iterative bounding box refinement
            # 這裡預設bbox_embed會是None
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            # 如果有開啟輔助訓練，這裡就會將每層decoder的輸出拿去做預測，最後都會進行損失計算
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            # 最後拼接再一起，預設stack會在第0個維度上面進行堆疊
            # output shape [decoder_layers, batch_size, num_queries, channel]
            # reference_points [decoder_layers, batch_size, num_queries=300, levels=4, 2]
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        # 這裡疑似有問題正常來說應該要加上[]否則會與上方的產生問題，因為回傳的shape會有差異導致後面處理有問題
        # 也就是回傳得應該要是 return [output], [reference_points]
        return output, reference_points


def _get_clones(module, N):
    # 將多層相同結構的層結構透過ModuleList構建在一起
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    # 透過輸入的字串回傳相對應的激活實例化對象
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    # 構建deforamble_transformer並且回傳
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries)


