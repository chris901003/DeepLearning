# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    # 由下面的build_position_encoding構建
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        # 已看過
        # normalize預設為True
        super().__init__()
        # num_pos_feats = 進入Transformer前的channel的一半，這裡預設會是128
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        # 已看過
        # NestedTensor詳細內容到detr.py中找或是到misc.py都可以
        # 將NestedTensor中的tensor以及mask分別拿出來
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        # 對mask做反向，變成擴充的地方是False真正有圖片的地方是True
        not_mask = ~mask
        # cumsum = 在第幾個維度上面做累加
        # ex:[[1, 2, 3], [4, 5, 6]].cumsum(dim=0) => [[1, 3, 6], [4, 9, 15]]
        # ex:[[1, 2, 3], [4, 5, 6]].cumsum(dim=1) => [[1, 2, 3], [5, 7, 9]]
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            # 會進來
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 我們多創建一個維度，這個維度表示每個特徵點用多少維度的向量表示
        # x_embed shape [batch_size, w, h] -> pos_x shape [batch_size, w, h, channel]
        # x_embed / dim_t就是照著公式來的
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # 偶數channel index我們用sin，奇數channel index我們用cos，最後在維度4方向做疊合
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # 重新調整順序，最後shape = [batch_size, channel, w, h]
        # 在上面init時會發現num_pos_feats會是輸入transformer的channel的一半，是因為這裡會做concat就會變回需要的channel數了
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # 這個就是pos_embedding的數值了，不會再改變
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    # 由下面的build_position_encoding構建
    def __init__(self, num_pos_feats=256):
        # 已看過
        super().__init__()
        # 這裡預設傳進來的num_pos_feats = 128
        # nn.Embedding(x, y) = 會構建出一個大小為x且每個x會給一個向量維度為num_pos_feats的東西
        # 這裡個給50個是一個給基數一個給偶數
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        # 已看過
        # 初始化row_embed以及col_embed，方式為正態分佈且mean=0, std=1
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        # NestedTensor詳細內容到detr.py中找或是到misc.py都可以
        # 先把tensor拿出來
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        # 找到對應index對應上去的向量
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    # 已看過
    # hidden_dim = 256，這裡就是在進入Transformer前會透過一個kernel_size=1的Conv把channel變成256
    # N_steps = 一半的hidden_dim，估計是會構建出兩個embedding一個是給基數一個給偶數
    N_steps = args.hidden_dim // 2
    # position_embedding在main中只有sine或是learned兩種選項
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    # 把實例化後的position_embedding物件回傳回去
    return position_embedding
