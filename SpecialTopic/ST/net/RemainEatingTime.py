import math
import torch
from torch import nn
import numpy as np
from SpecialTopic.ST.build import build_backbone, build_head


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim, num_classes):
        super(PositionEmbedding, self).__init__()

        def get_pe(pos, i, d_model):
            fe_nmu = 1e4 ** (i / d_model)
            pe = pos / fe_nmu
            if i % 2 == 0:
                return math.sin(pe)
            else:
                return math.cos(pe)
        pe = torch.empty(max_len, embed_dim)
        for i in range(max_len):
            for j in range(embed_dim):
                pe[i, j] = get_pe(i, j, embed_dim)
        pe = pe.unsqueeze(dim=0)
        self.register_buffer('pe', pe)
        self.embed_weight = nn.Embedding(num_classes, embed_dim)
        self.embed_weight.weight.data.normal_(0, 0.1)

    def forward(self, x):
        embed = self.embed_weight(x)
        embed = embed + self.pe
        return embed


class Encoder(nn.Module):
    def __init__(self, embed_dim, heads, encoder_layers, attention_norm, mlp_ratio, dropout_ratio=0.):
        super(Encoder, self).__init__()
        self.layers = list()
        for _ in range(encoder_layers):
            self.layers.append(EncoderLayer(embed_dim, heads, attention_norm, mlp_ratio, dropout_ratio))

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, attention_norm, mlp_ratio, dropout_ratio):
        super(EncoderLayer, self).__init__()
        self.multi_head = MultiHead(embed_dim, heads, attention_norm, dropout_ratio)
        self.fpn = FPN(embed_dim, mlp_ratio, dropout_ratio)

    def forward(self, x, mask):
        out = self.multi_head(x, x, x, mask)
        out = self.fpn(out)
        return out


class Decoder(nn.Module):
    def __init__(self, embed_dim, heads, decoder_layers, attention_norm, mlp_ratio, dropout_ratio):
        super(Decoder, self).__init__()
        self.layers = list()
        for _ in range(decoder_layers):
            self.layers.append(DecoderLayer(embed_dim, heads, attention_norm, mlp_ratio, dropout_ratio))

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        for layer in self.layers:
            y = layer(x, y, mask_pad_x, mask_tril_y)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, attention_norm, mlp_ratio, dropout_ratio):
        super(DecoderLayer, self).__init__()
        self.self_multi_head = MultiHead(embed_dim, heads, attention_norm, dropout_ratio)
        self.cross_multi_head = MultiHead(embed_dim, heads, attention_norm, dropout_ratio)
        self.fpn = FPN(embed_dim, mlp_ratio, dropout_ratio)

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.self_multi_head(y, y, y, mask_tril_y)
        y = self.cross_multi_head(y, x, x, mask_pad_x)
        y = self.fpn(y)
        return y


class MultiHead(nn.Module):
    def __init__(self, embed_dim, heads, attention_norm=None, dropout_ratio=0.):
        super(MultiHead, self).__init__()
        self.heads = heads
        self.attention_norm = attention_norm
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)
        self.out_fc = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(normalized_shape=embed_dim, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, q, k, v, mask):
        clone_q = q.clone()
        q = self.norm(q)
        k = self.norm(k)
        v = self.norm(v)
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        batch_size, length, channel = q.shape
        q = q.reshape(batch_size, length, self.heads, channel // self.heads).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, length, self.heads, channel // self.heads).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, length, self.heads, channel // self.heads).permute(0, 2, 1, 3)
        norm = self.attention_norm if self.attention_norm is not None else channel // self.heads
        score = self.attention(q, k, v, mask, length, channel, norm)
        score = self.dropout(self.out_fc(score))
        score = score + clone_q
        return score

    @staticmethod
    def attention(q, k, v, mask, length, channel, norm):
        score = torch.matmul(q, k.permute(0, 1, 3, 2))
        score /= norm ** 0.5
        score = score.masked_fill_(mask, -float('inf'))
        score = torch.softmax(score, dim=-1)
        score = torch.matmul(score, v)
        score = score.permute(0, 2, 1, 3).reshape(-1, length, channel)
        return score


class FPN(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, dropout_ratio):
        super(FPN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim * mlp_ratio),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim * mlp_ratio, out_features=embed_dim),
            nn.Dropout(p=dropout_ratio)
        )
        self.norm = nn.LayerNorm(normalized_shape=embed_dim, elementwise_affine=True)

    def forward(self, x):
        clone_x = x.clone()
        x = self.norm(x)
        out = self.fc(x)
        out = out + clone_x
        return out


class RemainEatingTimeBackbone(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, max_len, embed_dim, heads, attention_norm, mlp_ratio,
                 dropout_ratio, remain_pad_val, time_pad_val, num_remain_classes, num_time_classes):
        """
        Args:
            encoder_layers: encoder層的堆疊數量
            decoder_layers: decoder層的堆疊數量
            max_len: 最長檢測長度
            embed_dim: 在注意力模塊當中每個字的編碼長度
            heads: 多頭注意力當中的頭數
            attention_norm: 在自注意當中要除上的分母，這裡是還沒有開根號的部分
            mlp_ratio: 在MLP模塊當中channel深度縮放倍率
            dropout_ratio: dropout概率
            remain_pad_val: 剩餘量的padding值
            time_pad_val: 剩餘時間的padding值
            num_remain_classes: 剩餘量的類別數
            num_time_classes: 剩餘時間類別數
        """
        super(RemainEatingTimeBackbone, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.remain_pad_val = remain_pad_val
        self.time_pad_val = time_pad_val
        self.max_len = max_len
        assert embed_dim % heads == 0, '多頭注意力的embed無法被頭數整除'
        self.embed_remain = PositionEmbedding(max_len, embed_dim, num_remain_classes)
        self.embed_time = PositionEmbedding(max_len, embed_dim, num_time_classes)
        self.encoder = Encoder(embed_dim, heads, encoder_layers, attention_norm, mlp_ratio, dropout_ratio)
        self.decoder = Decoder(embed_dim, heads, decoder_layers, attention_norm, mlp_ratio, dropout_ratio)

    def forward(self, remain, remain_time):
        mask_pad_remain = self.mask_pad(remain, self.remain_pad_val, self.max_len)
        mask_tril_time = self.mask_tril(remain_time, self.time_pad_val, self.max_len)
        remain, remain_time = self.embed_remain(remain), self.embed_time(remain_time)
        remain_output = self.encoder(remain, mask_pad_remain)
        time_output = self.decoder(remain_output, remain_time, mask_pad_remain, mask_tril_time)
        return time_output

    @staticmethod
    def mask_pad(data, pad, len):
        mask = data == pad
        mask = mask.reshape(-1, 1, 1, len)
        mask = mask.expand(-1, 1, len, len)
        return mask

    @staticmethod
    def mask_tril(data, pad, len):
        tril = 1 - torch.tril(torch.ones(1, len, len, dtype=torch.long))
        mask = data == pad
        mask = mask.unsqueeze(1).long()
        mask = mask + tril
        mask = mask > 0
        mask = (mask == 1).unsqueeze(dim=1)
        return mask


class RemainEatingTimeHead(nn.Module):
    def __init__(self, embed_dim, num_time_classes, loss_cfg, time_pad_val):
        super(RemainEatingTimeHead, self).__init__()
        support_loss_func = {
            'CrossEntropyLoss': nn.CrossEntropyLoss()
        }
        self.num_time_classes = num_time_classes
        self.loss_cfg = loss_cfg
        self.time_pad_val = time_pad_val
        self.loss_func = support_loss_func.get(loss_cfg['type'], None)
        assert self.loss_func is not None, '目前只支援Cross entropy loss'
        self.cls_fc = nn.Linear(embed_dim, num_time_classes)

    def forward(self, y, labels=None, with_loss=True):
        out = self.cls_fc(y)
        if not with_loss:
            return out
        assert labels is not None, '如果要計算loss值請給出正確答案'
        loss_dict = dict()
        labels = labels.reshape(-1)
        pred = out.reshape(-1, self.num_time_classes)
        select = labels != self.time_pad_val
        pred = pred[select]
        labels = labels[select]
        loss = self.loss_func(pred, labels)
        loss_dict['loss'] = loss
        pred = pred.argmax(1)
        correct = (pred == labels).sum().item()
        accuracy = correct / len(pred)
        loss_dict['acc'] = accuracy
        return loss_dict


class RemainEatingTime(nn.Module):
    support_phi = {
        'm': {
            'backbone': {
                'type': 'RemainEatingTimeBackbone',
                'embed_dim': 32,
                'heads': 4,
                'encoder_layers': 3,
                'decoder_layers': 3,
                'mlp_ratio': 2,
                'dropout_ratio': 0.1,
                'attention_norm': 8
            },
            'cls_head': {
                'type': 'RemainEatingTimeHead',
                'loss_cfg': {
                    'type': 'CrossEntropyLoss'
                },
                'embed_dim': 32
            }
        }
    }

    def __init__(self, phi, num_remain_classes, num_time_classes, max_len, remain_pad_val, time_pad_val,
                 pretrained='none', with_head=True):
        """
        Args:
            phi: 模型大小
            num_remain_classes: 總共剩餘量類別
            num_time_classes: 總共時間剩餘類別
            max_len: 翻譯長度
            remain_pad_val: 剩餘量的padding值
            time_pad_val: 還需多少時間的padding值
            pretrained: 預訓練權重位置，如果沒有需要加載就會是none
            with_head: 是否需要分類頭，這裡默認都會開啟
        """
        super(RemainEatingTime, self).__init__()
        self.phi = phi
        self.num_remain_classes = num_remain_classes
        self.num_time_classes = num_time_classes
        self.pretrained = pretrained
        self.max_len = max_len
        self.remain_pad_val = remain_pad_val
        self.time_pad_val = time_pad_val
        self.with_head = with_head
        model_cfg = self.support_phi.get(phi, None)
        assert model_cfg is not None, f'指定{phi}大小不支援'
        model_cfg['backbone']['max_len'] = max_len
        model_cfg['backbone']['remain_pad_val'] = remain_pad_val
        model_cfg['backbone']['time_pad_val'] = time_pad_val
        model_cfg['backbone']['num_time_classes'] = num_time_classes
        model_cfg['backbone']['num_remain_classes'] = num_remain_classes
        model_cfg['cls_head']['num_time_classes'] = num_time_classes
        model_cfg['cls_head']['time_pad_val'] = time_pad_val
        self.backbone = build_backbone(model_cfg['backbone'])
        if with_head:
            self.cls_head = build_head(model_cfg['cls_head'])
        if pretrained != 'none':
            self.init_weight_pretrained()

    def forward(self, remain, remain_time, labels=None, with_loss=True):
        out = self.backbone(remain, remain_time)
        if self.with_head:
            loss = self.cls_head(out, labels, with_loss)
        else:
            loss = out
        return loss

    def init_weight_pretrained(self):
        pretrained_dict = torch.load(self.pretrained, map_location='cpu')
        model_dict = self.state_dict()
        load_key, no_load_key, temp_dict = list(), list(), dict()
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        model_dict.update(temp_dict)
        self.load_state_dict(model_dict)
