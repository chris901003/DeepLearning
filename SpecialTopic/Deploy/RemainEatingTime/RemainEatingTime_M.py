import os
import math
import pickle
import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim, num_classes):
        super(PositionEmbedding, self).__init__()
        pe = torch.asarray([[self.get_pe(i, j, embed_dim) for j in range(embed_dim)] for i in range(max_len)])
        pe = pe.unsqueeze(dim=0)
        self.register_buffer('pe', pe)
        self.embed_weight = nn.Embedding(num_classes, embed_dim)

    def forward(self, x):
        embed = self.embed_weight(x)
        embed = embed + self.pe
        return embed

    @staticmethod
    @torch.jit.ignore
    def get_pe(pos, i, d_model):
        fe_nmu = 1e4 ** (i / d_model)
        pe = pos / fe_nmu
        if i % 2 == 0:
            return math.sin(pe)
        else:
            return math.cos(pe)


class Encoder(nn.Module):
    def __init__(self, embed_dim, heads, encoder_layers, attention_norm, mlp_ratio, dropout_ratio=0.):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
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
        # 在驗證模式下dropout不會產生任何作用
        self.dropout = nn.Identity()

    def forward(self, q, k, v, mask):
        clone_q = q.clone()
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        batch_size, length, channel = q.shape
        q = q.reshape(batch_size, length, self.heads, channel // self.heads).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, length, self.heads, channel // self.heads).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, length, self.heads, channel // self.heads).permute(0, 2, 1, 3)
        norm = self.attention_norm
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
            nn.Identity()
        )
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=True)

    def forward(self, x):
        clone_x = x.clone()
        x = self.norm(x)
        out = self.fc(x)
        out = out + clone_x
        return out


class RemainEatingTimeM(nn.Module):
    def __init__(self, setting_file_path):
        super(RemainEatingTimeM, self).__init__()
        self.setting_file_path = setting_file_path
        setting = self.parser_cfg()
        num_remain_classes = setting['num_remain_classes']
        num_time_classes = setting['num_time_classes']
        max_len = setting['max_len']
        remain_pad_val = setting['remain_pad_val']
        time_pad_val = setting['time_pad_val']
        embed_dim = 32
        heads = 4
        encoder_layers = 3
        decoder_layers = 3
        mlp_ratio = 2
        dropout_ratio = 0.1
        attention_norm = 8
        self.embed_remain = PositionEmbedding(max_len, embed_dim, num_remain_classes)
        self.embed_time = PositionEmbedding(max_len, embed_dim, num_time_classes)
        self.encoder = Encoder(embed_dim, heads, encoder_layers, attention_norm, mlp_ratio, dropout_ratio)
        self.decoder = Decoder(embed_dim, heads, decoder_layers, attention_norm, mlp_ratio, dropout_ratio)

    def forward(self):
        pass

    @torch.jit.ignore
    def parser_cfg(self):
        assert os.path.exists(self.setting_file_path)
        if os.path.splitext(self.setting_file_path)[1] == '.pickle':
            with open(self.setting_file_path, 'rb') as f:
                results = pickle.load(f)
        else:
            raise NotImplementedError('目前只有支持[pickle]格式')
        return results


if __name__ == '__main__':
    print('Testing Remain Eating Time transfer to onnx model')
    RemainEatingTimeM(setting_file_path=None)
