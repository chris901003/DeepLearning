import os
import pickle
import torch
from torch import nn
import numpy as np
from SpecialTopic.Deploy.RemainEatingTime.RemainEatingTime_M import PositionEmbedding, Encoder, Decoder


def parser_setting(setting_file_path):
    assert setting_file_path is None or os.path.exists(setting_file_path), '給定的設定檔不存在'
    if os.path.splitext(setting_file_path)[1] == '.pickle':
        with open(setting_file_path, 'rb') as f:
            results = pickle.load(f)
    else:
        raise NotImplementedError('目前只有支持[pickle]格式')
    return results


def load_pretrained(model, pretrained_path):
    assert os.path.exists(pretrained_path), '提供的模型權重不存在'
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    if 'model_weight' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_weight']
    load_key, no_load_key, temp_dict = list(), list(), dict()
    for k, v in pretrained_dict.items():
        idx = k.find('.')
        if k[:idx] == 'backbone' or k[:idx] == 'cls_head':
            new_name = k[idx + 1:]
        else:
            new_name = k
        if new_name in model_dict.keys() and np.shape(model_dict[new_name]) == np.shape(v):
            temp_dict[new_name] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    assert len(no_load_key) == 0, '給定的預訓練權重與模型不匹配'
    return model


def load_encoder_pretrained(model, pretrained_path):
    assert os.path.exists(pretrained_path), '提供的模型權重不存在'
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    if 'model_weight' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_weight']
    load_key, no_load_key, temp_dict = list(), list(), dict()
    for k, v in pretrained_dict.items():
        if k.startswith('backbone.encoder.') or k.startswith('backbone.embed_remain.'):
            new_name = k[9:]
            if new_name in model_dict.keys() and np.shape(model_dict[new_name]) == np.shape(v):
                temp_dict[new_name] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    assert len(no_load_key) == 0, '給定的預訓練權重與模型不匹配'
    return model


def load_decoder_pretrained(model, pretrained_path):
    assert os.path.exists(pretrained_path), '提供的模型權重不存在'
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    if 'model_weight' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_weight']
    load_key, no_load_key, temp_dict = list(), list(), dict()
    for k, v in pretrained_dict.items():
        if k.startswith('backbone.decoder.') or k.startswith('backbone.embed_time.') or k.startswith('cls_head.'):
            new_name = k[9:]
            if new_name in model_dict.keys() and np.shape(model_dict[new_name]) == np.shape(v):
                temp_dict[new_name] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    assert len(no_load_key) == 0, '給定的預訓練權重與模型不匹配'
    return model


class RemainEatingTimeEncoder(nn.Module):
    def __init__(self, embed_dim, heads, encoder_layers, attention_norm, mlp_ratio, dropout_ratio, max_len,
                 remain_pad_val, num_remain_classes):
        super(RemainEatingTimeEncoder, self).__init__()
        self.max_len = max_len
        self.remain_pad_val = remain_pad_val
        self.embed_remain = PositionEmbedding(max_len, embed_dim, num_remain_classes)
        self.encoder = Encoder(embed_dim, heads, encoder_layers, attention_norm, mlp_ratio, dropout_ratio)

    @staticmethod
    def mask_pad(data, pad, len):
        mask = data == pad
        mask = mask.reshape(-1, 1, 1, len)
        mask = mask.expand(-1, 1, len, len)
        return mask

    def forward(self, food_remain):
        food_remain_mask = self.mask_pad(food_remain, self.remain_pad_val, self.max_len)
        food_remain = self.embed_remain(food_remain)
        food_remain = self.encoder(food_remain, food_remain_mask)
        return food_remain, food_remain_mask


class RemainEatingTimeDecoder(nn.Module):
    def __init__(self, embed_dim, heads, decoder_layers, attention_norm, mlp_ratio, dropout_ratio,
                 max_len, time_pad_val, num_time_classes):
        super(RemainEatingTimeDecoder, self).__init__()
        self.max_len = max_len
        self.time_pad_val = time_pad_val
        self.embed_time = PositionEmbedding(max_len, embed_dim, num_time_classes)
        self.decoder = Decoder(embed_dim, heads, decoder_layers, attention_norm, mlp_ratio, dropout_ratio)
        self.cls_fc = nn.Linear(embed_dim, num_time_classes)

    @staticmethod
    def mask_tril(data, pad, len):
        tril = 1 - torch.tril(torch.ones(1, len, len, dtype=torch.long))
        tril = tril.to(data.device)
        mask = data == pad
        mask = mask.unsqueeze(1).long()
        mask = mask + tril
        mask = mask > 0
        mask = (mask == 1).unsqueeze(dim=1)
        return mask

    def forward(self, food_remain, food_remain_mask, time_remain):
        y = time_remain
        mask_tril_y = self.mask_tril(y, self.time_pad_val, self.max_len)
        y = self.embed_time(y)
        y = self.decoder(food_remain, y, food_remain_mask, mask_tril_y)
        out = self.cls_fc(y)
        return out
