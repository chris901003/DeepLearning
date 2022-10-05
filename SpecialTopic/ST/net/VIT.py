import copy
import torch
from torch import nn
import numpy as np
from SpecialTopic.ST.build import build_backbone, build_head, build_norm
from SpecialTopic.ST.net.layer import PatchEmbed, VitBlock
from SpecialTopic.ST.net.weight_init import _init_vit_weights


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, representation_size=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed,
                 norm_layer='Default', act_layer='Default'):
        super(VisionTransformer, self).__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        if norm_layer == 'Default':
            norm_layer = dict(type='LN', eps=1e-6)
        if act_layer == 'Default':
            act_layer = dict(type='GELU')
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[
            VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                     drop_path_ratio=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = build_norm(norm_layer, embed_dim)[1]
        self.init_weights()

    def init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]


class VitHead(nn.Module):
    def __init__(self, num_classes, num_features):
        super(VitHead, self).__init__()
        self.num_classes = num_classes
        self.head = nn.Linear(num_features, num_classes)
        # 這個BN是沒有何作用的，只是為了配合優化器才加上去
        self.bn = nn.BatchNorm2d(num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels, with_loss=True):
        x = self.head(x)
        if not with_loss:
            return x
        assert labels is not None, '需要計算loss須提供labels'
        if labels.ndim == 2:
            labels = labels.squeeze(dim=-1)
        loss_dict = dict()
        losses = self.loss(x, labels)
        loss_dict['loss'] = losses
        preds = x.argmax(dim=1)
        acc = torch.eq(preds, labels).sum() / labels.size(0)
        loss_dict['acc'] = acc
        predict_score, predict_idx = torch.topk(x, k=min(self.num_classes, 3), dim=1)
        labels = labels.view(-1, 1)
        topk = (labels == predict_idx).sum()
        topk_acc = topk / labels.size(0)
        loss_dict['topk_acc'] = topk_acc
        return loss_dict


class VIT(nn.Module):
    build_format = {
        'm': {
            'backbone': {'type': 'VisionTransformer', 'img_size': 224, 'patch_size': 16, 'embed_dim': 768, 'depth': 12,
                         'num_heads': 12, 'representation_size': None},
            'cls_head': {'type': 'VitHead', 'num_features': 768}
        },
        'l': {
            'backbone': {'type': 'VisionTransformer', 'img_size': 224, 'patch_size': 16, 'embed_dim': 1024, 'depth': 24,
                         'num_heads': 16, 'representation_size': None},
            'cls_head': {'type': 'VitHead', 'num_features': 1024}
        }
    }

    def __init__(self, phi, num_classes, with_cls_head=True, pretrained='none'):
        super(VIT, self).__init__()
        self.phi = phi
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.with_cls_head = with_cls_head
        assert phi in self.build_format, '目前VIT有提供m以及l兩種大小的模型，歡迎添加'
        model_cfg = self.build_format.get(phi, None)
        model_cfg = copy.deepcopy(model_cfg)
        model_cfg['cls_head']['num_classes'] = num_classes
        self.backbone = build_backbone(model_cfg['backbone'])
        if with_cls_head:
            self.cls_head = build_head(model_cfg['cls_head'])
        if pretrained != 'none':
            self.init_weight_pretrained()
        else:
            self.apply(_init_vit_weights)

    def forward(self, images, labels=None, with_loss=True):
        output = self.backbone(images)
        if self.with_cls_head:
            result = self.cls_head(output, labels, with_loss)
        else:
            result = output
        return result

    def init_weight_pretrained(self):
        pretrained_dict = torch.load(self.pretrained, map_location='cpu')
        model_dict = self.state_dict()
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            new_layer_name = 'backbone.' + k
            if new_layer_name in model_dict.keys() and np.shape(model_dict[new_layer_name]) == np.shape(v):
                temp_dict[new_layer_name] = v
                load_key.append(new_layer_name)
            else:
                no_load_key.append(new_layer_name)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        model_dict.update(temp_dict)
        self.load_state_dict(model_dict)
