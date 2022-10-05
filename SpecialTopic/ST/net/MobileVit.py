import copy
import torch
from torch import nn
import numpy as np
from SpecialTopic.ST.build import build_backbone, build_head
from ..model_config.MobileVitConfig import MobileVit_config
from SpecialTopic.ST.net.basic import ConvModule
from SpecialTopic.ST.net.layer import InvertedResidual, MobileVitBlock


class MobileVitExtract(nn.Module):
    def __init__(self, layer1, layer2, layer3, layer4, layer5, last_layer_exp_factor, conv_cfg='Default',
                 norm_cfg='Default', act_cfg='Default'):
        super(MobileVitExtract, self).__init__()
        if conv_cfg == 'Default':
            conv_cfg = dict(type='Conv')
        if norm_cfg == 'Default':
            norm_cfg = dict(type='BN')
        if act_cfg == 'Default':
            act_cfg = dict(type='SiLU')
        image_channels = 3
        out_channels = 16
        self.conv_1 = ConvModule(in_channels=image_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                 padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.layer_1, out_channels = self._make_layer(input_channels=out_channels, cfg=layer1)
        self.layer_2, out_channels = self._make_layer(input_channels=out_channels, cfg=layer2)
        self.layer_3, out_channels = self._make_layer(input_channels=out_channels, cfg=layer3)
        self.layer_4, out_channels = self._make_layer(input_channels=out_channels, cfg=layer4)
        self.layer_5, out_channels = self._make_layer(input_channels=out_channels, cfg=layer5)
        exp_channels = min(last_layer_exp_factor * out_channels, 960)
        self.exp_channels = exp_channels
        self.conv_1x1_exp = ConvModule(in_channels=out_channels, out_channels=exp_channels, kernel_size=1,
                                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.layer_5(self.layer_4(self.layer_3(self.layer_2(self.layer_1(x)))))
        x = self.conv_1x1_exp(x)
        return x

    def _make_layer(self, input_channels, cfg):
        block_type = cfg.get('block_type', None)
        assert block_type is not None, '需要指定block type [mv2, mobilevit]'
        if block_type.lower() == 'mobilevit':
            return self._make_mit_layer(input_channels=input_channels, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channels=input_channels, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channels, cfg):
        out_channels = cfg.get('out_channels', None)
        assert out_channels is not None
        num_blocks = cfg.get('num_blocks', 2)
        expand_ratio = cfg.get('expand_ratio', 4)
        block = list()
        for i in range(num_blocks):
            stride = cfg.get('stride', 1) if i == 0 else 1
            layer = InvertedResidual(in_channels=input_channels, out_channels=out_channels, stride=stride,
                                     expand_ratio=expand_ratio)
            block.append(layer)
            input_channels = out_channels
        return nn.Sequential(*block), input_channels

    @staticmethod
    def _make_mit_layer(input_channels, cfg):
        stride = cfg.get('stride', 1)
        block = list()
        if stride == 2:
            out_channels = cfg.get('out_channels', None)
            assert out_channels is not None
            mv_expand_ratio = cfg.get('mv_expand_ratio', None)
            assert mv_expand_ratio is not None
            layer = InvertedResidual(in_channels=input_channels, out_channels=out_channels, stride=stride,
                                     expand_ratio=mv_expand_ratio)
            block.append(layer)
            input_channels = out_channels
        transformer_dim = cfg.get('transformer_channels', None)
        assert transformer_dim is not None
        ffn_dim = cfg.get('ffn_dim', None)
        assert ffn_dim is not None
        num_heads = cfg.get('num_heads', 4)
        head_dim = transformer_dim // num_heads
        if transformer_dim % head_dim != 0:
            raise ValueError('設定的多頭數與channel深度不匹配')
        transformer_blocks = cfg.get('transformer_blocks', 1)
        block.append(MobileVitBlock(
            in_channels=input_channels, transformer_dim=transformer_dim, ffn_dim=ffn_dim,
            transformer_blocks=transformer_blocks, patch_h=cfg.get('patch_h', 2), patch_w=cfg.get('patch_w', 2),
            dropout=cfg.get('dropout', 0.1), ffn_dropout=cfg.get('ffn_dropout', 0.0),
            attn_dropout=cfg.get('attn_dropout', 0.1), head_dim=head_dim, conv_ksize=3))
        return nn.Sequential(*block), input_channels


class MobileVitHead(nn.Module):
    def __init__(self, cls_dropout, num_classes, exp_channels):
        super(MobileVitHead, self).__init__()
        self.cls_dropout = cls_dropout
        self.num_classes = num_classes
        self.classifier = nn.Sequential()
        self.classifier.add_module(name='global_pool', module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name='flatten', module=nn.Flatten())
        if 0.0 < cls_dropout < 1.0:
            self.classifier.add_module(name='dropout', module=nn.Dropout(p=cls_dropout))
        self.classifier.add_module(name='fc', module=nn.Linear(exp_channels, num_classes))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels=None, with_loss=True):
        x = self.classifier(x)
        if not with_loss:
            return x
        assert labels is not None
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


class MobileVit(nn.Module):
    def __init__(self, phi, num_classes, with_cls_head=True, pretrained='none'):
        super(MobileVit, self).__init__()
        self.phi = phi
        self.num_classes = num_classes
        self.with_cls_head = with_cls_head
        self.pretrained = pretrained
        assert phi in MobileVit_config.keys(), '目前只支援[s, m, l]'
        model_cfg = MobileVit_config[phi]
        model_cfg = copy.deepcopy(model_cfg)
        for k in ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
            model_cfg['backbone'][k].update({"dropout": 0.1, "ffn_dropout": 0.0, "attn_dropout": 0.0})
        model_cfg['cls_head']['num_classes'] = num_classes
        self.backbone = build_backbone(model_cfg['backbone'])
        if with_cls_head:
            model_cfg['cls_head']['exp_channels'] = self.backbone.exp_channels
            self.cls_head = build_head(model_cfg['cls_head'])
        if pretrained != 'none':
            self.init_weight_pretrained()
        else:
            self.apply(self.init_weight)

    def forward(self, x, labels, with_loss=True):
        output = self.backbone(x)
        if self.with_cls_head:
            loss = self.cls_head(output, labels, with_loss)
        else:
            return output
        return loss

    def init_weight_pretrained(self):
        pretrained_dict = torch.load(self.pretrained, map_location='cpu')
        model_dict = self.state_dict()
        load_key, no_load_key, temp_dict = list(), list(), dict()
        for k, v in pretrained_dict.items():
            new_layer_name = None
            if 'fc' in k:
                no_load_key.append(k)
                continue
            elif 'block.conv' in k and 'block.norm' not in k:
                str_idx = k.rfind('block.conv')
                cur = k
                if str_idx != -1:
                    cur = cur[:str_idx] + cur[str_idx + 6:]
                new_layer_name = 'backbone.' + cur
            elif 'block.norm' in k:
                str_idx = k.rfind('block.norm')
                cur = k
                if str_idx != -1:
                    cur = cur[:str_idx] + 'bn.' + cur[str_idx + 11:]
                new_layer_name = 'backbone.' + cur
            elif 'qkv_proj' in k:
                str_idx = k.rfind('qkv_proj')
                cur = k
                if str_idx != -1:
                    cur = cur[:str_idx] + 'qkv.' + cur[str_idx + 9:]
                new_layer_name = 'backbone.' + cur
            elif 'out_proj' in k:
                str_idx = k.rfind('out_proj')
                cur = k
                if str_idx != -1:
                    cur = cur[:str_idx] + 'proj.' + cur[str_idx + 9:]
                new_layer_name = 'backbone.' + cur
            elif 'pre_norm_ffn' in k and 'pre_norm_ffn.0' not in k:
                str_idx = k.rfind('pre_norm_ffn')
                cur = k
                if str_idx != -1:
                    cur = cur[:str_idx + 13] + '1.'
                    if k[str_idx + 13] == '1':
                        cur += 'fc1.'
                    else:
                        cur += 'fc2.'
                    cur += k[str_idx + 15:]
                new_layer_name = 'backbone.' + cur
            else:
                new_layer_name = 'backbone.' + k
            if new_layer_name is not None:
                if new_layer_name in model_dict.keys() and np.shape(model_dict[new_layer_name]) == np.shape(v):
                    temp_dict[new_layer_name] = v
                    load_key.append(new_layer_name)
                else:
                    no_load_key.append(new_layer_name)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        model_dict.update(temp_dict)
        t1 = set(model_dict.keys())
        t2 = set(temp_dict.keys())
        print('Not in pretrained but in current model: ')
        print(t1 - t2)
        self.load_state_dict(model_dict)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass
