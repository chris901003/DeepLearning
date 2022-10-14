import torch
from torch import nn
import os
from typing import Union
import numpy as np
import torch.nn.functional as F
from SpecialTopic.ST.net.layer import PatchEmbedNormal, MultiheadAttention
from SpecialTopic.ST.build import build_backbone, build_head, build_norm, build_activation, build_dropout
from SpecialTopic.ST.utils import nlc_to_nchw, nchw_to_nlc
from SpecialTopic.ST.net.basic import ConvModule


class MixFFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, act_cfg='Default', ffn_drop=0., dropout_layer=None):
        super(MixFFN, self).__init__()
        if act_cfg == 'Default':
            act_cfg = dict(type='GELU')
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation(act_cfg)
        in_channels = embed_dims
        fc1 = nn.Conv2d(in_channels=in_channels, out_channels=feedforward_channels, kernel_size=1, stride=1, bias=True)
        pe_conv = nn.Conv2d(in_channels=feedforward_channels, out_channels=feedforward_channels, kernel_size=3,
                            stride=1, padding=1, bias=True, groups=feedforward_channels)
        fc2 = nn.Conv2d(in_channels=feedforward_channels, out_channels=in_channels, kernel_size=1, stride=1, bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = build_dropout(dropout_cfg=dropout_layer) if dropout_layer is None else nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class EfficientMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dims, num_heads, attn_drop=0., proj_drop=0., dropout_layer: Union[dict, str] = 'Default',
                 batch_first=True, qkv_bias=False, norm_cfg='Default', sr_ratio=1):
        if dropout_layer == 'Default':
            dropout_layer = None
        super(EfficientMultiheadAttention, self).__init__(embed_dims, num_heads, attn_drop, proj_drop,
                                                          dropout_layer=dropout_layer, batch_first=batch_first,
                                                          bias=qkv_bias)
        if norm_cfg == 'Default':
            norm_cfg = dict(type='LN')
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(in_channels=embed_dims, out_channels=embed_dims, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = build_norm(norm_cfg, embed_dims)[1]

    def forward(self, x, hw_shape, identity=None):
        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x
        if identity is None:
            identity = x_q
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)
        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]
        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.drop_layer(self.proj_drop(out))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dims, num_heads, feedforward_channels, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 qkv_bias=True, act_cfg='Default', norm_cfg='Default', batch_first=True, sr_ratio=1):
        super(TransformerEncoderLayer, self).__init__()
        if act_cfg == 'Default':
            act_cfg = dict(type='GELU')
        if norm_cfg == 'Default':
            norm_cfg = dict(type='LN')
        self.norm1 = build_norm(norm_cfg, embed_dims)[1]
        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims, num_heads=num_heads, attn_drop=attn_drop_rate, proj_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate), batch_first=batch_first, qkv_bias=qkv_bias,
            norm_cfg=norm_cfg, sr_ratio=sr_ratio)
        self.norm2 = build_norm(norm_cfg, embed_dims)[1]
        self.ffn = MixFFN(
            embed_dims=embed_dims, feedforward_channels=feedforward_channels, ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate), act_cfg=act_cfg)

    def forward(self, x, hw_shape):
        x = self.attn(self.norm1(x), hw_shape, identity=x)
        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        return x


class MixVisionTransformer(nn.Module):
    def __init__(self, in_channels=3, embed_dims=64, num_stages=4, num_layers=(3, 4, 6, 3), num_heads=(1, 2, 4, 8),
                 patch_sizes=(7, 3, 3, 3), strides=(4, 2, 2, 2), sr_ratios=(8, 4, 2, 1),
                 out_indices=(0, 1, 2, 3), mlp_ratio=4, qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., act_cfg='Default', norm_cfg='Default'):
        super(MixVisionTransformer, self).__init__()
        if act_cfg == 'Default':
            act_cfg = dict(type='GELU')
        if norm_cfg == 'Default':
            norm_cfg = dict(type='LN', eps=1e-6)
        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        assert num_stages == len(num_layers) == len(num_heads) == len(patch_sizes) == len(strides) == len(sr_ratios)
        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))]
        cur = 0
        self.layers = nn.ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbedNormal(in_channels=in_channels, embed_dims=embed_dims_i, kernel_size=patch_sizes[i],
                                           stride=strides[i], padding=patch_sizes[i] // 2, norm_cfg=norm_cfg)
            layer = nn.ModuleList([
                TransformerEncoderLayer(embed_dims=embed_dims_i, num_heads=num_heads[i],
                                        feedforward_channels=mlp_ratio * embed_dims_i, drop_rate=drop_rate,
                                        attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[cur + idx], qkv_bias=qkv_bias,
                                        act_cfg=act_cfg, norm_cfg=norm_cfg, sr_ratio=sr_ratios[i])
                for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            norm = build_norm(norm_cfg, embed_dims_i)[1]
            block = nn.ModuleList([patch_embed, layer, norm])
            self.layers.append(block)
            cur += num_layer
        self.init_weights()

    def forward(self, x):
        outs = list()
        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def init_weights(self):
        import math
        from SpecialTopic.ST.net.weight_init import constant_init, normal_init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[
                    1] * m.out_channels
                fan_out //= m.groups
                normal_init(
                    m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)


class BaseDecodeHead(nn.Module):
    def __init__(self, in_channels, channels, *, num_classes, dropout_ratio=0.1, conv_cfg=None, norm_cfg=None,
                 act_cfg='Default', in_index: Union[int, list] = -1, input_transform=None, loss_decode='Default',
                 ignore_index=255, align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        if act_cfg == 'Default':
            act_cfg = dict(type='ReLU')
        if loss_decode == 'Default':
            loss_decode = dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def _init_inputs(self, in_channels, in_index, input_transform):
        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                F.interpolate(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs

    def forward(self, **kwargs):
        pass


class SegformerHead(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super(SegformerHead, self).__init__(input_transform='multiple_select', **kwargs)
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)
        self.convs = nn.ModuleList()
        conv_cfg = dict(type='Conv')
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(in_channels=self.in_channels[i], out_channels=self.channels, kernel_size=1, stride=1,
                           conv_cfg=conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        self.fusion_conv = ConvModule(in_channels=self.channels * num_inputs, out_channels=self.channels, kernel_size=1,
                                      conv_cfg=conv_cfg, norm_cfg=self.norm_cfg)

    def forward(self, inputs, labels, topk=(1, 5), with_loss=True):
        inputs = self._transform_inputs(inputs)
        outs = list()
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                F.interpolate(input=conv(x), size=inputs[0].shape[2:], mode=self.interpolate_mode,
                              align_corners=self.align_corners)
            )
        out = self.fusion_conv(torch.cat(outs, dim=1))
        out = self.cls_seg(out)
        topk = tuple([min(k, self.num_classes) for k in topk])
        if not with_loss:
            # 這裡輸出的圖像只會是[128x128]的圖像，如果要應用上去需要進行差值運算
            return out
        losses = dict()
        seg_logit = F.interpolate(
            input=out, size=labels.shape[2:], mode='bilinear', align_corners=self.align_corners)
        labels = labels.squeeze(1)
        loss = F.cross_entropy(seg_logit, labels, reduction='none', ignore_index=self.ignore_index).mean()
        losses['loss'] = loss
        losses['acc'] = self.accuracy(seg_logit, labels, topk=topk, ignore_index=self.ignore_index)
        return losses

    @staticmethod
    def accuracy(pred, target, topk: Union[int, tuple, list] = 1, thresh=None, ignore_index=None):
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
            return_single = True
        else:
            return_single = False
        maxk = max(topk)
        if pred.size(0) == 0:
            accu = [pred.new_tensor(0.) for _ in range(len(topk))]
            return accu[0] if return_single else accu
        assert pred.ndim == target.ndim + 1
        assert pred.size(0) == target.size(0)
        assert maxk <= pred.size(1)
        pred_value, pred_label = pred.topk(maxk, dim=1)
        # [batch_size, maxk, height, width] -> [maxk, batch_size, height, width]
        pred_label = pred_label.transpose(0, 1)
        correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
        if thresh is not None:
            correct = correct & (pred_value > thresh).t()
        if ignore_index is not None:
            correct = correct[:, target != ignore_index]
        res = list()
        eps = torch.finfo(torch.float32).eps
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
            if ignore_index is not None:
                total_num = target[target != ignore_index].numel() + eps
            else:
                total_num = target.numel() + eps
            res.append(correct_k.mul_(100 / total_num))
        return res[0] if return_single else res


class Segformer(nn.Module):
    def __init__(self, backbone, decode_head, pretrained):
        super(Segformer, self).__init__()
        if not os.path.exists(pretrained):
            print('未使用預訓練權重，全部從隨機權重開始')
            pretrained = 'none'
        self.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.decode_head = build_head(decode_head)
        if pretrained != 'none':
            self.init_weights_pretrained()

    def forward(self, images, labels=None, with_loss=True):
        feat = self.backbone(images)
        loss = self.decode_head(feat, labels, with_loss=with_loss)
        return loss

    def init_weights_pretrained(self):
        pretrained_dict = torch.load(self.pretrained, map_location='cpu')
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = self.state_dict()
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            new_layer_name = k
            if 'conv_seg' in k:
                str_idx = k.find('conv_seg')
                new_layer_name = k[:str_idx] + 'cls_seg.' + k[str_idx + 9:]
            if new_layer_name in model_dict.keys() and np.shape(model_dict[new_layer_name]) == np.shape(v):
                temp_dict[new_layer_name] = v
                load_key.append(new_layer_name)
            else:
                no_load_key.append(new_layer_name)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        model_dict.update(temp_dict)
        self.load_state_dict(model_dict)
