import torch
from torch import nn
import numpy as np
from SpecialTopic.ST.net.layer import BasicBlockResnet, BottleneckResnet
from SpecialTopic.ST.net.basic import ConvModule
from SpecialTopic.ST.build import build_backbone, build_head


class ResnetExtract(nn.Module):
    def __init__(self, block, blocks_num, groups=1, width_per_group=64):
        super(ResnetExtract, self).__init__()
        self.in_channel = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.stem = ConvModule(3, self.in_channel, kernel_size=7, stride=2, padding=2,
                               conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.init_weight()

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = ConvModule(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride,
                                    bias=False, conv_cfg=dict(type='Conv'), norm_cfg=dict(type='BN'), act_cfg=None)
        layers = list()
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride,
                            groups=self.groups, width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, imgs):
        output = self.stem(imgs)
        output = self.maxpool(output)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return output

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class ResnetHead(nn.Module):
    def __init__(self, num_classes, block):
        super(ResnetHead, self).__init__()
        self.num_classes = num_classes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels, with_loss=True):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if not with_loss:
            return x
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


class ResNet(nn.Module):
    # s => resnet34, m => resnet50, l => resnet101
    build_format = {
        's': {
            'backbone': {'type': 'ResnetExtract', 'block': BasicBlockResnet, 'blocks_num': [3, 4, 6, 3],
                         'groups': 1, 'width_per_group': 64},
            'cls_head': {'type': 'ResnetHead', 'num_classes': 1000, 'block': BasicBlockResnet}
        },
        'm': {
            'backbone': {'type': 'ResnetExtract', 'block': BottleneckResnet, 'blocks_num': [3, 4, 6, 3],
                         'groups': 1, 'width_per_group': 64},
            'cls_head': {'type': 'ResnetHead', 'num_classes': 1000, 'block': BottleneckResnet}
        },
        'l': {
            'backbone': {'type': 'ResnetExtract', 'block': BottleneckResnet, 'blocks_num': [3, 4, 23, 3],
                         'groups': 1, 'width_per_group': 64},
            'cls_head': {'type': 'ResnetHead', 'num_classes': 1000, 'block': BottleneckResnet}
        }
    }

    def __init__(self, phi, num_classes, with_cls_head=True, pretrained='none', group=-1, width_per_group=-1):
        super(ResNet, self).__init__()
        self.with_cls_head = with_cls_head
        self.num_classes = num_classes
        assert phi in self.build_format.keys(), '目前resnet提供[s, m, l]三種尺寸分別對應上[resnet34, resnet50, resnet101]'
        model_cfg = self.build_format[phi]
        model_cfg['cls_head']['num_classes'] = num_classes
        if group != -1:
            model_cfg['backbone']['group'] = group
        if width_per_group != -1:
            model_cfg['backbone']['width_per_group'] = width_per_group
        self.backbone = build_backbone(model_cfg['backbone'])
        if with_cls_head:
            self.cls_head = build_head(model_cfg['cls_head'])
        if pretrained != 'none':
            self.pretrained = pretrained
            self.init_weights()

    def forward(self, imgs, labels, with_loss=True):
        output = self.backbone(imgs)
        if self.with_cls_head:
            results = self.cls_head(output, labels, with_loss)
        else:
            results = output
        return results

    def init_weights(self):
        pretrained_dict = torch.load(self.pretrained, map_location='cpu')
        model_dict = self.state_dict()
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            new_layer_name = None
            if 'fc' in k:
                no_load_key.append(k)
                continue
            if 'layer' not in k:
                tmp = k.split('.')
                tmp[0] = tmp[0][:-1]
                new_layer_name = 'backbone.stem.' + '.'.join(tmp)
            elif 'conv' in k:
                str_idx = k.find('conv')
                layer_idx = k[str_idx + 4]
                end_name = k[str_idx + 6:]
                new_layer_name = 'backbone.' + k[:str_idx] + f'layer{layer_idx}.conv.' + end_name
            elif 'bn' in k:
                str_idx = k.find('bn')
                layer_idx = k[str_idx + 2]
                end_name = k[str_idx + 4:]
                new_layer_name = 'backbone.' + k[:str_idx] + f'layer{layer_idx}.bn.' + end_name
            elif 'downsample' in k:
                tmp = k.split('.')
                if tmp[-2] == '0':
                    tmp[-2] = 'conv'
                elif tmp[-2] == '1':
                    tmp[-2] = 'bn'
                else:
                    raise ValueError('權重有誤')
                new_layer_name = 'backbone.' + '.'.join(tmp)
            if new_layer_name is not None:
                if new_layer_name in model_dict.keys() and np.shape(model_dict[new_layer_name]) == np.shape(v):
                    temp_dict[new_layer_name] = v
                    load_key.append(new_layer_name)
                else:
                    no_load_key.append(new_layer_name)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        model_dict.update(temp_dict)
        self.load_state_dict(model_dict)
