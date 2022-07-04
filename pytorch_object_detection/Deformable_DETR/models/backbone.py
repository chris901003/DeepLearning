# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        """
        :param backbone: 特徵提取backbone
        :param train_backbone: 是否需要訓練backbone
        :param return_interm_layers: 是否需要提取中間層的backbone
        """
        super().__init__()
        for name, parameter in backbone.named_parameters():
            # 將backbone中的層結構進行訓練凍結
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # 如果需要回傳中間層結構就會在這裡決定哪些結構需要回傳
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            # 每一個層結構對應上的下採樣倍率
            self.strides = [8, 16, 32]
            # 每一個層結構的輸出channel深度
            self.num_channels = [512, 1024, 2048]
        else:
            # 回傳最後一層層結構輸出
            return_layers = {'layer4': "0"}
            # 最後一層對應原圖的下採樣倍率
            self.strides = [32]
            # 最後一層的channel深度
            self.num_channels = [2048]
        # 將backbone不需要的部分拿掉，同時稍微重構一下backbone，讓backbone輸出的是一個dict，key會是[0, 1, 2]，value就會是tensor
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        # 圖像從這裡開始向前傳播
        # 取出NestedTensor當中圖像tensors的部分
        # 不是訓練segmentation就只會有一個輸出
        # xs = {'0': tensor (batch_size, channel, height, width)}
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            # 取出每個圖像的mask部分
            m = tensor_list.mask
            assert m is not None
            # 將mask大小調整到跟特徵圖一樣大，這裡因為雙線性差值輸入格式我們需要先擴維之後再把擴維的部分拿掉
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # 輸出就會是一個處理好的NestedTensor
            out[name] = NestedTensor(x, mask)
        # 最後輸出出去，會是一個dict且key為'0'同時value為NestedTensor格式
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        """
        :param name: 選擇backbone
        :param train_backbone: backbone是否需要進行訓練
        :param return_interm_layers: 是否需要將backbone的中間層結構輸出出去
        :param dilation: 是否使用膨脹卷積
        """
        # 標準化層使用FrozenBatchNorm2d
        norm_layer = FrozenBatchNorm2d
        # 獲取pytorch官方實現的resnet作為backbone同時也會設定膨脹卷積的使用
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        # 將資料往父類傳遞進行初始化
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        # 將backbone以及position_embedding傳到Sequential當中
        super().__init__(backbone, position_embedding)
        # 將一些資料進行賦值
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        # 先放到backbone進行特徵提取
        # xs = dict{'0': NestedTensor}
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        # 將xs依照key進行排序放到out裏面
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        # 進行位置編碼
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        # 回傳backbone輸出，以及每一個輸出對應上去的位置編碼
        return out, pos


def build_backbone(args):
    # 實例化位置編碼
    position_embedding = build_position_encoding(args)
    # 確認backbone是否需要被訓練
    train_backbone = args.lr_backbone > 0
    # 是否需要將backbone中的中間層結構輸出保留下來，在segmentation當中會需要用到
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    # 構建backbone
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # 構建完整的backbone
    model = Joiner(backbone, position_embedding)
    # 最後回傳模型
    return model
