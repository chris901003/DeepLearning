# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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

from obj_detection.util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    # 與正常的BatchNorm一樣只是在訓練的時候不會更新，還是會把一個batch的均值調整成0方差為1
    # 已看過
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

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
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        """
        :param backbone: resnet50
        :param train_backbone: 是否要進行訓練
        :param num_channels: 最後一層輸出的channel
        :param return_interm_layers: 訓練第二階段segmentation會是True
        """
        # 已看過
        super().__init__()
        # 看要凍結哪部分的網路
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # 看要不要把中間的layer也輸出出去，訓練第二階段segmentation時會是True
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        # IntermediateLayerGetter作用是可以把backbone中layer名稱與return_layers的key對上的輸出拿出來
        # 同時body會是一個dict，key為return_layers的value，value就是輸出值
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        # 已看過
        # NestedTensor詳細內容到detr.py中找或是到misc.py都可以
        # tensor_list中的tensors就是圖片轉成的tensor shape [batch_size, 3, height, width]
        # xs = dict格式，裡面有return_layers指定的輸出layer的特徵圖
        xs = self.body(tensor_list.tensors)
        # 最後要出書的東西
        out: Dict[str, NestedTensor] = {}
        # name = key, x = value
        for name, x in xs.items():
            # 拿到我們的mask shape [batch_size, height, width]
            m = tensor_list.mask
            assert m is not None
            # 用pytorch的interpolate來將mask的高寬調整到與輸出特徵圖一樣
            # 原始的mask一定會比特徵圖大，所以這裡一定是對mask做縮小，interpolate再縮小的時候丟棄右邊以及下面的數據
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # 加入out
            out[name] = NestedTensor(x, mask)
        # 如果是Object Detection的話out會是 out = {'0': NestedTensor}
        # 也就是只有最後一個輸出層會輸出

        # 如果是Segmentation的話out會是 out = {'0': NestedTensor, '1': NestedTensor, '2': NestedTensor, '3': NestedTensor}
        # 也就是會有多層輸出
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # 已看過
        # return_interm_layers在第二階段訓練segmentation時會是True
        # dilation一直都會是False，在官方文檔中沒有看到會是True的時候
        # ----------------------------------------------------------------------------
        # replace_stride_with_dilation = 可以決定在resnet中哪個layer要不要用膨脹卷積
        # pretrained = 可以決定要不要載入預訓練權重
        # norm_layer = 可以決定自己要用什麼NormLayer原先預設是BatchNorm
        # ----------------------------------------------------------------------------
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        # 使用不同深度的resnet在最後輸出的時候會有不同的channel
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        # 已看過
        # 繼承nn.Sequential後在forward函數中可以用self[0]來呼叫backbone的forward函數
        # 也可以用self[1]來呼叫position_embedding的forward函數
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        # 已看過
        # NestedTensor詳細內容到detr.py中找或是到misc.py都可以
        # ----------------------------------------------------------------------------
        # self[0]是backbone的forward調用
        # 如果是Object Detection的話xs會是 xs = {'0': NestedTensor}
        # 也就是只有最後一個輸出層會輸出
        # 如果是Segmentation的話xs會是 xs = {'0': NestedTensor, '1': NestedTensor, '2': NestedTensor, '3': NestedTensor}
        # 也就是會有多層輸出
        # ----------------------------------------------------------------------------
        # 首先先傳入backbone的forward
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            # 將NestedTensor加入out
            out.append(x)
            # position encoding
            # 做進入transformer前的position encoding
            # self[1]是position_embedding的forward
            # 根據x中的mask會製造出一個位置編碼，高寬會跟特徵圖一樣
            # pos shape [batch_size, channel, height, weight]
            pos.append(self[1](x).to(x.tensors.dtype))

        # out (List[NestedTensor])，list長度就是從backbone拿出多少層的輸出
        # pos (List[tensor]) tensor shape [batch_size, channel, height, width]，list長度就是從backbone拿出多少層的輸出
        return out, pos


def build_backbone(args):
    # 已看過
    position_embedding = build_position_encoding(args)
    # lr_backbone應該是backbone的學習率
    train_backbone = args.lr_backbone > 0
    # 在訓練第二階段segmentation時會是True
    return_interm_layers = args.masks
    # ----------------------------------------------------------------------------
    # backbone = 選擇要用哪個backbone預設為resnet50
    # train_backbone = backbone的學習率是否大於0，估計是表示backbone要不要學習
    # return_interm_layers = 如果要做segmentation會設定成True，會將resnet的中間layer都輸出出來
    # dilation = 在resnet的最後一層不會進行下採樣，會使用空洞卷積做替代，也就是不會減少高和寬但是可以增加感受野
    # dilation估計會配合return_interm_layers一起使用
    # ----------------------------------------------------------------------------
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    # 在model中添加一個變數num_channels來知道最後輸出的channel
    model.num_channels = backbone.num_channels
    return model
