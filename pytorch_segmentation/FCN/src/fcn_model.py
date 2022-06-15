from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .backbone import resnet50, resnet101


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        """
        :param model: 模型
        :param return_layers: key=需要模型中的哪個層結構輸出，value=回傳時key的名稱
        """
        # 已看過
        # 如果想要的層結構不在model裡面就會報錯
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        # 深拷貝一份
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            # 當最後需要的層結構拿到後剩下的就刪除了
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # 已看過
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            # 如果有在需要的返回中，那就加入到返回當中
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FCN(nn.Module):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        # 已看過
        # 三個模塊的實例對象
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # 已看過
        # x shape [batch_size, channel, w, h] -> [batch_size, 3, 480, 480]
        # input_shape [480, 480]
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        # features = {
        #   'out': [batch_size, 2048, 60, 60]
        #   'aux': [batch_size, 1024, 60, 60]
        # }
        features = self.backbone(x)

        result = OrderedDict()
        # x shape [batch_size, 2048, 60, 60]
        x = features["out"]
        # 放入classifier中進行forward
        # x shape [batch_size, num_classes, 60, 60]
        x = self.classifier(x)
        # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
        # 使用雙線性插值方式，將特徵圖大小縮放回原圖大小
        # x shape [batch_size, num_classes, 480, 480]
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        # 如果有用輔助分類就跟上面做一樣的事情，只是在dict中的key不同
        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # 原论文中虽然使用的是ConvTranspose2d，但权重是冻结的，所以就是一个bilinear插值
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        # result = {
        #   'out': [batch_size, num_classes, 480, 480]
        #   'aux': [batch_size, num_classes, 480, 480]
        # }
        return result


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        """
        :param in_channels: 輸入channel深度
        :param channels: 分類類別數量，這裡等於輸出的channel深度
        """
        # 已看過
        # 中間層的channel深度
        inter_channels = in_channels // 4
        # 一系列層結構
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        ]

        # 最後網上丟到nn.Sequential中
        super(FCNHead, self).__init__(*layers)


def fcn_resnet50(aux, num_classes=21, pretrain_backbone=False):
    # 已看過
    # 'resnet50_imagenet': 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    # 'fcn_resnet50_coco': 'https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth'

    # 構建backbone，傳入的replace_stride_with_dilation表示在resnet中的哪幾層要使用膨脹卷積
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet50 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet50.pth", map_location='cpu'))

    # 最後輸出的channel深度以及輔助輸出的channel深度
    out_inplanes = 2048
    aux_inplanes = 1024

    # 構建回傳dict
    return_layers = {'layer4': 'out'}
    # 如果有輔助輸出就加入
    if aux:
        return_layers['layer3'] = 'aux'
    # 從backbone拿出我們要的對應輸出
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    # 是否需要輔助輸出
    if aux:
        # 輔助輸出的FCN頭
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    # 最後輸出的FCN頭
    classifier = FCNHead(out_inplanes, num_classes)

    # 全部結合就是model
    model = FCN(backbone, classifier, aux_classifier)

    return model


def fcn_resnet101(aux, num_classes=21, pretrain_backbone=False):
    # 'resnet101_imagenet': 'https://download.pytorch.org/models/resnet101-63fe2227.pth'
    # 'fcn_resnet101_coco': 'https://download.pytorch.org/models/fcn_resnet101_coco-7ecb50ca.pth'
    backbone = resnet101(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        # 载入resnet101 backbone预训练权重
        backbone.load_state_dict(torch.load("resnet101.pth", map_location='cpu'))

    out_inplanes = 2048
    aux_inplanes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers['layer3'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model
