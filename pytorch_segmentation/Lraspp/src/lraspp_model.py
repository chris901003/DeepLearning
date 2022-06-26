from collections import OrderedDict

from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .mobilenet_backbone import mobilenet_v3_large


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a models

    It has a strong assumption that the modules have been registered
    into the models in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the models. So if `models` is passed, `models.feature1` can
    be returned, but not `models.feature1.layer2`.

    Args:
        model (nn.Module): models on which we will extract the features
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
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in models")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class LRASPP(nn.Module):
    """
    Implements a Lite R-ASPP Network for semantic segmentation from
    `"Searching for MobileNetV3"
    <https://arxiv.org/abs/1905.02244>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the models.
            The backbone should return an OrderedDict[Tensor], with the key being
            "high" for the high level feature map and "low" for the low level feature map.
        low_channels (int): the number of channels of the low level features.
        high_channels (int): the number of channels of the high level features.
        num_classes (int): number of output classes of the models (including the background).
        inter_channels (int, optional): the number of channels for intermediate computations.
    """
    __constants__ = ['aux_classifier']

    def __init__(self,
                 backbone: nn.Module,
                 low_channels: int,
                 high_channels: int,
                 num_classes: int,
                 inter_channels: int = 128) -> None:
        # 構建完整模型
        super(LRASPP, self).__init__()
        self.backbone = backbone
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        out = self.classifier(features)
        # 做雙線性插值回到原圖大小
        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)

        result = OrderedDict()
        result["out"] = out

        return result


class LRASPPHead(nn.Module):
    def __init__(self,
                 low_channels: int,
                 high_channels: int,
                 num_classes: int,
                 inter_channels: int) -> None:
        """
        :param low_channels: 較低層的channel深度
        :param high_channels: 最後一層的channel深度
        :param num_classes: 分類類別數
        :param inter_channels: 中間層channel深度
        """
        super(LRASPPHead, self).__init__()
        # 這裡可以去對照圖，看是哪幾個層結構
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, inputs: Dict[str, Tensor]) -> Tensor:
        # inputs是從backbone出來的結果，是一個dict的格式
        low = inputs["low"]
        high = inputs["high"]

        # 最後一層輸出的處理
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)


def lraspp_mobilenetv3_large(num_classes=21, pretrain_backbone=False):
    # 'mobilenetv3_large_imagenet': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'
    # 'lraspp_mobilenet_v3_large_coco': 'https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth'
    # 構建骨幹，這裡是用mobilenet作為骨幹
    backbone = mobilenet_v3_large(dilated=True)

    if pretrain_backbone:
        # 载入mobilenetv3 large backbone预训练权重
        backbone.load_state_dict(torch.load("mobilenet_v3_large.pth", map_location='cpu'))

    # 只拿出特徵提取的部分，後面分類就不要了
    backbone = backbone.features

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    # 拿出第一層以及stride=2還有最後一層
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "is_strided", False)] + [len(backbone) - 1]
    # 拿到我們要的從backbone輸出的兩層
    low_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    high_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    # 獲取輸出channel的深度
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels

    return_layers = {str(low_pos): "low", str(high_pos): "high"}
    # 拔除不需要的層結構
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = LRASPP(backbone, low_channels, high_channels, num_classes)
    return model
