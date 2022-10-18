import copy
from .utils import get_cls_from_dict
from torch import nn


def build_detector(detector_cfg):
    from .net.yolox import YoloBody
    from .net.Recognizer3D import Recognizer3D
    from .net.resnet import ResNet
    from .net.VIT import VIT
    from .net.MobileVit import MobileVit
    from .net.Segformer import Segformer
    from .net.RemainEatingTime import RemainEatingTime
    support_detector = {
        'YoloBody': YoloBody,
        'Recognizer3D': Recognizer3D,
        'ResNet': ResNet,
        'VIT': VIT,
        'MobileVit': MobileVit,
        'Segformer': Segformer,
        'RemainEatingTime': RemainEatingTime
    }
    detector_cls = get_cls_from_dict(support_detector, detector_cfg)
    detector = detector_cls(**detector_cfg)
    return detector


def build_backbone(backbone_cfg):
    from .net.yolox import YOLOPAFPN
    from .net.backbone import CSPDarknet, ResNet3d
    from .net.resnet import ResnetExtract
    from .net.VIT import VisionTransformer
    from .net.MobileVit import MobileVitExtract
    from .net.Segformer import MixVisionTransformer
    from .net.RemainEatingTime import RemainEatingTimeBackbone
    support_backbone = {
        'YOLOPAFPN': YOLOPAFPN,
        'CSPDarknet': CSPDarknet,
        'ResNet3d': ResNet3d,
        'ResnetExtract': ResnetExtract,
        'VisionTransformer': VisionTransformer,
        'MobileVitExtract': MobileVitExtract,
        'MixVisionTransformer': MixVisionTransformer,
        'RemainEatingTimeBackbone': RemainEatingTimeBackbone
    }
    backbone_cls = get_cls_from_dict(support_backbone, backbone_cfg)
    backbone = backbone_cls(**backbone_cfg)
    return backbone


def build_head(head_cfg):
    from .net.yolox import YOLOXHead
    from .net.Recognizer3D import I3DHead
    from .net.resnet import ResnetHead
    from .net.VIT import VitHead
    from .net.MobileVit import MobileVitHead
    from .net.Segformer import SegformerHead
    from .net.RemainEatingTime import RemainEatingTimeHead
    support_head = {
        'YOLOXHead': YOLOXHead,
        'I3DHead': I3DHead,
        'ResnetHead': ResnetHead,
        'VitHead': VitHead,
        'MobileVitHead': MobileVitHead,
        'SegformerHead': SegformerHead,
        'RemainEatingTimeHead': RemainEatingTimeHead
    }
    head_cls = get_cls_from_dict(support_head, head_cfg)
    head = head_cls(**head_cfg)
    return head


def build_neck(neck_cfg):
    support_neck = {
    }
    neck_cls = get_cls_from_dict(support_neck, neck_cfg)
    neck = neck_cls(**neck_cfg)
    return neck


def build_activation(act_cfg):
    act_cfg_ = act_cfg.copy()
    support_act = {
        'ReLU': nn.ReLU,
        'LeakyReLU': nn.LeakyReLU,
        'SELU': nn.SELU,
        'Sigmoid': nn.Sigmoid,
        'GELU': nn.GELU,
        'SiLU': nn.SiLU
    }
    act_cls = get_cls_from_dict(support_act, act_cfg_)
    act = act_cls(**act_cfg_)
    return act


def build_conv(conv_cfg, *args, **kwargs):
    conv_cfg_ = copy.deepcopy(conv_cfg)
    support_conv = {
        'Conv1d': nn.Conv1d,
        'Conv2d': nn.Conv2d,
        'Conv3d': nn.Conv3d,
        'Conv': nn.Conv2d
    }
    conv_cls = get_cls_from_dict(support_conv, conv_cfg_)
    conv = conv_cls(*args, **kwargs)
    return conv


def build_norm(norm_cfg, num_features, postfix=''):
    norm_cfg_ = norm_cfg.copy()
    support_norm = {
        'BN': nn.BatchNorm2d,
        'BN1d': nn.BatchNorm1d,
        'BN2d': nn.BatchNorm2d,
        'BN3d': nn.BatchNorm3d,
        'LN': nn.LayerNorm
    }
    norm_cls = get_cls_from_dict(support_norm, norm_cfg_)
    abbr = 'bn' if 'BN' in norm_cfg['type'] else 'ln'
    name = abbr + str(postfix)
    requires_grad = norm_cfg_.pop('requires_grad', True)
    norm_cfg_.setdefault('eps', 1e-5)
    layer = norm_cls(num_features, **norm_cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return name, layer


def build_dropout(dropout_cfg):
    from SpecialTopic.ST.net.layer import DropPath
    dropout_cfg_ = copy.deepcopy(dropout_cfg)
    support_dropout_layer = {
        'Dropout': nn.Dropout,
        'DropPath': DropPath
    }
    dropout_cls = get_cls_from_dict(support_dropout_layer, dropout_cfg_)
    dropout = dropout_cls(**dropout_cfg_)
    return dropout


def build_loss(loss_cfg):
    from .net.yolox import YOLOLoss
    support_loss = {
        'YOLOLoss': YOLOLoss,
        'CrossEntropyLoss': nn.CrossEntropyLoss
    }
    loss_cls = get_cls_from_dict(support_loss, loss_cfg)
    loss = loss_cls(**loss_cfg)
    return loss


def build_dataset(dataset_cfg):
    from .dataset.dataset import YoloDataset, VideoDataset, RemainingDataset, SegformerDataset
    support_dataset = {
        'YoloDataset': YoloDataset,
        'VideoDataset': VideoDataset,
        'RemainingDataset': RemainingDataset,
        'SegformerDataset': SegformerDataset
    }
    dataset_cfg_ = copy.deepcopy(dataset_cfg)
    dataset_cls = get_cls_from_dict(support_dataset, dataset_cfg_)
    dataset = dataset_cls(**dataset_cfg_)
    return dataset
