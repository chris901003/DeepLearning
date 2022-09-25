from .utils import get_cls_from_dict
from torch import nn


def build_detector(detector_cfg):
    from .net.yolox import YoloBody
    from .net.Recognizer3D import Recognizer3D
    support_detector = {
        'YoloBody': YoloBody,
        'Recognizer3D': Recognizer3D
    }
    detector_cls = get_cls_from_dict(support_detector, detector_cfg)
    detector = detector_cls(**detector_cfg)
    return detector


def build_backbone(backbone_cfg):
    from .net.yolox import YOLOPAFPN
    from .net.backbone import CSPDarknet, ResNet3d
    support_backbone = {
        'YOLOPAFPN': YOLOPAFPN,
        'CSPDarknet': CSPDarknet,
        'ResNet3d': ResNet3d
    }
    backbone_cls = get_cls_from_dict(support_backbone, backbone_cfg)
    backbone = backbone_cls(**backbone_cfg)
    return backbone


def build_head(head_cfg):
    from .net.yolox import YOLOXHead
    from .net.Recognizer3D import I3DHead
    support_head = {
        'YOLOXHead': YOLOXHead,
        'I3DHead': I3DHead
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
        'Sigmoid': nn.Sigmoid
    }
    act_cls = get_cls_from_dict(support_act, act_cfg_)
    act = act_cls(**act_cfg_)
    return act


def build_conv(conv_cfg, *args, **kwargs):
    conv_cfg_ = conv_cfg.copy()
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
    abbr = 'bn'
    name = abbr + str(postfix)
    requires_grad = norm_cfg_.pop('requires_grad', True)
    norm_cfg_.setdefault('eps', 1e-5)
    layer = norm_cls(num_features, **norm_cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return name, layer


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
    from .dataset.dataset import YoloDataset, VideoDataset
    support_dataset = {
        'YoloDataset': YoloDataset,
        'VideoDataset': VideoDataset
    }
    dataset_cls = get_cls_from_dict(support_dataset, dataset_cfg)
    dataset = dataset_cls(**dataset_cfg)
    return dataset
