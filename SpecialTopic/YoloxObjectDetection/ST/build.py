from .utils import get_cls_from_dict


def build_detector(detector_cfg):
    from .net.yolox import YoloBody
    support_detector = {
        'YoloBody': YoloBody
    }
    detector_cls = get_cls_from_dict(support_detector, detector_cfg)
    detector = detector_cls(**detector_cfg)
    return detector


def build_backbone(backbone_cfg):
    from .net.yolox import YOLOPAFPN
    from .net.backbone import CSPDarknet
    support_backbone = {
        'YOLOPAFPN': YOLOPAFPN,
        'CSPDarknet': CSPDarknet
    }
    backbone_cls = get_cls_from_dict(support_backbone, backbone_cfg)
    backbone = backbone_cls(**backbone_cfg)
    return backbone


def build_head(head_cfg):
    from .net.yolox import YOLOXHead
    support_head = {
        'YOLOXHead': YOLOXHead
    }
    head_cls = get_cls_from_dict(support_head, head_cfg)
    head = head_cls(**head_cfg)
    return head


def build_loss(loss_cfg):
    from .net.yolox import YOLOLoss
    support_loss = {
        'YOLOLoss': YOLOLoss
    }
    loss_cls = get_cls_from_dict(support_loss, loss_cfg)
    loss = loss_cls(**loss_cfg)
    return loss


def build_dataset(dataset_cfg):
    from .dataset.dataset import YoloDataset
    support_dataset = {
        'YoloDataset': YoloDataset
    }
    dataset_cls = get_cls_from_dict(support_dataset, dataset_cfg)
    dataset = dataset_cls(**dataset_cfg)
    return dataset
