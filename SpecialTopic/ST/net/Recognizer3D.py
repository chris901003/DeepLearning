from torch import nn
import torch
from ..build import build_backbone, build_head, build_neck, build_loss
from .weight_init import normal_init


class I3DHead(nn.Module):
    def __init__(self, num_classes, in_channels, loss_cls='Default', spatial_type='avg',
                 dropout_ratio=0.5, init_std=0.01):
        super(I3DHead, self).__init__()
        if loss_cls == 'Default':
            loss_cls = dict(type='CrossEntropyLoss')
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        if self.spatial_type == 'avg':
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        cls_score = self.fc_cls(x)
        return cls_score

    def loss(self, cls_score, labels):
        labels = labels.to(torch.long)
        if labels.ndim == 2:
            labels = labels.argmax(dim=1)
        elif labels.ndim > 3:
            raise ValueError('給定的標籤型態有誤')
        loss_dict = dict()
        losses = self.loss_cls(cls_score, labels)
        loss_dict['loss'] = losses
        predict = cls_score.argmax(dim=1)
        acc = torch.eq(predict, labels).sum() / labels.size(0)
        loss_dict['acc'] = acc
        return loss_dict


class Recognizer3D(nn.Module):
    def __init__(self, backbone, cls_head=None, neck=None, train_cfg=None, test_cfg=None):
        super(Recognizer3D, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        self.cls_head = build_head(cls_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if self.cls_head is not None:
            self.cls_head.init_weights()
        if self.neck is not None:
            self.neck.init_weights()

    def extract_feat(self, imgs):
        x = self.backbone(imgs)
        return x

    def forward(self, imgs, label=None, return_loss=True):
        if return_loss:
            if label is None:
                raise ValueError('續練時需要提供label資訊')
            return self.forward_train(imgs, label)
        return self.forward_test(imgs)

    def forward_train(self, imgs, labels):
        assert self.cls_head is not None
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()
        x = self.extract_feat(imgs)
        if self.neck:
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)
        cls_score = self.cls_head(x)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels)
        losses.update(loss_cls)
        return losses

    def forward_test(self, imgs):
        pass
