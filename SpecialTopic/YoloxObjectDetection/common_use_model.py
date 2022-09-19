import warnings
import torch
from torch import nn
import torch.nn.functional as F
from utils import get_specified_option
from weight_init import kaiming_init, constant_init


def build_conv_layer(cfg, *args, **kwargs) -> nn.Module:
    support_conv_layer = {
        'Conv1d': nn.Conv1d,
        'Conv2d': nn.Conv2d,
        'Conv3d': nn.Conv3d,
        'Conv': nn.Conv2d
    }
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        cfg_ = cfg.copy()
    conv_cls = get_specified_option(support_conv_layer, cfg_)
    conv = conv_cls(*args, **kwargs)
    return conv


def build_norm_layer(cfg, num_features, postfix=''):
    support_norm_layer = {
        'BN': nn.BatchNorm2d,
        'BN1d': nn.BatchNorm1d,
        'BN2d': nn.BatchNorm2d,
        'BN3d': nn.BatchNorm3d,
        'GN': nn.GroupNorm,
        'LN': nn.LayerNorm,
        'IN': nn.InstanceNorm2d,
        'IN1d': nn.InstanceNorm1d,
        'IN2d': nn.InstanceNorm2d,
        'IN3d': nn.InstanceNorm3d
    }
    assert isinstance(cfg, dict), '構建標準化層時，傳入的cfg資料不是dict格式'
    cfg_ = cfg.copy()
    norm_type = cfg_.get('type')
    norm_cls = get_specified_option(support_norm_layer, cfg_)
    name = f'NormLayer_{postfix}' if postfix != '' else 'NormLayer'
    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if norm_type != 'GN':
        norm = norm_cls(num_features, **cfg_)
    else:
        norm = norm_cls(num_channels=num_features, **cfg_)
    for param in norm.parameters():
        param.requires_grad = requires_grad
    return name, norm


def build_activation_layer(cfg):
    support_act_layer = {
        'ReLU': nn.ReLU,
        'Swish': Swish,
        'Sigmoid': nn.Sigmoid,
        'SiLU': nn.SiLU
    }
    act_cls = get_specified_option(support_act_layer, cfg)
    act = act_cls(**cfg)
    return act


def build_optimizer(model, cfg):
    support_optimizer = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam
    }
    optimizer_cls = get_specified_option(support_optimizer, cfg)
    optimizer = optimizer_cls(model.parameters(), **cfg)
    return optimizer


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        return x.float().clamp(min, max).half()
    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols
    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))
    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
    enclosed_rb = []
    enclosed_lt = []
    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]
        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]
        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]
        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:])
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class ConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg='Default',
                 inplace: bool = True,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, (dict, str))
        if act_cfg == 'Default':
            act_cfg = dict(type='ReLU')
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {'conv', 'norm', 'act'}
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias
        self.conv = build_conv_layer(
            conv_cfg, in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_norm:
            if order.index('norm') < order.index('conv'):
                norm_channels = in_channels
            else:
                norm_channels = out_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)
            if self.with_bias:
                warnings.warn('使用Norm層後在卷積層可以不使用偏置')

        if self.with_activation:
            act_cfg_ = act_cfg.copy()
            if act_cfg_['type'] not in ['Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU']:
                act_cfg_['inplace'] = act_cfg_.get('inplace', inplace)
            self.activate = build_activation_layer(act_cfg_)
        self.init_weights()

    def init_weights(self):
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def forward(self, x, norm=True, act=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and act and self.with_activation:
                x = self.activate(x)
        return x


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_cfg=None,
                 act_cfg='Default',
                 dw_norm_cfg='Default',
                 dw_act_cfg='Default',
                 pw_norm_cfg='Default',
                 pw_act_cfg='Default',
                 **kwargs):
        super(DepthwiseSeparableConvModule, self).__init__()
        assert 'groups' not in kwargs, '在使用dw卷積時，不可以手動設定卷積的組'
        if act_cfg == 'Default':
            act_cfg = dict(type='ReLU')
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'Default' else norm_cfg
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'Default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'Default' else norm_cfg
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'Default' else act_cfg
        self.depthwise_conv = ConvModule(
            in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, norm_cfg=dw_norm_cfg, act_cfg=dw_act_cfg, **kwargs)
        self.pointwise_conv = ConvModule(
            in_channels, out_channels, kernel_size=1, norm_cfg=pw_norm_cfg, act_cfg=pw_act_cfg, **kwargs)
        self.init_weights()

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def reduce_loss(loss, reduction):
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    if weight is not None:
        loss = loss * weight
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError
    return loss


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask & (labels < label_channels), as_tuple=False)
    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1
    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0), label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask
    return bin_labels, bin_label_weights, valid_mask


class CrossEntropyLoss(nn.Module):
    def __init__(self, use_sigmoid=False, use_mask=False, reduction='mean', class_weight=None, ignore_index=None,
                 loss_weight=1.0, avg_non_ignore=None):
        super(CrossEntropyLoss, self).__init__()
        assert (use_mask is False) or (use_sigmoid is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if self.use_sigmoid:
            self.cls_criterion = self.binary_cross_entropy
        else:
            self.cls_criterion = self.cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, ignore_index=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        if ignore_index is None:
            ignore_index = self.ignore_index
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss = self.cls_criterion(cls_score, label, weight, class_weight=class_weight, reduction=reduction,
                                  avg_factor=avg_factor, ignore_index=ignore_index, avg_non_ignore=self.avg_non_ignore)
        loss = loss * self.loss_weight
        return loss

    @staticmethod
    def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None, ignore_index=-100,
                      avg_non_ignore=False):
        ignore_index = -100 if ignore_index is None else ignore_index
        loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none', ignore_index=ignore_index)
        if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
            avg_factor = label.numel() - (label == ignore_index).sum().item()
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return loss

    @staticmethod
    def binary_cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None,
                             ignore_index=-100, avg_non_ignore=False):
        ignore_index = -100 if ignore_index is None else ignore_index
        if pred.dim() != label.dim():
            label, weight, valid_mask = _expand_onehot_labels(label, weight, pred.size(-1), ignore_index)
        else:
            valid_mask = ((label >= 0) & (label != ignore_index)).float()
            if weight is not None:
                weight = weight * valid_mask
            else:
                weight = valid_mask
        if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
            avg_factor = valid_mask.sum().item()
        weight = weight.float()
        loss = F.binary_cross_entropy_with_logits(pred, label.float(), pos_weight=class_weight, reduction='none')
        loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
        return loss


class IoULoss(nn.Module):
    def __init__(self, linear=False, eps=1e-6, reduction='mean', loss_weight=1.0, mode='log'):
        super(IoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override if reduction_override else self.reduction
        loss = self.loss_weight * self.get_iou_loss(pred, target, weight, mode=self.mode, eps=self.eps,
                                                    reduction=reduction, avg_factor=avg_factor, **kwargs)
        return loss

    def get_iou_loss(self, pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs):
        loss = self.iou_loss(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    @staticmethod
    def iou_loss(pred, target, mode='log', eps=1e-6):
        assert mode in ['linear', 'square', 'log']
        ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
        if mode == 'linear':
            loss = 1 - ious
        elif mode == 'square':
            loss = 1 - ious ** 2
        elif mode == 'log':
            loss = -ious.log()
        else:
            raise NotImplementedError
        return loss
