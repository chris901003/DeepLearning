import torch
from torch import nn
import torch.nn.functional as F
from utils import get_specified_option
import numpy as np
import torchvision
from common_use_model import ConvModule, DepthwiseSeparableConvModule, CrossEntropyLoss, IoULoss, build_optimizer
from yolox_submodel import Focus, SPPBottleneck, CSPLayer, MlvlPointGenerator, SimOTAAssigner, PseudoSampler


def build_detector(model_cfg):
    support_detector = {
        'YOLOX': YOLOX
    }
    detection_cls = get_specified_option(support_detector, model_cfg)
    detection = detection_cls(**model_cfg)
    return detection


def build_backbone(backbone_cfg):
    support_backbone = {
        'CSPDarknet': CSPDarknet
    }
    backbone_cls = get_specified_option(support_backbone, backbone_cfg)
    backbone = backbone_cls(**backbone_cfg)
    return backbone


def build_neck(neck_cfg):
    support_neck = {
        'YOLOXPAFPN': YOLOXPAFPN,
    }
    neck_cls = get_specified_option(support_neck, neck_cfg)
    neck = neck_cls(**neck_cfg)
    return neck


def build_head(head_cfg):
    support_head = {
        'YOLOXHead': YOLOXHead
    }
    head_cls = get_specified_option(support_head, head_cfg)
    head = head_cls(**head_cfg)
    return head


def build_loss(loss_cfg):
    support_loss = {
        'IoULoss': IoULoss,
        'CrossEntropyLoss': CrossEntropyLoss
    }
    loss_cls = get_specified_option(support_loss, loss_cfg)
    loss = loss_cls(**loss_cfg)
    return loss


def build_assigner(assigner_cfg):
    support_assigner = {
        'SimOTAAssigner': SimOTAAssigner
    }
    assigner_cls = get_specified_option(support_assigner, assigner_cfg)
    assigner = assigner_cls(**assigner_cfg)
    return assigner


def build_sampler(sampler_cfg, **kwargs):
    support_sampler = {
        'PseudoSampler': PseudoSampler
    }
    sampler_cls = get_specified_option(support_sampler, sampler_cfg)
    sampler = sampler_cls(**sampler_cfg, **kwargs)
    return sampler


class YOLOX(nn.Module):
    def __init__(self, backbone, neck, bbox_head, train_cfg=None, test_cfg=None,
                 input_size=(640, 640), size_multiplier=32, optimizer_cfg='default'):
        super(YOLOX, self).__init__()
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._default_input_size = input_size
        self._input_size = input_size
        self._size_multiplier = size_multiplier
        self._progress_in_iter = 0
        if optimizer_cfg == 'default':
            optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9)
        self.optimizer = build_optimizer(self, optimizer_cfg)

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.neck:
            x = self.neck(x)
        return x

    def forward_train(self, img, gt_bboxes, gt_labels):
        self.optimizer.zero_grad()
        img, gt_bboxes = self._preprocess(img, gt_bboxes)
        x = self.extract_feat(img)
        if torch.isnan(x[0]).sum():
            import numpy as np
            t = np.max(img.numpy())
            print('f')
        losses = self.bbox_head.forward_train(x, gt_bboxes, gt_labels)
        losses['loss_bbox'] = torch.sum(losses['loss_bbox'])
        loss = sum(_value for _key, _value in losses.items())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def forward_test(self, imgs, scale_factor=None):
        assert imgs.shape[0] == 1, '目前只支持驗證時使用單張圖像'
        rescale = scale_factor is not None
        feat = self.extract_feat(imgs)
        results_list = self.bbox_head.simple_test(feat, scale_factor, rescale=rescale)
        bbox_results = [self.bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
                        for det_bboxes, det_labels in results_list]
        return bbox_results

    def forward(self, img, gt_bboxes=None, gt_labels=None, scale_factor=None, return_loss=True):
        if return_loss:
            assert gt_bboxes is not None
            assert gt_labels is not None
            return self.forward_train(img, gt_bboxes, gt_labels)
        else:
            return self.forward_test(img, scale_factor=scale_factor)

    @staticmethod
    def bbox2result(bboxes, labels, num_classes):
        if bboxes.shape[0] == 0:
            return [np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)]
        else:
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
            return [bboxes[labels == i, :] for i in range(num_classes)]

    def _preprocess(self, img, gt_bboxes):
        scale_y = self._input_size[0] / self._default_input_size[0]
        scale_x = self._input_size[1] / self._default_input_size[1]
        if scale_x != 1 or scale_y != 1:
            img = F.interpolate(img, size=self._input_size, mode='bilinear', align_corners=False)
            for gt_bboxes in gt_bboxes:
                gt_bboxes[..., 0::2] = gt_bboxes[..., 0::2] * scale_x
                gt_bboxes[..., 1::2] = gt_bboxes[..., 1::2] * scale_y
        return img, gt_bboxes


class CSPDarknet(nn.Module):
    def __init__(self, deepen_factor=1.0, widen_factor=1.0, out_indices=(2, 3, 4), use_depthwise=False,
                 spp_kernel_size=(5, 9, 13), conv_cfg=None, norm_cfg=None, act_cfg=None, norm_eval=False):
        super(CSPDarknet, self).__init__()
        # 分別表示的是[in_channel, out_channel, num_blocks, add_identity, use_spp]
        arch_setting = [[64, 128, 3, True, False], [128, 256, 9, True, False],
                        [256, 512, 9, True, False], [512, 1024, 3, False, True]]
        assert set(out_indices).issubset(i for i in range(len(arch_setting) + 1))
        self.out_indices = out_indices
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        if norm_cfg is None:
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        if act_cfg is None:
            act_cfg = dict(type='Swish')
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        self.stem = Focus(
            in_channels=3, out_channels=int(arch_setting[0][0] * widen_factor), kernel_size=3,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.layers = ['stem']
        for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)
            out_channels = int(out_channels * widen_factor)
            num_blocks = max(round(num_blocks * deepen_factor), 1)
            stage = []
            conv_layer = conv(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(out_channels, out_channels, kernel_size=spp_kernel_size,
                                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
                stage.append(spp)
            csp_layer = CSPLayer(out_channels, out_channels, num_blocks=num_blocks, add_identity=add_identity,
                                 use_depthwise=use_depthwise, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            stage.append(csp_layer)
            self.add_module(f'stage{i + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{i + 1}')

    def forward(self, x):
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs


class YOLOXPAFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_csp_blocks=3, use_depthwise=False, upsample_cfg='Default',
                 conv_cfg=None, norm_cfg='Default', act_cfg='Default'):
        super(YOLOXPAFPN, self).__init__()
        if upsample_cfg == 'Default':
            upsample_cfg = dict(scale_factor=2, mode='nearest')
        if norm_cfg == 'Default':
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        if act_cfg == 'Default':
            act_cfg = dict(type='Swish')
        self.in_channels = in_channels
        self.out_channels = out_channels
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(in_channels[idx], in_channels[idx - 1], kernel_size=1,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(in_channels[idx - 1] * 2, in_channels[idx - 1], num_blocks=num_csp_blocks, add_identity=False,
                         use_depthwise=use_depthwise, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(in_channels[idx], in_channels[idx], kernel_size=3, stride=2, padding=1,
                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(in_channels[idx] * 2, in_channels[idx + 1], num_blocks=num_csp_blocks, add_identity=False,
                         use_depthwise=use_depthwise, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(in_channels[i], out_channels, kernel_size=1,
                           conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = self.upsample(feat_high)
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])
        return tuple(outs)


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, in_channels, feat_channels=256, stacked_convs=2, strides='Default',
                 use_depthwise=False, conv_bias='auto', conv_cfg=None, norm_cfg='Default', act_cfg='Default',
                 loss_cls='Default', loss_bbox='Default', loss_obj='Default', loss_l1='Default', train_cfg=None,
                 test_cfg=None):
        super(YOLOXHead, self).__init__()
        if strides == 'Default':
            strides = [8, 16, 32]
        if norm_cfg == 'Default':
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        if act_cfg == 'Default':
            act_cfg = dict(type='Swish')
        if loss_cls == 'Default':
            loss_cls = dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='sum', loss_weight=1.0)
        if loss_bbox:
            loss_bbox = dict(type='IoULoss', mode='square', eps=1e-16, reduction='sum', loss_weight=5.0)
        if loss_obj == 'Default':
            loss_obj = dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='sum', loss_weight=1.0)
        if loss_l1 == 'Default':
            loss_l1 = dict(type='L1loss', reduction='sum', loss_weight=1.0)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_obj = build_loss(loss_obj)
        self.loss_l1 = loss_l1
        self.use_l1 = False
        self.prior_generator = MlvlPointGenerator(strides, offset=0)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg['assigner'])
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self._init_layers()

    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)

    def _build_stacked_convs(self):
        conv = DepthwiseSeparableConvModule if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(chn, self.feat_channels, kernel_size=3, stride=1, padding=1,
                     conv_cfg=conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, bias=self.conv_bias)
            )
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self):
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, kernel_size=1)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        return conv_cls, conv_reg, conv_obj

    def forward_train(self, x, gt_bboxes, gt_labels):
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + gt_bboxes
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels)
        losses = self.loss(*loss_inputs)
        return losses

    def simple_test(self, feats, scale_factor, rescale=False):
        outs = self(feats)
        results_list = self.get_bboxes(*outs, scale_factor=scale_factor, rescale=rescale)
        return results_list

    def get_bboxes(self, cls_scores, bbox_preds, objectnesses, scale_factor=None, cfg=None,
                   rescale=False, with_nms=True):
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        scale_factor = np.stack(scale_factor)
        num_imgs = len(cls_scores[0])
        featmap_size = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_size, dtype=cls_scores[0].dtype,
                                                       device=cls_scores[0].device, with_stride=True)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
                              for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
                              for bbox_pred in bbox_preds]
        flatten_objectness = [objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                              for objectness in objectnesses]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        if rescale:
            flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(scale_factor).unsqueeze(1)
        result_list = []
        for img_id in range(num_imgs):
            cls_scores = flatten_cls_scores[img_id]
            score_factor = flatten_objectness[img_id]
            bboxes = flatten_bboxes[img_id]
            result_list.append(self._bboxes_nms(cls_scores, bboxes, score_factor, cfg))
        return result_list

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= cfg['score_thr']
        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]
        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, labels = self.batched_nms(bboxes, scores, labels, cfg['nms'])
            return dets, labels

    @staticmethod
    def batched_nms(bboxes, scores, labels, nms):
        max_wh = 4096
        offset_bbox = bboxes.clone() + labels.view(-1, 1) * max_wh
        i = torchvision.ops.nms(offset_bbox, scores, nms['iou_threshold'])
        max_num = nms.get('max_num', None)
        if max_num is not None:
            i = i[:min(len(i), max_num)]
        bboxes = bboxes[i]
        scores = scores[i]
        labels = labels[i]
        dets = torch.cat([bboxes, scores.unsqueeze(dim=1)], dim=1)
        return dets, labels

    @staticmethod
    def forward_single(x, cls_convs, reg_convs, conv_cls, conv_reg, conv_obj):
        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)
        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        objectness = conv_obj(reg_feat)
        return cls_score, bbox_pred, objectness

    def forward(self, feats):
        cls_scores, bbox_preds, objectnesses = list(), list(), list()
        for idx, feat in enumerate(feats):
            cls_score, bbox_pred, objectness = self.forward_single(feat,
                                                                   self.multi_level_cls_convs[idx],
                                                                   self.multi_level_reg_convs[idx],
                                                                   self.multi_level_conv_cls[idx],
                                                                   self.multi_level_conv_reg[idx],
                                                                   self.multi_level_conv_obj[idx])
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            objectnesses.append(objectness)
        return cls_scores, bbox_preds, objectnesses

    @staticmethod
    def _bbox_decode(priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]
        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)
        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=-1)
        return decoded_bboxes

    def _get_target_single(self, cls_preds, objectness, priors, decode_bboxes, gt_bboxes, gt_labels, idx):
        cls_preds = cls_preds[idx]
        objectness = objectness[idx]
        priors = priors[idx]
        decode_bboxes = decode_bboxes[idx]
        gt_bboxes = gt_bboxes[idx]
        gt_labels = gt_labels[idx]

        num_priors = priors.size(0)
        num_gts = gt_bboxes.size(0)
        gt_bboxes = gt_bboxes.to(decode_bboxes.dtype)
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return foreground_mask, cls_target, obj_target, bbox_target, l1_target, 0
        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)
        assign_result = self.assigner.assign(cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
                                             offset_priors, decode_bboxes, gt_bboxes, gt_labels)
        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]
        cls_target = F.one_hot(sampling_result.pos_gt_labels, self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            pass
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        return foreground_mask, cls_target, obj_target, bbox_target, l1_target, num_pos_per_img

    def loss(self, cls_scores, bbox_preds, objectnesses, gt_bboxes, gt_labels):
        num_imgs = len(gt_bboxes)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype,
                                                       device=cls_scores[0].device, with_stride=True)
        flatten_cls_preds = [cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
                             for cls_pred in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
                              for bbox_pred in bbox_preds]
        flatten_objectness = [objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                              for objectness in objectnesses]
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)
        pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets, num_fg_imgs = list(), list(), list(), list(), \
                                                                                     list(), list()
        for idx in range(num_imgs):
            pos_mask, cls_target, obj_target, bbox_target, l1_target, num_fg_img = self._get_target_single(
                flatten_cls_preds.detach(), flatten_objectness.detach(),
                flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1), flatten_bboxes.detach(), gt_bboxes, gt_labels, idx
            )
            pos_masks.append(pos_mask)
            cls_targets.append(cls_target)
            obj_targets.append(obj_target)
            bbox_targets.append(bbox_target)
            l1_targets.append(l1_target)
            num_fg_imgs.append(num_fg_img)
        num_pos = torch.tensor(sum(num_fg_imgs), dtype=torch.float, device=flatten_cls_preds.device)
        num_total_samples = max(num_pos, 1.0)

        pos_masks = torch.cat(pos_masks, dim=0)
        cls_targets = torch.cat(cls_targets, dim=0)
        obj_targets = torch.cat(obj_targets, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, dim=0)
        loss_bbox = self.loss_bbox(flatten_bboxes.view(-1, 4)[pos_masks], bbox_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1), obj_targets) / num_total_samples
        loss_cls = self.loss_cls(flatten_cls_preds.view(-1, self.num_classes)
                                 [pos_masks], cls_targets) / num_total_samples

        loss_dict = dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)
        return loss_dict
