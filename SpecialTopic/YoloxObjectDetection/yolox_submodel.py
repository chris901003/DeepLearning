from torch import nn
import torch
import torch.nn.functional as F
from common_use_model import ConvModule, DepthwiseSeparableConvModule, bbox_overlaps


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(Focus, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        if act_cfg is None:
            act_cfg = dict(type='Swish')
        self.conv = ConvModule(
            in_channels * 4,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_top_right, patch_bot_left, patch_bot_right), dim=1)
        out = self.conv(x)
        return out


class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 9, 13),
                 conv_cfg=None, norm_cfg='Default', act_cfg='Default'):
        super(SPPBottleneck, self).__init__()
        if norm_cfg == 'Default':
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        if act_cfg == 'Default':
            act_cfg = dict(type='Swish')
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid_channels, kernel_size=1, stride=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.poolings = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_size])
        conv2_channels = mid_channels * (len(kernel_size) + 1)
        self.conv2 = ConvModule(conv2_channels, out_channels, kernel_size=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


class DarknetBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.5, add_identity=True, use_depthwise=False,
                 conv_cfg=None, norm_cfg='Default', act_cfg='Default'):
        super(DarknetBottleneck, self).__init__()
        if norm_cfg == 'Default':
            norm_cfg = dict(type='BN', momentum=0.03, eps=0.01)
        if act_cfg == 'Default':
            act_cfg = dict(type='Swish')
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(in_channels, hidden_channels, kernel_size=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = conv(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1,
                          conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.add_identity:
            out = out + identity
        return out


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, num_blocks=1, add_identity=True,
                 use_depthwise=False, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(CSPLayer, self).__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels, mid_channels, kernel_size=1,
                                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.short_conv = ConvModule(in_channels, mid_channels, kernel_size=1,
                                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.final_conv = ConvModule(2 * mid_channels, out_channels, kernel_size=1,
                                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.blocks = nn.Sequential(*[
            DarknetBottleneck(
                in_channels=mid_channels,
                out_channels=mid_channels,
                expansion=1.0,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        x_final = torch.cat((x_main, x_short), dim=1)
        out = self.final_conv(x_final)
        return out


class MlvlPointGenerator:
    def __init__(self, strides, offset=0.5):
        self.strides = [(stride, stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self):
        return len(self.strides)

    @staticmethod
    def _meshgrid(x, y, row_major=True):
        yy, xx = torch.meshgrid(y, x)
        if row_major:
            return xx.reshape(-1), yy.reshape(-1)
        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self, featmap_size, dtype=torch.float32, device='cuda', with_stride=False):
        assert self.num_levels == len(featmap_size)
        multi_level_priors = []
        for i in range(self.num_levels):
            priors = self.single_level_grid_priors(featmap_size[i], level_idx=i, dtype=dtype,
                                                   device=device, with_stride=with_stride)
            multi_level_priors.append(priors)
        return multi_level_priors

    def single_level_grid_priors(self, featmap_size, level_idx, dtype=torch.float32, device='cuda', with_stride=False):
        feat_h, feat_w = featmap_size
        stride_w, stride_h = self.strides[level_idx]
        shift_x = (torch.arange(0, feat_w, device=device) + self.offset) * stride_w
        shift_x = shift_x.to(dtype)
        shift_y = (torch.arange(0, feat_h, device=device) + self.offset) * stride_h
        shift_y = shift_y.to(dtype)
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            stride_w = shift_xx.new_full((shift_xx.shape[0], ), stride_w).to(dtype)
            stride_h = shift_xx.new_full((shift_yy.shape[0], ), stride_h).to(dtype)
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)
        all_points = shifts.to(device)
        return all_points


class AssignResult:
    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels


class SimOTAAssigner:
    def __init__(self, center_radius=2.5, candidate_topk=10, iou_weight=3.0, cls_weight=1.0):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        num_gt = gt_bboxes.size(0)
        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y
        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y
        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y
        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all
        is_in_boxes_and_centers = (is_in_gts[is_in_gts_or_centers, :] & is_in_cts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1
        del topk_ious, dynamic_ks, pos_idx
        prior_match_gt_mask = matching_matrix.sum(dim=1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(dim=1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds

    def assign(self, pred_scores, priors, decoded_bboxes, gt_bboxes, gt_labels, eps=1e-7):
        INF = 100000.0
        num_gt = gt_bboxes.size(0)
        num_bboxes = decoded_bboxes.size(0)
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ), 0, dtype=torch.long)
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(priors, gt_bboxes)
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + eps)
        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float().unsqueeze(0).repeat(num_valid, 1, 1))
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        try:
            cls_cost = (F.binary_cross_entropy(
                valid_pred_scores.to(dtype=torch.float32).sqrt_(),
                gt_onehot_label,
                reduction='none'
            ).sum(dim=-1).to(dtype=valid_pred_scores.dtype))
        except:
            print('f')
        cost_matrix = (
            cls_cost * self.cls_weight + iou_cost * self.iou_weight + (~is_in_boxes_and_center) * INF)
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt, valid_mask)
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)


class SamplingResult:
    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_bboxes = bboxes[pos_inds]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]
        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        if gt_bboxes.numel() == 0:
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long(), :]
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None


class PseudoSampler:
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def sample(assign_result, bboxes, gt_bboxes):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        return sampling_result
