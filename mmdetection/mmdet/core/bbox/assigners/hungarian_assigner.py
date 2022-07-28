# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..match_costs import build_match_cost
from ..transforms import bbox_cxcywh_to_xyxy
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0)):
        """ 已看過，匈牙利匹配的初始化類
        Args:
            cls_cost: 類別的cost計算方式
            reg_cost: 預測匡的cost計算方式
            iou_cost: iou的cost計算方式
        """

        # 構建各種cost計算方式的實例對象
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        # 已看過，將預測匡與真實匡進行配對
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        # 獲取真實匡數量以及預測匡數量
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        # 預設都先將配對到的index都先設定成-1，shape [num_pred]
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        # 每個預測匡對應到的正確label都先設定成-1，shape [num_pred]
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # 如果圖像當中沒有標註匡就會到這裡
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
        # 獲取圖像高寬
        img_h, img_w, _ = img_meta['img_shape']
        # 構建需要調整預測匡的縮放比例，因為預測的會是相對座標而標註的是絕對座標
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        # 獲取類別匹配的cost，shape [num_pred, num_gt]
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        # 將gt標註匡變成相對座標
        normalize_gt_bboxes = gt_bboxes / factor
        # 計算標註匡的l1損失，shape [num_pred, num_gt]
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        # 將預測的匡轉成[xmin, ymin, xmax, ymax]並且變成絕對座標
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        # 計算iou損失，shape [num_pred, num_gt]
        iou_cost = self.iou_cost(bboxes, gt_bboxes)
        # weighted sum of above three costs，配對的總cost就會是全部加在一起
        cost = cls_cost + reg_cost + iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        # cost shape ndarray [num_pred, num_gt]
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            # 這裡我們會用到python內建的km算法庫，所以會檢查是否有進行安裝
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        # 透過linear_sum_assignment就會進行最佳匹配
        # matched_row_inds = 哪個預測匡應該要對應上哪個真實匡
        # matched_col_inds = 哪個真實匡應該要對應上哪個預測匡
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        # 轉成tensor格式
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        # 將assigned_gt_inds全部設定成0
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        # 因為原先將預測匡對應的標註匡設定成0，所以我們這邊將有對應上的標註匡index都加上1
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        # 該標註匡對應到的類別，沒有對應上的就會是-1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        # 將資料放到AssignResult變成一個實例對象
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
