# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class SimOTAAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth.

    Args:
        center_radius (int | float, optional): Ground truth center size
            to judge whether a prior is in center. Default 2.5.
        candidate_topk (int, optional): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Default 10.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 3.0.
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
    """

    def __init__(self,
                 center_radius=2.5,
                 candidate_topk=10,
                 iou_weight=3.0,
                 cls_weight=1.0):
        """ 匹配預測匡以及真實匡
        Args:
            center_radius: 真實標註匡中心點往外擴散半徑內會設定成正樣本候選點
            candidate_topk: 每個標註匡分配的正樣本當中會取出前k大的iou作為二階段正樣本候選點
            iou_weight: iou在計算cost時的權重
            cls_weight: 分類損失在計算cost時的權重
        """
        # 保存傳入參數
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight

    def assign(self,
               pred_scores,
               priors,
               decoded_bboxes,
               gt_bboxes,
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Assign gt to priors using SimOTA. It will switch to CPU mode when
        GPU is out of memory.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            eps (float): A value added to the denominator for numerical
                stability. Default 1e-7.
        Returns:
            assign_result (obj:`AssignResult`): The assigned result.
        """
        """ 進行SimOTA的正負樣本匹配
        Args:
            pred_scores: 類別置信度，由分類類別概率乘上正樣本概率獲取，tensor shape [sum(height * width), 4]
            priors: 每個特徵圖的方格的中心點座標以及當前是那個縮放比例
            decoded_bboxes: 經過縮放到原始圖像的預測匡
            gt_bboxes: 標註匡部分，tensor shape [num_object, 4]
            gt_labels: 標註匡對應上的類別，tensor shape [num_object]
            gt_bboxes_ignore: 被忽略掉的標註匡部分
            eps: 極小值，避免分母為0的情況
        """
        try:
            # 使用assign嘗試正負樣本匹配
            assign_result = self._assign(pred_scores, priors, decoded_bboxes,
                                         gt_bboxes, gt_labels,
                                         gt_bboxes_ignore, eps)
            # 如果匹配成功就直接回傳
            return assign_result
        except RuntimeError:
            # 如果運行過程發生記憶體不足就會到這裡
            origin_device = pred_scores.device
            # 跳出警告表示需要減少batch_size
            warnings.warn('OOM RuntimeError is raised due to the huge memory '
                          'cost during label assignment. CPU mode is applied '
                          'in this batch. If you want to avoid this issue, '
                          'try to reduce the batch size or image size.')
            # 將cache部分進行清除
            torch.cuda.empty_cache()

            # 將資料轉到cpu上，因為這樣就會吃電腦的ram當電腦ram不足時會吃硬碟，基本上就不會有記憶體不足問題
            pred_scores = pred_scores.cpu()
            priors = priors.cpu()
            decoded_bboxes = decoded_bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu().float()
            gt_labels = gt_labels.cpu()

            # 在cpu下進行配對
            assign_result = self._assign(pred_scores, priors, decoded_bboxes,
                                         gt_bboxes, gt_labels,
                                         gt_bboxes_ignore, eps)
            # 最後將結果放回到訓練設備上
            assign_result.gt_inds = assign_result.gt_inds.to(origin_device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(origin_device)
            assign_result.labels = assign_result.labels.to(origin_device)

            # 回傳結果
            return assign_result

    def _assign(self,
                pred_scores,
                priors,
                decoded_bboxes,
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore=None,
                eps=1e-7):
        """Assign gt to priors using SimOTA.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            eps (float): A value added to the denominator for numerical
                stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        """ 進行動態正負樣本匹配SimOTA的主要部分
        Args:
            pred_scores: 類別置信度分數，tensor shape [sum(height * width), num_cls]
            priors: 不同尺度特徵圖映射回原圖的座標點，後的座標匡的中心點位置以及縮放倍率，tensor shape [sum(height * width), 4]
            decoded_bboxes: 經過縮放回原圖座標的預測匡位置，tensor shape [sum(height * width), 4]
            gt_bboxes: 標註匡資訊，tensor shape [num_object, 4]
            gt_labels: 標註匡對應類別，tensor shape [num_object]
            gt_bboxes_ignore: 被忽略掉的標註匡資訊
            eps: 極小值，避免分母為0
        """
        # 先設定一個極大值作為INF
        INF = 100000.0
        # 獲取總共有多少個標註匡
        num_gt = gt_bboxes.size(0)
        # 獲取總共有多少個預測匡
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        # 這裡會先構建全為0且shape是[num_bboxes]的tensor，預設的0表示沒有與任何真實匡有匹配
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes, ), 0, dtype=torch.long)
        # 獲取哪些像素點有在標註匡當中
        # valid_mask[sum(height * width)]，True部分就表示該像素點至少在某一個標註匡當中
        # is_in_boxes_and_center，在valid_mask當中為True的地方詳細表示哪個標註匡是有在裡面的
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(priors, gt_bboxes)
        # 將decoded_bbox進行過濾，將沒有在標註匡內的預測結果去除
        valid_decoded_bbox = decoded_bboxes[valid_mask]
        # 同時去除置信度部分資料
        valid_pred_scores = pred_scores[valid_mask]
        # 獲取剩下有多少是合法的預測匡
        num_valid = valid_decoded_bbox.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # 如果沒有剩下合法的預測匡就會到這裡
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_bboxes.new_full((num_bboxes, ), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # 計算預測匡與真實匡的iou值，pairwise_ious shape = [valid_pred_box, num_object]
        pairwise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        # 計算iou的cost值，這裡iou值越大越好所以會加上負號
        iou_cost = -torch.log(pairwise_ious + eps)

        # 獲取標註匡類別的one-hot格式，shape [num_valid, num_object, num_cls]
        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1]).float().unsqueeze(0).repeat(num_valid, 1, 1))

        # 調整類別預測置信度，shape [num_valid, num_object, num_cls]
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        # 計算類別置信度的cost，shape [num_valid, num_object]
        cls_cost = (
            F.binary_cross_entropy(
                valid_pred_scores.to(dtype=torch.float32).sqrt_(),
                gt_onehot_label,
                reduction='none',
            ).sum(-1).to(dtype=valid_pred_scores.dtype))

        # 計算cost矩陣，會使用cost矩陣進行正負樣本匹配，shape [num_valid, num_object]
        cost_matrix = (
            cls_cost * self.cls_weight + iou_cost * self.iou_weight + (~is_in_boxes_and_center) * INF)

        # 開始進行動態正負樣本匹配
        # matched_pred_ious = 預測匡與應對應上的真實匡的iou值，shape [num_valid]
        # matched_gt_inds = 預測匡所對應上的真實匡index，shape [num_valid]
        # 這裡因為在內部有更動valid_mask，所以可以知道有哪些預測匡是有對應上正樣本的，這裡的num_valid是最後還有經過過濾的
        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format
        # 將有對應上真實匡的預測匡的紀錄對應上真實匡的資料進行更動，這裡會將對應上的真實匡idx加一，這樣才可以區別負樣本
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        # 先構建一個理論上要匹配的分類類別，這裡初始化為-1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        # 將對應的預測匡位置所對應的類別添加上去，這樣就可以知道該預測匡應該需要預測的類別
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        # 這裡會構建最大重疊，shape會是[sum(height, width)]，初始化為-INF
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ), -INF, dtype=torch.float32)
        # 將指定地方的重疊部分設定成該預測匡與應預測到的真實匡之間的iou值
        max_overlaps[valid_mask] = matched_pred_ious
        # 將資料用AssignResult包裝後回傳
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        # 獲取哪些預測匡有在真實匡當中且在中心點附近
        # 獲取總共有多少真實匡
        num_gt = gt_bboxes.size(0)

        # 獲取中心點x與y的位置以及高寬，這裡會對通道進行調整
        # [sum(height * width)] -> [sum(height * width), 1] -> [sum(height, width), num_gt]
        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes, shape: [n_prior, n_gt]
        # 獲取所有標註匡左上角對於所有像素點左側多少距離
        l_ = repeated_x - gt_bboxes[:, 0]
        # 獲取所有標註匡左上角對於所有像素點上方多少距離
        t_ = repeated_y - gt_bboxes[:, 1]
        # 獲取所有標註匡右下角對於所有像素點右側多少距離
        r_ = gt_bboxes[:, 2] - repeated_x
        # 獲取所有標註匡右下角對於所有像素點下方多少距離
        b_ = gt_bboxes[:, 3] - repeated_y

        # 在第二個維度進行堆疊，deltas shape [sum(height * width), 4, num_object]
        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        # 如果一個像素點在一個標註匡當中理論上(l_, t_, r_, b_)都會大於0，所以這裡會看該像素點有在哪些標註匡當中，如果在當中的會是True
        is_in_gts = deltas.min(dim=1).values > 0
        # 獲取哪個像素點有在所有的標註匡當中，如果該像素點有在隨意一個標註匡中就會是True
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # is prior centers in gt centers
        # 獲取標註匡中心點座標x
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        # 獲取標註匡中心點座標y
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        # 根據中心點以及給定的半徑畫出範圍，tensor shape [sum(height * width), num_object]
        ct_box_l = gt_cxs - self.center_radius * repeated_stride_x
        ct_box_t = gt_cys - self.center_radius * repeated_stride_y
        ct_box_r = gt_cxs + self.center_radius * repeated_stride_x
        ct_box_b = gt_cys + self.center_radius * repeated_stride_y

        # 獲取與標註匡中心點的相對距離
        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        # 進行堆疊，ct_deltas shape [sum(height, width), 4, num_object]
        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        # 如果一個像素點在一個標註匡當中理論上(cl_, ct_, cr_, cb_)都會大於0，所以這裡會看該像素點有在哪些標註匡當中，如果在當中的會是True
        is_in_cts = ct_deltas.min(dim=1).values > 0
        # 獲取哪個像素點有在所有的標註匡當中，如果該像素點有在隨意一個標註匡中就會是True
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        # 這裡只要吻合其中一個就認定是在標註匡當中
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        # 提取出哪些像素點在某個標註匡的中心點半徑內同時又在標註匡內，shape [有符合其中一個標註匡的像素點數量, num_object]
        # 如果有符合該標註匡就會是True
        is_in_boxes_and_centers = (is_in_gts[is_in_gts_or_centers, :] & is_in_cts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, valid_mask):
        """ 動態匹配正負樣本
        Args:
            cost: 某一預測匡匹配到某一真實標註匡所需的cost
            pairwise_ious: 剩餘合法的預測匡與每個標註匡的iou值
            num_gt: 總共有多少個標註匡
            valid_mask: 原先所有預測匡有哪些是有在至少一個標註匡的合法範圍當中
        """
        # 構建一個存放正負樣本匹配的tensor shape [num_valid, num_object]，初始化全為0
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        # 決定要選前k大的預測匡，這裡最多只會選candidate_topk個
        candidate_topk = min(self.candidate_topk, pairwise_ious.size(0))
        # 使用torch.topk獲取前k大的iou，每個標註匡與每個預測匡的前k大的iou
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        # 每個類別對自己的前k大的iou進行加總，最小不會小於1，這個是用來之後說明一個標註匡可以配對上多少個預測匡
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        # 開始遍歷每個標註匡
        for gt_idx in range(num_gt):
            # 獲取當前標註匡應該要由哪些預測匡進行預測，這裡會根據上面計算的dynamic_ks知道需要獲取前k小的cost進行匹配
            # pos_idx就會是對應的預測匡的index
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            # 將對應的地方標註成1，表示該預測匡希望預測到的是當前標註匡，這裡可能會產生一個預測匡要預測到多個不同真實匡的問題
            # 不過這個後面會進行解決，目前先這樣沒有關係
            matching_matrix[:, gt_idx][pos_idx] = 1

        # 將不在需要使用到的記憶體進行釋放
        del topk_ious, dynamic_ks, pos_idx

        # 獲取有哪些預選匡發生，一個預選匡需要預測到多個真實匡，shape [num_valid]
        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            # 如果有任何一個預測匡發生問題就會到這裡
            # 獲取需重新配對的預測匡，對於每個真實匡的最小cost是哪個真實匡
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            # 將需重新配對的預測匡的正樣本匹配不分清空
            matching_matrix[prior_match_gt_mask, :] *= 0
            # 變成只有最小的cost才會進行正樣本匹配，這樣一個預測匡就一定只會匹配上一個真實匡，同時也是該預測匡與其他真實匡的最小cost
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        # 獲取哪些預測匡有匹配到正樣本，如果有匹配到正樣本的預測匡會是True，否則就會是False
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        # 更新valid_mask資訊，原先只有過濾掉不在圖像中或不在標註匡半徑範圍內，現在將沒有匹配到正樣本的也變成False
        # 也就是在valid_mask當中是True就一定有配對到某個真實標註匡
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        # 獲取有匹配到正樣本的預測匡所匹配到的真實匡index，shape [num_valid]
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        # 獲取正樣本與應該要配對上的真實標註匡的iou值，shape [num_valid]
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
