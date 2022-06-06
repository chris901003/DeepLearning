# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        # 由下面實例化
        # 已看過
        super().__init__()
        # 就是先附值而已
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # 已看過
        # 輸入參數部分可以看上面的註解，注意dim基本就可以了
        # 取出batch_size以及num_queries的數量
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # 將batch_size以及num_queries維度展平
        # out_prob對分類部分進行softmax
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # batch_size與num_queries展平
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        # 同時target的部分我們需要把batch_size的部分拼接在一起
        # 如果對於targets裡面存的東西有疑問可以到coco.py裡的ConvertCocoPolysToMask的__call__的最後面看看
        # targets會是(List(Dict))，List的長度就是batch_size
        # 這裡的boxes是相對座標
        # tgt_idx shape [total_gt_box], total_gt_box = all_image_in_batch's gt_box
        # tgt_bbox shape [total_gt_box, 4]
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # 抓出out_prob中對應idx地方的概率，這邊先都不考慮到底是哪張照片因為batch_size都已經混合了
        # cost_class shape [batch_size * num_queries, total_gt_box]
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        # cdist參數p表示要用L幾
        # 用每一個預測匡去跟所有的真實匡做L1正則項係數，計算方式就是相減後絕對值在相加
        # Ex:
        # v1 = torch.tensor([[2., 4.], [5., 9.], [4., 8.]])
        # v2 = torch.tensor([[4., 6.]])
        # v3 = torch.cdist(v1, v2, p=1)
        # v3 = tensor([[4.], [4.], [2.]])
        # cost_bbox shape [batch_size * num_queries, total_gt_box]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # 先轉成左上右下後再去計算giou
        # cost_giou shape [batch_size * num_queries, total_gt_box]
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        # 計算總損失，需乘上對應的權重超參數
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # 將shape變形一下 [batch_size * num_queries, total_gt_box] -> [batch_size, num_queries, total_gt_box]
        C = C.view(bs, num_queries, -1).cpu()

        # 獲得每張照片有多少個gt_box，size shape [batch_size]，存放的值就是這張照片有多少個gt_box
        sizes = [len(v["boxes"]) for v in targets]
        # ---------------------------------------------------------
        # C.split = 給定一個分割的list表示怎麼分，後面的-1表示在最後一個維度做切分
        # 這樣我們就可以依據不同的照片分類出來，歸類好哪些損失是屬於哪些部分的
        # c shape [batch_size, num_queries, gt_box_for_index_i]
        # 這裡我們沒有指定batch中的哪張圖片要對上這些gt_box所以下面有寫c[i]
        # c[i] = 取出指定的batch index，這樣就完全對上了
        # ---------------------------------------------------------
        # linear_sum_assignment = 匈牙利算法，找到最小的cost
        # 輸入的shape [num_queries, gt_box_for_index_i]
        # 我們可以看成總共有num_queries個工人以及gt_box_for_index_i個工作
        # 裡面存放的值就是這個工人做這個工作所需消耗的工資為多少
        # linear_sum_assignment會幫我們算出最小花費
        # 這裡回傳的會是tuple類型裡面有(row, col)
        # 相同index的row以及col就可以找回原始的座標位置，c[row[i]][col[i]]
        # ---------------------------------------------------------
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # 把每張照片拆出來
        # return List[tuple(row, col)]，List長度就是batch_size大小
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    # 已看過
    # cost_class = class coefficient
    # cost_bbox = L1 box coefficient
    # cost_giou = giou box coefficient
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
