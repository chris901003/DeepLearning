#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import math
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class IOUloss(nn.Module):
    # 由下面的YOLOloss實例化
    def __init__(self, reduction="none", loss_type="iou"):
        # 已看過
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        # 已看過
        # 就是iou或giou計算
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class YOLOLoss(nn.Module):
    # 由train.py實例化
    def __init__(self, num_classes, fp16, strides=[8, 16, 32]):
        # 已看過
        super().__init__()
        self.num_classes = num_classes
        # 第i個特徵圖還原到原圖的比例
        self.strides = strides

        # BCEWithLogitsLoss就是會先對輸入做sigmoid在與正確值做CrossEntropy
        # reduction就是計算完後不會做任何的後處理
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        # IOUloss在上面，等調用時我們再去看forward
        self.iou_loss = IOUloss(reduction="none")
        # grids = [0., 0., 0.]
        self.grids = [torch.zeros(1)] * len(strides)
        self.fp16 = fp16

    def forward(self, inputs, labels=None):
        # 已看過
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        # -----------------------------------------------#
        # inputs    [[batch_size, num_classes + 5, 20, 20]
        #            [batch_size, num_classes + 5, 40, 40]
        #            [batch_size, num_classes + 5, 80, 80]]
        # outputs   [[batch_size, 400, num_classes + 5]
        #            [batch_size, 1600, num_classes + 5]
        #            [batch_size, 6400, num_classes + 5]]
        # x_shifts  [[batch_size, 400]
        #            [batch_size, 1600]
        #            [batch_size, 6400]]
        # -----------------------------------------------#
        for k, (stride, output) in enumerate(zip(self.strides, inputs)):
            # k表示現在在第幾個yolo head，可以知道要用哪個strides
            # output shape [(batch_size, 5 + num_classes, w * h)]
            # (center_x, center_y, w, h)
            # output的預測匡都已經映射回輸入網路中圖像的位置
            output, grid = self.get_output_and_grid(output, k, stride)
            # x_shifts就是x座標 [[0, 1, 0, 1, 0, 1]] => hsize = 3, wsize = 2
            x_shifts.append(grid[:, :, 0])
            # y_shifts就是y座標 [[0, 0, 1, 1, 2, 2]] => hsize = 3, wsize = 2
            y_shifts.append(grid[:, :, 1])
            # shape與x_shifts一樣用處是知道特徵圖對應原圖的縮放比例
            expanded_strides.append(torch.ones_like(grid[:, :, 0]) * stride)
            outputs.append(output)

        # torch.cat(outputs, 1) = 把每個yolo head出來的全部拼接，因為我們已經縮放回input_size的圖片上面了，所以可以拼接起來
        # torch.cat(outputs, 1) shape = (batch_size, 400 + 1600 + 6400, 5 + num_classes)
        return self.get_losses(x_shifts, y_shifts, expanded_strides, labels, torch.cat(outputs, 1))

    def get_output_and_grid(self, output, k, stride):
        # 已看過
        grid = self.grids[k]
        # hsize, wsize = 特徵圖的高和寬
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            # 第一次計算損失的時候會沒有圖，在這裡建圖
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, hsize, wsize, 2).type(output.type())
            self.grids[k] = grid
        # Ex: 2 * 3 (y * x)
        # grid = [[x, y],[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
        grid = grid.view(1, -1, 2)

        # flatten = (batch_size, 5 + num_classes, w, h) => (batch_size, 5 + num_classes, w * h)
        # permute = (batch_size, 5 + num_classes, w * h) => (batch_size, w * h, 5 + num_classes)
        output = output.flatten(start_dim=2).permute(0, 2, 1)
        # type_as = 換成一樣的變數型態
        # output[..., :2] + grid.type_as(output) = 將左上角網格點位置再加上偏移量獲得在特徵圖上的中心點位置
        # * stride = 轉換回輸入圖大小的絕對位置
        output[..., :2] = (output[..., :2] + grid.type_as(output)) * stride
        # 將預測的高和寬透過公式變成要的樣子，同時也乘上stride縮放回輸入圖的大小
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        # 這裡縮放回的都是input_size的大小，因為target紀錄的也是縮放到input_shape大小的gt_box參數
        # 在dataloader中就有把gt_box轉換過了
        return output, grid

    def get_losses(self, x_shifts, y_shifts, expanded_strides, labels, outputs):
        # 已看過
        # outputs shape (batch_size, total_anchors, 5 + num_classes)
        # -----------------------------------------------#
        #   [batch, n_anchors_all, 4]
        # -----------------------------------------------#
        bbox_preds = outputs[:, :, :4]
        # -----------------------------------------------#
        #   [batch, n_anchors_all, 1]
        # -----------------------------------------------#
        obj_preds = outputs[:, :, 4:5]
        # -----------------------------------------------#
        #   [batch, n_anchors_all, n_cls]
        # -----------------------------------------------#
        cls_preds = outputs[:, :, 5:]

        total_num_anchors = outputs.shape[1]
        # -----------------------------------------------#
        #   x_shifts            [1, n_anchors_all]
        #   y_shifts            [1, n_anchors_all]
        #   expanded_strides    [1, n_anchors_all]
        # -----------------------------------------------#
        x_shifts = torch.cat(x_shifts, 1).type_as(outputs)
        y_shifts = torch.cat(y_shifts, 1).type_as(outputs)
        expanded_strides = torch.cat(expanded_strides, 1).type_as(outputs)

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        # 分別遍歷每張圖片
        for batch_idx in range(outputs.shape[0]):
            num_gt = len(labels[batch_idx])
            if num_gt == 0:
                # 圖中沒有任何gt_box
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                # -----------------------------------------------#
                #   gt_bboxes_per_image     [num_gt, 4]
                #   gt_classes              [num_gt]
                #   bboxes_preds_per_image  [n_anchors_all, 4]
                #   cls_preds_per_image     [n_anchors_all, num_classes]
                #   obj_preds_per_image     [n_anchors_all, 1]
                # -----------------------------------------------#
                gt_bboxes_per_image = labels[batch_idx][..., :4].type_as(outputs)
                gt_classes = labels[batch_idx][..., 4].type_as(outputs)
                bboxes_preds_per_image = bbox_preds[batch_idx]
                cls_preds_per_image = cls_preds[batch_idx]
                obj_preds_per_image = obj_preds[batch_idx]

                # gt_matched_classes = 剩下的anchor對應上的目標類別
                # fg_mask = 通過第一次篩選的anchor為True否則為False
                # pred_ious_this_matching = 剩下的anchor對應上應對上的gt_box的iou
                # matched_gt_inds = 剩下的anchor對應上的gt_box的index
                # num_fg = 最後剩下來的anchor數量
                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = \
                    self.get_assignments(num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes,
                                         bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image,
                                         expanded_strides, x_shifts, y_shifts,)
                torch.cuda.empty_cache()
                # 統計總共有多少anchor
                num_fg += num_fg_img
                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes)\
                                 .float() * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                # 把我們需要的gt_box的預測類別拿出來
                reg_target = gt_bboxes_per_image[matched_gt_inds]
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            # 將obj_target的type轉為cls_target的type之後放入obj_targets的list
            obj_targets.append(obj_target.type(cls_target.type()))
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum()
        loss_obj = (self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)).sum()
        loss_cls = (self.bcewithlog_loss(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)).sum()
        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls
        loss_dict = {
            "iou_loss": reg_weight * loss_iou / num_fg,
            "obj_loss": loss_obj / num_fg,
            "cls_loss": loss_cls / num_fg
        }

        # 恭喜完成損失計算
        return loss / num_fg, loss_dict

    @torch.no_grad()
    def get_assignments(self, num_gt, total_num_anchors, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image,
                        cls_preds_per_image, obj_preds_per_image, expanded_strides, x_shifts, y_shifts):
        """
        :param num_gt: 圖片中gt_box數量
        :param total_num_anchors: 所有yolo head拿到了多少個匡
        :param gt_bboxes_per_image: 圖中的gt_box座標位置(center_x, center_y, w, h)絕對座標
        :param gt_classes: 圖中的gt_box對應上正確的分類類別(1)
        :param bboxes_preds_per_image: 所有預測的框框(center_x, center_y, w, h)絕對座標
        :param cls_preds_per_image: 所有預測匡對於每個類別的預測值(num_classes)
        :param obj_preds_per_image: 所有預測匡對於匡的置信度
        :param expanded_strides: 特徵圖對於原圖的縮放比例
        :param x_shifts: x座標拉成一列
        :param y_shifts: y座標拉成一列
        :return:
        """
        # 已看過
        # 兩個絕對座標都是在同一張圖上了
        # -------------------------------------------------------#
        #   fg_mask                 [n_anchors_all]
        #   is_in_boxes_and_center  [num_gt, len(fg_mask)]
        # -------------------------------------------------------#
        # fg_mask = 每個anchor不是True就是False表示有沒有對應上gt_box
        # is_in_boxes_and_center = 每個gt_box內有哪些可用的anchor
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts,
                                                                 y_shifts, total_num_anchors, num_gt)

        # -------------------------------------------------------#
        #   fg_mask                 [n_anchors_all]
        #   bboxes_preds_per_image  [fg_mask, 4]
        #   cls_preds_              [fg_mask, num_classes]
        #   obj_preds_              [fg_mask, 1]
        # -------------------------------------------------------#
        # 拿出過濾後的預測anchor box
        # 先把有用的anchor拿出來，沒有用的就丟棄了
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        # -------------------------------------------------------#
        #   pair_wise_ious      [num_gt, fg_mask]
        # -------------------------------------------------------#
        # 所有存留下來的anchors跟所有的gt_box做iou計算
        pair_wise_ious = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # -------------------------------------------------------#
        #   cls_preds_          [num_gt, fg_mask, num_classes]
        #   gt_cls_per_image    [num_gt, fg_mask, num_classes]
        # -------------------------------------------------------#
        if self.fp16:
            # 使用amp
            with torch.cuda.amp.autocast(enabled=False):
                # cls_preds_ = 分類概率乘上目標置信度得到最後目標分類分數
                cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1)\
                                 .sigmoid_() * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                # 將gt_class轉成one_hot的格式
                gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes)\
                    .float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
                # binary_cross_entropy根據官網需要將target轉成one_hot格式
                # input與target需要相同的shape
                pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none")\
                    .sum(-1)
        else:
            cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1)\
                             .sigmoid_() * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes)\
                .float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
            pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
            del cls_preds_

        # 神奇計算
        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center).float()

        # num_fg = 最後剩下來的anchor數量
        # gt_matched_classes = 剩下的anchor對應上的目標類別
        # pred_ious_this_matching = 剩下的anchor對應上應對上的gt_box的iou
        # matched_gt_inds = 剩下的anchor對應上的gt_box的index
        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds = \
            self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def bboxes_iou(self, bboxes_a, bboxes_b, xyxy=True):
        # 已看過
        if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
            # 檢測shape有沒有錯誤
            raise IndexError

        if xyxy:
            tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
            br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
            area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
            area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
        else:
            tl = torch.max(
                (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
            )
            br = torch.min(
                (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
            )

            area_a = torch.prod(bboxes_a[:, 2:], 1)
            area_b = torch.prod(bboxes_b[:, 2:], 1)
        en = (tl < br).type(tl.type()).prod(dim=2)
        area_i = torch.prod(br - tl, 2) * en
        return area_i / (area_a[:, None] + area_b - area_i)

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt,
                          center_radius=2.5):
        # 已看過
        # -------------------------------------------------------#
        #   expanded_strides_per_image  [n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        #   x_centers_per_image         [num_gt, n_anchors_all]
        # -------------------------------------------------------#
        # expanded_strides[0] = 把batch_size維度拿掉，可以獲得特徵圖對應原圖得縮放比例
        expanded_strides_per_image = expanded_strides[0]
        # repeat(num_gt, 1)去讓每個anchor都可以去跟gt_box做計算
        # x_shifts, y_shifts是根據特徵圖生成的所以這裡我們需要乘上stride映射到輸入圖像中
        # + 0.5就是從左上角移動到中心點，這是根據論文的
        x_centers_per_image = ((x_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        y_centers_per_image = ((y_shifts[0] + 0.5) * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)

        # -------------------------------------------------------#
        #   gt_bboxes_per_image_x       [num_gt, n_anchors_all]
        # -------------------------------------------------------#
        # gt_bboxes_per_image[l, r, t, b] = [x_left, x_right, y_top, y_bottom]
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2]).\
            unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2]).\
            unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3]).\
            unsqueeze(1).repeat(1, total_num_anchors)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3]).\
            unsqueeze(1).repeat(1, total_num_anchors)

        # -------------------------------------------------------#
        #   bbox_deltas     [num_gt, n_anchors_all, 4]
        # -------------------------------------------------------#
        # b_l, b_r, b_t, b_b = 特徵圖上每個框框中心到每個gt_box四個邊的距離
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        # 在第二維度上做堆疊
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        # -------------------------------------------------------#
        #   is_in_boxes     [num_gt, n_anchors_all] (True, False)
        #   is_in_boxes_all [n_anchors_all] (True, False)
        # -------------------------------------------------------#
        # 過濾掉點不在gt_box中的情況，如果點不在框框中不管怎麼調都無法match
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        # is_in_boxes_all看這個anchor是否至少匹配上一個gt_box的mask
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0

        # 我們不會取所有在gt_box內的點，我們只會取在gt_box內且在中心點附近的幾個點
        # gt_bboxes_per_image[l, r, t, b] = [x_left, x_right, y_top, y_bottom]
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors)\
                                - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(1, total_num_anchors)\
                                + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors)\
                                - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(1, total_num_anchors)\
                                + center_radius * expanded_strides_per_image.unsqueeze(0)

        # -------------------------------------------------------#
        #   center_deltas   [num_gt, n_anchors_all, 4]
        # -------------------------------------------------------#
        # c_l, c_r, c_t, c_b =  特徵圖上每個框框中心到一定距離內邊匡的長度
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)

        # -------------------------------------------------------#
        #   is_in_centers       [num_gt, n_anchors_all] (True, False)
        #   is_in_centers_all   [n_anchors_all] (True, False)
        # -------------------------------------------------------#
        # 過濾掉點不在center一定範圍內中的情況，如果點不在框框中不管怎麼調都無法match
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        # is_in_boxes_all看這個anchor是否至少匹配上一個gt_box的mask
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # -------------------------------------------------------#
        #   is_in_boxes_anchor      [n_anchors_all] (True, False)
        #   is_in_boxes_and_center  [num_gt, is_in_boxes_anchor]
        # -------------------------------------------------------#
        # 用maks過濾anchor，同時也可以知道每個anchor可以對應上哪些gt_box
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        # 留下的anchors中每個anchors跟每個gt_box的關係，如果他可以對應上那個gt_box就會是True否則是False
        is_in_boxes_and_center = is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # 已看過
        # 細挑選正樣本
        # 忘記了就去看看吧
        # https://zhuanlan.zhihu.com/p/437931011
        # -------------------------------------------------------#
        #   cost                [num_gt, fg_mask]
        #   pair_wise_ious      [num_gt, fg_mask]
        #   gt_classes          [num_gt]        
        #   fg_mask             [n_anchors_all]
        # -------------------------------------------------------#
        #   matching_matrix     [num_gt, fg_mask]
        # -------------------------------------------------------#
        matching_matrix = torch.zeros_like(cost)

        # ------------------------------------------------------------#
        #   选取iou最大的n_candidate_k个点
        #   然后求和，判断应该有多少点用于该框预测
        #   topk_ious           [num_gt, n_candidate_k]
        #   dynamic_ks          [num_gt]
        #   matching_matrix     [num_gt, fg_mask]
        # ------------------------------------------------------------#
        n_candidate_k = min(10, pair_wise_ious.size(1))
        # topk_ious會是該gt_box對應上前n_candidate_k個anchors的iou值
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        # 把每個gt_box對應上前n_candidate_k個anchros的iou值全部加在一起，如果小於1會自動補到1
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)

        for gt_idx in range(num_gt):
            # ------------------------------------------------------------#
            #   给每个真实框选取最小的动态k个点
            # ------------------------------------------------------------#
            # 每個gt_box找出k個cost最小的index
            # 拿cost小的來當作正樣本
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            # 在matching_matrix中把對應位置設定成1.0
            matching_matrix[gt_idx][pos_idx] = 1.0
        del topk_ious, dynamic_ks, pos_idx

        # ------------------------------------------------------------#
        #   anchor_matching_gt  [fg_mask]
        # ------------------------------------------------------------#
        # 過濾掉共用的匡，一個anchor匡只能對應上一個gt_box
        anchor_matching_gt = matching_matrix.sum(0)
        # 我們在dim=0地方做求和就可以得到一個anchor到底對應上了多少個gt_box
        # 所以會看到下面寫的，當這個值大於1時表示發生了一個anchor對應上多個gt_box
        if (anchor_matching_gt > 1).sum() > 0:
            # ------------------------------------------------------------#
            #   当某一个特征点指向多个真实框的时候
            #   选取cost最小的真实框。
            # ------------------------------------------------------------#
            # 找到那些行有發生問題，找到在cost中該行最小的index
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            # 先把該行所有的1變成0
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            # 再把該行最小cost的地方填為1
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        # matching_matrix [num_gt, fg_mask]
        # 如果該gt_box有對應上的anchor就會在該位置是1否則為0
        # ------------------------------------------------------------#
        #   fg_mask_inboxes  [fg_mask]
        #   num_fg为正样本的特征点个数
        # ------------------------------------------------------------#
        # 這裡是二次篩選後還是有對應gt_box的anchor
        # fg_mask_inboxes真的有對應上gt_box的anchor
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        # 還活著的anchor數量
        num_fg = fg_mask_inboxes.sum().item()

        # ------------------------------------------------------------#
        #   对fg_mask进行更新
        # ------------------------------------------------------------#
        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        # ------------------------------------------------------------#
        #   获得特征点对应的物品种类
        # ------------------------------------------------------------#
        # 每個有效anchor最後對應上的gt_box的index
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds


def is_parallel(model):
    # Returns True if models is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a models: returns single-GPU models if models is of type DP or DDP
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    # 由train.py實例化
    # 總之就是一種當前權重的調整會跟上一次的比較有關聯跟更久之前的比較沒關聯
    # q 可能是0.1，C就是變化量
    # W(t) = (1 - q) * C(t-1) + q * C(t)
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the models state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # 裡面的東西我就不詳細看了，反正就是反向傳遞的時候的小trick
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        # if next(models.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()  # models state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


def weights_init(net, init_type='normal', init_gain=0.02):
    # 已看過
    # 就是初始化網路內部的權重，這個是在沒有加載任何預訓練權重時的預設權重
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    # 已看過
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        # cos學習率調整
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 +
                math.cos(math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        # 比較傳統的方式調整
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    # 回傳一個function每次要更新學習率的時候就會給這個function當前的epoch，上面的兩個函數對應的iters位置
    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    # 已看過
    # train.py調用
    lr = lr_scheduler_func(epoch)
    # 更改當前學習率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
