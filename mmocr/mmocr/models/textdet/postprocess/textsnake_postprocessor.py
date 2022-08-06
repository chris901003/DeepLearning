# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize

from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor
from .utils import centralize, fill_hole, merge_disks


@POSTPROCESSOR.register_module()
class TextSnakePostprocessor(BasePostprocessor):
    """Decoding predictions of TextSnake to instances. This was partially
    adapted from https://github.com/princewang1994/TextSnake.pytorch.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_text_region_confidence (float): The confidence threshold of text
            region in TextSnake.
        min_center_region_confidence (float): The confidence threshold of text
            center region in TextSnake.
        min_center_area (int): The minimal text center region area.
        disk_overlap_thr (float): The radius overlap threshold for merging
            disks.
        radius_shrink_ratio (float): The shrink ratio of ordered disks radii.
    """

    def __init__(self,
                 text_repr_type='poly',
                 min_text_region_confidence=0.6,
                 min_center_region_confidence=0.2,
                 min_center_area=30,
                 disk_overlap_thr=0.03,
                 radius_shrink_ratio=1.03,
                 **kwargs):
        """ 已看過，TextSnake的後處理初始化函數
        Args:
            text_repr_type: 輸出匡選匡的方式，這裡預設會是多邊形
            min_text_region_confidence: 一個文字團的平均置信度閾值
            min_center_region_confidence: 一個文字團中心地區的平均置信度閾值
            min_center_area: 一個文字團的中心地區最小面積
            disk_overlap_thr: 圓盤重疊的閾值
            radius_shrink_ratio: 半徑縮小率
        """
        # 繼承自BasePostprocessor，對繼承對象進行初始化
        super().__init__(text_repr_type)
        assert text_repr_type == 'poly'
        # 保存傳入的參數
        self.min_text_region_confidence = min_text_region_confidence
        self.min_center_region_confidence = min_center_region_confidence
        self.min_center_area = min_center_area
        self.disk_overlap_thr = disk_overlap_thr
        self.radius_shrink_ratio = radius_shrink_ratio

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(C, H, W)`.

        Returns:
            list[list[float]]: The instance boundary and its confidence.
        """
        # 已看過，獲取預測出來的標註匡資訊
        # preds = 預測圖，tensor shape [channel=5, height, width]

        # 檢查傳入的preds是否合法
        assert preds.dim() == 3

        # channel=0是預測是否為文字的置信度圖，channel=1是預測文字中心區的置信度圖，所以這裡需要經過sigmoid將值控制在[0, 1]之間
        preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
        # 將preds從tensor轉成ndarray型態
        preds = preds.detach().cpu().numpy()

        # 取出preds[0]部分，這個會是預測是否為文字的置信度圖，ndarray shape [height, width]
        pred_text_score = preds[0]
        # 構建文字的mask，當置信度分數小於設定閾值時該地方會是False，其他地方會是True
        pred_text_mask = pred_text_score > self.min_text_region_confidence
        # 將文字中心置信度分數提取出來，這裡會將text_score與其相乘
        pred_center_score = preds[1] * pred_text_score
        # 獲取中心mask，當center的置信度小於閾值時該地方會是False，其他地方會是True
        pred_center_mask = \
            pred_center_score > self.min_center_region_confidence
        # 獲取sin的預測圖
        pred_sin = preds[2]
        # 獲取cos的預測圖
        pred_cos = preds[3]
        # 獲取radius的預測圖
        pred_radius = preds[4]
        # 獲取mask的shape = [height, width]
        mask_sz = pred_text_mask.shape

        # 獲取sin與cos的放大倍率，這裡與訓練時操作相同
        scale = np.sqrt(1.0 / (pred_sin**2 + pred_cos**2 + 1e-8))
        # 將sin與cos乘上放大倍率獲取最後的sin與cos
        pred_sin = pred_sin * scale
        pred_cos = pred_cos * scale

        # 獲取中心部分的mask，ndarray shape [height, width]，在中心部分會是1非中心部分會是0
        pred_center_mask = fill_hole(pred_center_mask).astype(np.uint8)
        # 透過findContours獲取輪廓，center_contours會是tuple(ndarray) ndarray shape [points, 1, 2]
        # 表示文字中心的輪廓的多個點的座標
        center_contours, _ = cv2.findContours(pred_center_mask, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)

        # 最終邊匡的座標點
        boundaries = []
        # 遍歷文字中心的輪廓匡
        for contour in center_contours:
            if cv2.contourArea(contour) < self.min_center_area:
                # 如果匡出的面積小於設定的閾值就會跳過該中心輪廓
                continue
            # 獲取全為0且shape與mask_sz相同的ndarray，shape [height, width]
            instance_center_mask = np.zeros(mask_sz, dtype=np.uint8)
            # 使用cv2的drawContours畫出中心文字部分
            # (要被標註的圖像, 標註輪廓, -1=畫出所有傳入標註匡, 填充的值, -1=使用填充方式，將輪廓中的範圍都填充起來)
            # 當前遍歷到的文字中心區塊會被填充為1
            cv2.drawContours(instance_center_mask, [contour], -1, 1, -1)
            # 將填充好的instance_center_mask放入到skeletonize當中
            # skeletonize = 骨架提取，二值圖像細緻化。這種算法能將一個連通區域細化成一個像素的寬度，用於特徵提取和目標拓撲表示。
            # 如果對於結果有保持疑問可以畫出一個矩形看結果會是如何
            skeleton = skeletonize(instance_center_mask)
            # 獲取skeleton哪些座標的值大於0，shape ndarray [points, 2]
            skeleton_yx = np.argwhere(skeleton > 0)
            # 將(x, y)提取出來，row會是y，col會是x，ndarray shape [points]
            y, x = skeleton_yx[:, 0], skeleton_yx[:, 1]
            # 獲取指定座標的預測cos與sin與radius的值，ndarray shape [points, 1]
            cos = pred_cos[y, x].reshape((-1, 1))
            sin = pred_sin[y, x].reshape((-1, 1))
            radius = pred_radius[y, x].reshape((-1, 1))

            # 獲取中心線，ndarray shape [points, 2]
            # 將骨架點網上頂到合法點的最上限與骨架點頂到最下限合法點的平均值，就會是最後的中心線
            center_line_yx = centralize(skeleton_yx, cos, -sin, radius,
                                        instance_center_mask)
            # 將x與y分別提取出來
            y, x = center_line_yx[:, 0], center_line_yx[:, 1]
            # 將指定點的預測radius取出來並且乘上radius_shrink_ratio值，ndarray shape [points, 1]
            radius = (pred_radius[y, x] * self.radius_shrink_ratio).reshape(
                (-1, 1))
            # 將指定點的預測文本中心置信度分數提取出來，ndarray shape [points, 1]
            score = pred_center_score[y, x].reshape((-1, 1))
            # np.fliplr = [[1, 2], [3, 4]] -> [[2, 1], [4, 3]]
            # instance_disks = ndarray shape [points, 4]
            instance_disks = np.hstack(
                [np.fliplr(center_line_yx), radius, score])
            # 通過merge_disks進行disk融合，會將比較接近的disk進行融合，這樣最後結果不會有一堆圓圈
            instance_disks = merge_disks(instance_disks, self.disk_overlap_thr)

            # 構建一個全為0的ndarray shape [height, width]
            instance_mask = np.zeros(mask_sz, dtype=np.uint8)
            # 遍歷所有的圓盤資訊
            for x, y, radius, score in instance_disks:
                if radius > 1:
                    # 當半徑大於1就會到這裡，將指定的位置填充為1
                    cv2.circle(instance_mask, (int(x), int(y)), int(radius), 1,
                               -1)
            # contours = 圖像當中連通塊的最小外接矩形的點，ndarray [points, 1, 2]
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # 獲取有mask到的地方的置信度總和之後再算出平均置信度值
            score = np.sum(instance_mask * pred_text_score) / (
                np.sum(instance_mask) + 1e-8)
            if (len(contours) > 0 and cv2.contourArea(contours[0]) > 0
                    and contours[0].size > 8):
                # 如果面積大於0且點數超過8就會進來
                # boundary shape = [points * 2]
                boundary = contours[0].flatten().tolist()
                # 最後在後端加上預測的平均置信度分數就可以了
                boundaries.append(boundary + [score])

        return boundaries
