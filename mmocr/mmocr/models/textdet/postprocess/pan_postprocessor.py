# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
import torch
from mmcv.ops import pixel_group

from mmocr.core import points2boundary
from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor


@POSTPROCESSOR.register_module()
class PANPostprocessor(BasePostprocessor):
    """Convert scores to quadrangles via post processing in PANet. This is
    partially adapted from https://github.com/WenmuZhou/PAN.pytorch.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_text_confidence (float): The minimal text confidence.
        min_kernel_confidence (float): The minimal kernel confidence.
        min_text_avg_confidence (float): The minimal text average confidence.
        min_text_area (int): The minimal text instance region area.
    """

    def __init__(self,
                 text_repr_type='poly',
                 min_text_confidence=0.5,
                 min_kernel_confidence=0.5,
                 min_text_avg_confidence=0.85,
                 min_text_area=16,
                 **kwargs):
        """ 已看過，PAN模型的後處理部分
        Args:
            text_repr_type: 匡選的方式，可以是矩形或是多邊形
            min_text_confidence: 最小的識別文字的置信度
            min_kernel_confidence: 最小kernel的置信度
            min_text_avg_confidence: 一個連通文字區塊的平均智信度
            min_text_area: 最小的匡選面積大小
        """
        # 繼承自BasePostprocessor，對繼承對象進行初始化
        super().__init__(text_repr_type)

        # 保存傳入的參數
        self.min_text_confidence = min_text_confidence
        self.min_kernel_confidence = min_kernel_confidence
        self.min_text_avg_confidence = min_text_avg_confidence
        self.min_text_area = min_text_area

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(C, H, W)`.

        Returns:
            list[list[float]]: The instance boundary and its confidence.
        """
        # 已看過，PANet後處理方式
        # preds = 預測結果，tensor shape [channel=6, height, width]
        assert preds.dim() == 3

        # preds前面兩個tensor分別表示的是，第一個為預測為文字區域的置信度，第二個為文字團的中心線
        # 所以這兩個部分我們需要透過sigmoid來將值控制在[0, 1]之間
        preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
        # 將preds轉成ndarray格式
        preds = preds.detach().cpu().numpy()

        # 將預測是否為文字區的型態轉成float32格式
        text_score = preds[0].astype(np.float32)
        # 將置信度低於閾值的部分設定為False，其他地方為True，ndarray shape [height, width]
        text = preds[0] > self.min_text_confidence
        # 將kernel置信度大於kernel閾值且該點文字置信度也大於閾值的地方設定為True否則為False，ndarray shape [height, width]
        kernel = (preds[1] > self.min_kernel_confidence) * text
        # 將剩下的相似度向量的通道重新排列，[channel=4, height, width] -> [height, width, channel=4]
        embeddings = preds[2:].transpose((1, 2, 0))  # (h, w, 4)

        # 使用cv2的connectedComponents將連通的地方用相同的index標註，每個連通塊會有自己的index進行標注
        # region_num = 有多少個連通塊，int
        # labels = 連通圖不同連通塊會用不同index表示，ndarray shape [height, width]
        region_num, labels = cv2.connectedComponents(
            kernel.astype(np.uint8), connectivity=4)
        # 透過findContours獲取邊緣座標，這裡傳入的kernel值都乘上255可以有更大的反差
        # contours = 會是一個list當中的資料會是ndarray裡面存了輪廓座標，ndarray shape [points, 1, 2]
        contours, _ = cv2.findContours((kernel * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # 構建ndarray shape [height, width]且全為0
        kernel_contours = np.zeros(text.shape, dtype='uint8')
        # 透過drawContours將img以及輪廓座標放入，-1表示所有輪廓，255表示畫線的顏色
        cv2.drawContours(kernel_contours, contours, -1, 255)
        # text_points = list[list]，第一個list會是總共有多少個文字團，第二個list會是一個文字團有哪些座標點
        text_points = pixel_group(text_score, text, embeddings, labels,
                                  kernel_contours, region_num,
                                  self.min_text_avg_confidence)

        # 最終標註匡的範圍
        boundaries = []
        # 遍歷所有標註團
        for text_point in text_points:
            # 獲取標註團置信度
            text_confidence = text_point[0]
            # 獲取標註團的座標
            text_point = text_point[2:]
            # 將座標變成[points, 2]shape
            text_point = np.array(text_point, dtype=int).reshape(-1, 2)
            # 文字團面積就會是有多少點
            area = text_point.shape[0]

            # 檢查是否合法，不合法就會直接continue
            if not self.is_valid_instance(area, text_confidence,
                                          self.min_text_area,
                                          self.min_text_avg_confidence):
                continue

            # 透過points2boundary將標註點用最小外接矩形包起來
            vertices_confidence = points2boundary(text_point,
                                                  self.text_repr_type,
                                                  text_confidence)
            if vertices_confidence is not None:
                boundaries.append(vertices_confidence)

        return boundaries
