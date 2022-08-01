# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import numpy as np
import torch
from mmcv.ops import contour_expand

from mmocr.core import points2boundary
from mmocr.models.builder import POSTPROCESSOR
from .base_postprocessor import BasePostprocessor


@POSTPROCESSOR.register_module()
class PSEPostprocessor(BasePostprocessor):
    """Decoding predictions of PSENet to instances. This is partially adapted
    from https://github.com/whai362/PSENet.

    Args:
        text_repr_type (str): The boundary encoding type 'poly' or 'quad'.
        min_kernel_confidence (float): The minimal kernel confidence.
        min_text_avg_confidence (float): The minimal text average confidence.
        min_kernel_area (int): The minimal text kernel area.
        min_text_area (int): The minimal text instance region area.
    """

    def __init__(self,
                 text_repr_type='poly',
                 min_kernel_confidence=0.5,
                 min_text_avg_confidence=0.85,
                 min_kernel_area=0,
                 min_text_area=16,
                 **kwargs):
        """ 已看過，對於PSENet的預測進行解碼，也就是後處理部分
        Args:
            text_repr_type: encoding的邊界
        """
        # 繼承自BasePostprocessor，對繼承對象初始化
        super().__init__(text_repr_type)

        # 檢查一些參數有沒有問題
        assert 0 <= min_kernel_confidence <= 1
        assert 0 <= min_text_avg_confidence <= 1
        assert isinstance(min_kernel_area, int)
        assert isinstance(min_text_area, int)

        # 保存傳入的參數
        self.min_kernel_confidence = min_kernel_confidence
        self.min_text_avg_confidence = min_text_avg_confidence
        self.min_kernel_area = min_kernel_area
        self.min_text_area = min_text_area

    def __call__(self, preds):
        """
        Args:
            preds (Tensor): Prediction map with shape :math:`(C, H, W)`.

        Returns:
            list[list[float]]: The instance boundary and its confidence.
        """
        # 已看過，文字匡選的後處理部分
        # preds = 模型預測的結果，tensor shape [channel, height, width]

        # 檢查preds維度需要是3
        assert preds.dim() == 3

        # 將預測透過sigmoid將值控制在[0, 1]之間，作為認定為文字的置信度
        preds = torch.sigmoid(preds)  # text confidence

        # 這裡我們會將第一個特徵圖作為score，score shape [height, width]
        score = preds[0, :, :]
        # 獲取哪些部分的置信度大於設定的min_kernel_confidence，大於的部分會是True否則就會是False
        masks = preds > self.min_kernel_confidence
        # 獲取文字置信度的mask，shape [height, width]
        text_mask = masks[0, :, :]
        # 獲取kernel的mask，shape [channel=layers, height, width]，需要本身位置置信度大於閾值且text的位置也需要是True
        # text_mask會是最大kernel預測的，當我們在擴大kernel時不會發生原先預測為文字的地方變成沒有文字，所以最後的地方沒有
        # 預測為文字那麼前面的部分也需要是False，不過這樣解釋也是會有點問題，目前先這樣解釋
        kernel_masks = masks[0:, :, :] * text_mask

        # 將置信度分數轉成ndarray型態且是以float32儲存
        score = score.data.cpu().numpy().astype(np.float32)

        # 將kernel的maks變成ndarray型態，沒有被mask會是1否則就會是0
        kernel_masks = kernel_masks.data.cpu().numpy().astype(np.uint8)

        # connectedComponents是可以找出連通的部分將連通的部分用相同的index標注起來，傳入的需要是二值圖，connectivity=設定幾連通
        # 會有回傳兩個值，region_num=有多少個區塊(int)，labels=圖像(ndarray shape [height, width])
        region_num, labels = cv2.connectedComponents(
            kernel_masks[-1], connectivity=4)

        # 將kernel_masks與label與min_kernel_area與region_num傳入
        # labels = list[list]，第一個list長度會是height，第二個list長度會是width
        labels = contour_expand(kernel_masks, labels, self.min_kernel_area,
                                region_num)
        # 轉成ndarray格式，ndarray shape [height, width]
        labels = np.array(labels)
        # 獲取labels當中最大的值就可以知道總共分成多少區塊的文字
        label_num = np.max(labels)
        # 保存文字邊界值
        boundaries = []
        # 遍歷所有的標註文字群
        for i in range(1, label_num + 1):
            # 透過np.where找到等於i的座標位置，會有兩個回傳一個是row另一個是col
            # 使用np.array將回傳的row與col進行打包，tuple(array, array) -> ndarray shape [2, num_pixel]
            # 透過transpose轉換排列方式shape [2, num_pixel] -> [num_pixel, 2]，這樣更容易讀取資料
            # 最後將(row, col)調換順序，變成(col, row)
            points = np.array(np.where(labels == i)).transpose((1, 0))[:, ::-1]
            # 獲取總共有多少個點，也就是所佔面積
            area = points.shape[0]
            # 獲取該地方的平均置信度
            score_instance = np.mean(score[labels == i])
            # 檢查面積是否有大於最小面積以及平均置信度是否大於設定閾值
            if not self.is_valid_instance(area, score_instance,
                                          self.min_text_area,
                                          self.min_text_avg_confidence):
                # 如果條件不滿足就會直接continue跳過
                continue

            # 根據傳入的點座標位置畫出最小外接矩形匡的四個座標，這裡會是[置信度分數, 展平後的座標(8個)]
            vertices_confidence = points2boundary(points, self.text_repr_type,
                                                  score_instance)
            if vertices_confidence is not None:
                # 將結果放到boundaries保存
                boundaries.append(vertices_confidence)

        # list[list]，第一個list長度就會是有多少個匡，第二個list就會是匡的細節內容
        return boundaries
