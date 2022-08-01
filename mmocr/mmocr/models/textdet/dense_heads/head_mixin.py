# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

from mmocr.models.builder import HEADS, build_loss, build_postprocessor
from mmocr.utils import check_argument


@HEADS.register_module()
class HeadMixin:
    """Base head class for text detection, including loss calcalation and
    postprocess.

    Args:
        loss (dict): Config to build loss.
        postprocessor (dict): Config to build postprocessor.
    """

    def __init__(self, loss, postprocessor):
        """ 已看過，文字檢測頭的基礎類，其中飽含計算以及後處理
        Args:
            loss: 損失計算方式
            postprocessor: 後處理方式
        """

        # 檢查loss與postprocessor需要是dict格式
        assert isinstance(loss, dict)
        assert isinstance(postprocessor, dict)

        # 構建loss實例對象
        self.loss_module = build_loss(loss)
        # 構建後處理實例對象
        self.postprocessor = build_postprocessor(postprocessor)

    def resize_boundary(self, boundaries, scale_factor):
        """Rescale boundaries via scale_factor.

        Args:
            boundaries (list[list[float]]): The boundary list. Each boundary
                has :math:`2k+1` elements with :math:`k>=4`.
            scale_factor (ndarray): The scale factor of size :math:`(4,)`.

        Returns:
            list[list[float]]: The scaled boundaries.
        """
        # 已看過，將boundary進行resize調整
        # boundaries = 文字匡選範圍，list[list]第一個list長度會是匡選數量，第二個list會是一個匡選匡的資訊
        # scale_factor = 縮放比例，ndarray shape [4]，表示高寬需要縮放的大小 (height, width, height, width)

        # 檢查boundaries是否為list[list]格式
        assert check_argument.is_2dlist(boundaries)
        # 檢查scale_factor是否為ndarray格式
        assert isinstance(scale_factor, np.ndarray)
        # 檢查scale_factor是否為[4]
        assert scale_factor.shape[0] == 4

        # 遍歷所有的boundaries
        for b in boundaries:
            # 獲取一個標注匡有多少個資訊
            sz = len(b)
            # 檢查boundary是否合法，後面的True表示我們有加上置信度
            check_argument.valid_boundary(b, True)
            # 最後一個會是置信度，所以不會取到最後一個值
            # 這裡會用到tile將scale_factor調整到可以與b相乘
            b[:sz -
              1] = (np.array(b[:sz - 1]) *
                    (np.tile(scale_factor[:2], int(
                        (sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
        # 將調整好的boundaries回傳
        return boundaries

    def get_boundary(self, score_maps, img_metas, rescale):
        """Compute text boundaries via post processing.

        Args:
            score_maps (Tensor): The text score map.
            img_metas (dict): The image meta info.
            rescale (bool): Rescale boundaries to the original image resolution
                if true, and keep the score_maps resolution if false.

        Returns:
            dict: A dict where boundary results are stored in
            ``boundary_result``.
        """
        # 已看過，計算文字匡選邊界位置，透過後處理
        # score_maps = 預測出來的結果，tensor shape [channel, batch_size, height]
        # img_metas = 當前圖像的詳細資訊
        # rescale = 是否需要進行rescale

        # 檢查輸入的img_metas是否為dict格式
        assert check_argument.is_type_list(img_metas, dict)
        # rescale需要是bool型態
        assert isinstance(rescale, bool)

        # 將score_maps當中維度為1的部分進行壓縮
        score_maps = score_maps.squeeze()
        # boundaries = list[list]，第一個list長度就會是有多少個匡，第二個list就會是匡的細節內容
        boundaries = self.postprocessor(score_maps)

        if rescale:
            # 如果有需要rescale就會到這裡
            boundaries = self.resize_boundary(
                boundaries,
                1.0 / self.downsample_ratio / img_metas[0]['scale_factor'])

        # 將最終調整好的boundaries與filename包裝成dict進行回傳
        results = dict(
            boundary_result=boundaries, filename=img_metas[0]['filename'])

        # 最後將整個results回傳
        return results

    def loss(self, pred_maps, **kwargs):
        """Compute the loss for scene text detection.

        Args:
            pred_maps (Tensor): The input score maps of shape
                :math:`(NxCxHxW)`.

        Returns:
            dict: The dict for losses.
        """
        # 已看過，計算損失，loss會有loss_text與loss_kernel兩種損失
        losses = self.loss_module(pred_maps, self.downsample_ratio, **kwargs)

        return losses
