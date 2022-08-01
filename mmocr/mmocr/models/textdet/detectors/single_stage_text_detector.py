# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmocr.models.builder import DETECTORS
from mmocr.models.common.detectors import SingleStageDetector


@DETECTORS.register_module()
class SingleStageTextDetector(SingleStageDetector):
    """The class for implementing single stage text detector."""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        """ 已看過，文字檢測
        Args:
            backbone: backbone的設定資料
            neck: 將backbone的輸出進行加工，不一定會有
            bbox_head: 將提取出來的特徵進行預測，也就是預測頭
            train_cfg: train相關的設定
            test_cfg: test相關的設定
            pretrained: 預訓練權重相關資料
            init_cfg: 初始化設定方式
        """
        # 繼承自SingleStageDetector，對繼承對象進行初始化
        SingleStageDetector.__init__(self, backbone, neck, bbox_head,
                                     train_cfg, test_cfg, pretrained, init_cfg)

    def forward_train(self, img, img_metas, **kwargs):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # 已看過，向前傳遞流程
        # img = 圖像的tensor格式，shape [batch_size, channel, height, width]
        # img_metas = 圖像的詳細資訊
        # kwargs = 其他額外資訊，可能會有標註圖像資訊

        # 透過extract_feat進行特徵提取，x shape [batch_size, channel, height, width]
        x = self.extract_feat(img)
        # preds shape = [batch_size, channel, height, width]
        preds = self.bbox_head(x)
        # 計算損失，loss會有loss_text與loss_kernel兩種
        losses = self.bbox_head.loss(preds, **kwargs)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """ 已看過，驗證模式下的簡單向前傳遞
        Args:
            img: 圖像的tensor資料shape [batch_size, channel, height, width]
            img_metas: 一個batch的圖像詳細資料
            rescale: 是否需要進行rescale
        """
        # 進行特徵提取，img shape [batch_size, channel, height, width]
        x = self.extract_feat(img)
        # 經過bbox_head進行匡選預測，outs shape [batch_size, channel, height, width]
        outs = self.bbox_head(x)

        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return outs

        if len(img_metas) > 1:
            boundaries = [
                self.bbox_head.get_boundary(*(outs[i].unsqueeze(0)),
                                            [img_metas[i]], rescale)
                for i in range(len(img_metas))
            ]

        else:
            # 透過bbox_head當中get_boundary獲取匡選位置
            boundaries = [
                self.bbox_head.get_boundary(*outs, img_metas, rescale)
            ]

        # boundaries = list[dict]，list長度就會是一個batch的圖像數量
        # dict = {
        #   'filename': 檔案名稱
        #   'boundary_result': list[list]，第一個list長度就會是有多少個匡，第二個list就是匡的詳細資訊
        # }
        return boundaries
