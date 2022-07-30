# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core import BitmapMasks

from mmocr.models.builder import LOSSES
from mmocr.utils import check_argument
from . import PANLoss


@LOSSES.register_module()
class PSELoss(PANLoss):
    r"""The class for implementing PSENet loss. This is partially adapted from
    https://github.com/whai362/PSENet.

    PSENet: `Shape Robust Text Detection with
    Progressive Scale Expansion Network <https://arxiv.org/abs/1806.02559>`_.

    Args:
        alpha (float): Text loss coefficient, and :math:`1-\alpha` is the
            kernel loss coefficient.
        ohem_ratio (float): The negative/positive ratio in ohem.
        reduction (str): The way to reduce the loss. Available options are
            "mean" and "sum".
    """

    def __init__(self,
                 alpha=0.7,
                 ohem_ratio=3,
                 reduction='mean',
                 kernel_sample_type='adaptive'):
        """ 已看過，PSENet專用的損失計算方式
        Args:
            alpha: 整個文本的權重，(1-alpha)會是縮放後文本的權重
            ohem_ratio: 在ohem當中正負樣本的比例
            reduction: 計算後的數字處理
            kernel_sample_type:
        """

        # 繼承於PANLoss，對繼承對象進行初始化
        super().__init__()
        # 對於減少loss的方式只有兩種mean或是sum
        assert reduction in ['mean', 'sum'
                             ], "reduction must be either of ['mean','sum']"
        # 將傳入的資料進行保存
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.kernel_sample_type = kernel_sample_type

    def forward(self, score_maps, downsample_ratio, gt_kernels, gt_mask):
        """Compute PSENet loss.

        Args:
            score_maps (tensor): The output tensor with size of Nx6xHxW.
            downsample_ratio (float): The downsample ratio between score_maps
                and the input img.
            gt_kernels (list[BitmapMasks]): The kernel list with each element
                being the text kernel mask for one img.
            gt_mask (list[BitmapMasks]): The effective mask list
                with each element being the effective mask for one img.

        Returns:
            dict:  A loss dict with ``loss_text`` and ``loss_kernel``.
        """
        # 已看過，計算PSENet的損失值
        # score_maps = 預測圖，tensor shape [batch_size, layers=7, height, width]
        # downsample_ratio = 下採樣倍率，這裡會在[0, 1]之間
        # gt_kernels = 不同減縮比例的標註圖像
        # gt_mask = 正樣本區域的mask

        # 確認輸入的型態是否正確
        assert check_argument.is_type_list(gt_kernels, BitmapMasks)
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        assert isinstance(downsample_ratio, float)
        # 保存losses的空間
        losses = []

        # 預測文字特徵圖，shape [batch_size, height, width]
        pred_texts = score_maps[:, 0, :, :]
        # 預測的核心，shape [batch_size, layers-1=6, height, width]
        pred_kernels = score_maps[:, 1:, :, :]
        # feature_sz = (batch_size, channel, height, width)
        feature_sz = score_maps.size()

        # 進行masks縮放調整，會將maks大小調整到與特徵圖相同
        gt_kernels = [item.rescale(downsample_ratio) for item in gt_kernels]
        # 將bitmasks轉成tensor，gt_kernels=list[tensor]，list長度就會是多少放大層，tensor shape [batch_size, height, width]
        gt_kernels = self.bitmasks2tensor(gt_kernels, feature_sz[2:])
        # 將tensor轉到設備上面
        gt_kernels = [item.to(score_maps.device) for item in gt_kernels]

        # 同樣將gt_mask進行大小調整，同樣會調整到與特徵圖相同
        gt_mask = [item.rescale(downsample_ratio) for item in gt_mask]
        # 將bitmask轉成tensor格式，list[tensor]，list長度只會有1，tensor shape [batch_size, height, width]
        gt_mask = self.bitmasks2tensor(gt_mask, feature_sz[2:])
        # 將gt_mask轉到設備上
        gt_mask = [item.to(score_maps.device) for item in gt_mask]

        # compute text loss，計算text損失
        # sampled_masks_text = (大於threshold或是有被標記為文字)且(沒有被ignore標注到的地方)設定為True，否則為False
        # shape [batch_size, height, width]
        sampled_masks_text = self.ohem_batch(pred_texts.detach(),
                                             gt_kernels[0], gt_mask[0])
        # 將資料傳入到dice_loss_with_logits當中，計算損失值，這裡計算的是dice損失，shape tensor [batch_size]
        loss_texts = self.dice_loss_with_logits(pred_texts, gt_kernels[0],
                                                sampled_masks_text)
        # 將dice損失乘上權重alpha
        losses.append(self.alpha * loss_texts)

        # compute kernel loss
        if self.kernel_sample_type == 'hard':
            sampled_masks_kernel = (gt_kernels[0] > 0.5).float() * (
                gt_mask[0].float())
        elif self.kernel_sample_type == 'adaptive':
            # 如果kernel_sample_type為adaptive就會到這裡
            # 如果pred_texts大於0且gt_mask為1的地方就會是1，否則就會是0，shape [batch_size, height, width]
            sampled_masks_kernel = (pred_texts > 0).float() * (
                gt_mask[0].float())
        else:
            # 其他狀況就會直接報錯
            raise NotImplementedError

        # 獲取總共有多少pred_kernels
        num_kernel = pred_kernels.shape[1]
        # 檢查長度是否相同
        assert num_kernel == len(gt_kernels) - 1
        # 損失列表
        loss_list = []
        # 遍歷所有kernel
        for idx in range(num_kernel):
            # 計算每個kernel的dice損失
            loss_kernels = self.dice_loss_with_logits(
                pred_kernels[:, idx, :, :], gt_kernels[1 + idx],
                sampled_masks_kernel)
            # 將損失傳入進去
            loss_list.append(loss_kernels)

        # 計算平均損失，這裡的權重會是1-alpha，可以到論文上面看
        losses.append((1 - self.alpha) * sum(loss_list) / len(loss_list))

        if self.reduction == 'mean':
            # 如果使用mean去減少loss
            losses = [item.mean() for item in losses]
        elif self.reduction == 'sum':
            # 只進行加總
            losses = [item.sum() for item in losses]
        else:
            # 其他就會報錯
            raise NotImplementedError
        # 將損失計算結果用dict包裝後返回
        results = dict(loss_text=losses[0], loss_kernel=losses[1])
        return results
