# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import BitmapMasks
from torch import nn

from mmocr.models.builder import LOSSES
from mmocr.utils import check_argument


@LOSSES.register_module()
class PANLoss(nn.Module):
    """The class for implementing PANet loss. This was partially adapted from
    https://github.com/WenmuZhou/PAN.pytorch.

    PANet: `Efficient and Accurate Arbitrary-
    Shaped Text Detection with Pixel Aggregation Network
    <https://arxiv.org/abs/1908.05900>`_.

    Args:
        alpha (float): The kernel loss coef.
        beta (float): The aggregation and discriminative loss coef.
        delta_aggregation (float): The constant for aggregation loss.
        delta_discrimination (float): The constant for discriminative loss.
        ohem_ratio (float): The negative/positive ratio in ohem.
        reduction (str): The way to reduce the loss.
        speedup_bbox_thr (int):  Speed up if speedup_bbox_thr > 0
            and < bbox num.
    """

    def __init__(self,
                 alpha=0.5,
                 beta=0.25,
                 delta_aggregation=0.5,
                 delta_discrimination=3,
                 ohem_ratio=3,
                 reduction='mean',
                 speedup_bbox_thr=-1):
        super().__init__()
        assert reduction in ['mean', 'sum'], "reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.delta_aggregation = delta_aggregation
        self.delta_discrimination = delta_discrimination
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.speedup_bbox_thr = speedup_bbox_thr

    def bitmasks2tensor(self, bitmasks, target_sz):
        """Convert Bitmasks to tensor.

        Args:
            bitmasks (list[BitmapMasks]): The BitmapMasks list. Each item is
                for one img.
            target_sz (tuple(int, int)): The target tensor of size
                :math:`(H, W)`.

        Returns:
            list[Tensor]: The list of kernel tensors. Each element stands for
            one kernel level.
        """
        # 已看過，將bitmasks轉成tensor格式

        # 檢查格式是否正確
        assert check_argument.is_type_list(bitmasks, BitmapMasks)
        assert isinstance(target_sz, tuple)

        # 獲取batch_size
        batch_size = len(bitmasks)
        # 獲取有多少個mask
        num_masks = len(bitmasks[0])

        results = []

        # 遍歷所有的mask
        for level_inx in range(num_masks):
            kernel = []
            # 遍歷一個batch當中所有的圖像
            for batch_inx in range(batch_size):
                # 將ndarray轉成tensor格式
                mask = torch.from_numpy(bitmasks[batch_inx].masks[level_inx])
                # hxw，獲取特徵圖高寬
                mask_sz = mask.shape
                # left, right, top, bottom，獲取需要padding的大小
                pad = [
                    0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]
                ]
                # 如果需要padding就會進行padding且padding的值全為0
                mask = F.pad(mask, pad, mode='constant', value=0)
                # 將結果保存到kernel當中
                kernel.append(mask)
            # kernel shape = [batch_size, height, width]
            kernel = torch.stack(kernel)
            results.append(kernel)

        # results = list[tensor]，list長度就會是有多少放大層，tensor shape [batch_size, height, width]
        return results

    def forward(self, preds, downsample_ratio, gt_kernels, gt_mask):
        """Compute PANet loss.

        Args:
            preds (Tensor): The output tensor of size :math:`(N, 6, H, W)`.
            downsample_ratio (float): The downsample ratio between preds
                and the input img.
            gt_kernels (list[BitmapMasks]): The kernel list with each element
                being the text kernel mask for one img.
            gt_mask (list[BitmapMasks]): The effective mask list
                with each element being the effective mask for one img.

        Returns:
            dict:  A loss dict with ``loss_text``, ``loss_kernel``,
            ``loss_aggregation`` and ``loss_discrimination``.
        """

        assert check_argument.is_type_list(gt_kernels, BitmapMasks)
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        assert isinstance(downsample_ratio, float)

        pred_texts = preds[:, 0, :, :]
        pred_kernels = preds[:, 1, :, :]
        inst_embed = preds[:, 2:, :, :]
        feature_sz = preds.size()

        mapping = {'gt_kernels': gt_kernels, 'gt_mask': gt_mask}
        gt = {}
        for key, value in mapping.items():
            gt[key] = value
            gt[key] = [item.rescale(downsample_ratio) for item in gt[key]]
            gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            gt[key] = [item.to(preds.device) for item in gt[key]]
        loss_aggrs, loss_discrs = self.aggregation_discrimination_loss(
            gt['gt_kernels'][0], gt['gt_kernels'][1], inst_embed)
        # compute text loss
        sampled_mask = self.ohem_batch(pred_texts.detach(),
                                       gt['gt_kernels'][0], gt['gt_mask'][0])
        loss_texts = self.dice_loss_with_logits(pred_texts,
                                                gt['gt_kernels'][0],
                                                sampled_mask)

        # compute kernel loss

        sampled_masks_kernel = (gt['gt_kernels'][0] > 0.5).float() * (
            gt['gt_mask'][0].float())
        loss_kernels = self.dice_loss_with_logits(pred_kernels,
                                                  gt['gt_kernels'][1],
                                                  sampled_masks_kernel)
        losses = [loss_texts, loss_kernels, loss_aggrs, loss_discrs]
        if self.reduction == 'mean':
            losses = [item.mean() for item in losses]
        elif self.reduction == 'sum':
            losses = [item.sum() for item in losses]
        else:
            raise NotImplementedError

        coefs = [1, self.alpha, self.beta, self.beta]
        losses = [item * scale for item, scale in zip(losses, coefs)]

        results = dict()
        results.update(
            loss_text=losses[0],
            loss_kernel=losses[1],
            loss_aggregation=losses[2],
            loss_discrimination=losses[3])
        return results

    def aggregation_discrimination_loss(self, gt_texts, gt_kernels,
                                        inst_embeds):
        """Compute the aggregation and discrimnative losses.

        Args:
            gt_texts (Tensor): The ground truth text mask of size
                :math:`(N, 1, H, W)`.
            gt_kernels (Tensor): The ground truth text kernel mask of
                size :math:`(N, 1, H, W)`.
            inst_embeds(Tensor): The text instance embedding tensor
                of size :math:`(N, 1, H, W)`.

        Returns:
            (Tensor, Tensor): A tuple of aggregation loss and discriminative
            loss before reduction.
        """

        batch_size = gt_texts.size()[0]
        gt_texts = gt_texts.contiguous().reshape(batch_size, -1)
        gt_kernels = gt_kernels.contiguous().reshape(batch_size, -1)

        assert inst_embeds.shape[1] == 4
        inst_embeds = inst_embeds.contiguous().reshape(batch_size, 4, -1)

        loss_aggrs = []
        loss_discrs = []

        for text, kernel, embed in zip(gt_texts, gt_kernels, inst_embeds):

            # for each image
            text_num = int(text.max().item())
            loss_aggr_img = []
            kernel_avgs = []
            select_num = self.speedup_bbox_thr
            if 0 < select_num < text_num:
                inds = np.random.choice(
                    text_num, select_num, replace=False) + 1
            else:
                inds = range(1, text_num + 1)

            for i in inds:
                # for each text instance
                kernel_i = (kernel == i)  # 0.2ms
                if kernel_i.sum() == 0 or (text == i).sum() == 0:  # 0.2ms
                    continue

                # compute G_Ki in Eq (2)
                avg = embed[:, kernel_i].mean(1)  # 0.5ms
                kernel_avgs.append(avg)

                embed_i = embed[:, text == i]  # 0.6ms
                # ||F(p) - G(K_i)|| - delta_aggregation, shape: nums
                distance = (embed_i - avg.reshape(4, 1)).norm(  # 0.5ms
                    2, dim=0) - self.delta_aggregation
                # compute D(p,K_i) in Eq (2)
                hinge = torch.max(
                    distance,
                    torch.tensor(0, device=distance.device,
                                 dtype=torch.float)).pow(2)

                aggr = torch.log(hinge + 1).mean()
                loss_aggr_img.append(aggr)

            num_inst = len(loss_aggr_img)
            if num_inst > 0:
                loss_aggr_img = torch.stack(loss_aggr_img).mean()
            else:
                loss_aggr_img = torch.tensor(
                    0, device=gt_texts.device, dtype=torch.float)
            loss_aggrs.append(loss_aggr_img)

            loss_discr_img = 0
            for avg_i, avg_j in itertools.combinations(kernel_avgs, 2):
                # delta_discrimination - ||G(K_i) - G(K_j)||
                distance_ij = self.delta_discrimination - (avg_i -
                                                           avg_j).norm(2)
                # D(K_i,K_j)
                D_ij = torch.max(
                    distance_ij,
                    torch.tensor(
                        0, device=distance_ij.device,
                        dtype=torch.float)).pow(2)
                loss_discr_img += torch.log(D_ij + 1)

            if num_inst > 1:
                loss_discr_img /= (num_inst * (num_inst - 1))
            else:
                loss_discr_img = torch.tensor(
                    0, device=gt_texts.device, dtype=torch.float)
            if num_inst == 0:
                warnings.warn('num of instance is 0')
            loss_discrs.append(loss_discr_img)
        return torch.stack(loss_aggrs), torch.stack(loss_discrs)

    def dice_loss_with_logits(self, pred, target, mask):
        """ 已看過，計算dice損失
        Args:
            pred: 預測出來的圖，tensor shape [batch_size, height, width]
            target: 標註內容，沒有文字部分會是0，有文字部分會根據處在的字團有不同數字，shape [batch_size, height, width]
            mask: True表示我們需要的False表示不要的，shape [batch_size, height, width]
        """

        # 設定smooth參數
        smooth = 0.001

        # 將預測值透過sigmoid將值控制在[0, 1]之間
        pred = torch.sigmoid(pred)
        # 將target小於等於0.5的設定成0，不過其實target本身就只會是正數型態，所以不會有作用
        target[target <= 0.5] = 0
        # 將大於0.5的部分全部設定成1
        target[target > 0.5] = 1
        # 調整通道，將pred以及target以及mask的shape [batch_size, height, width] -> [batch_size, height * width]
        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        # 當mask的部分為False時pred該位置就會變成0，否則就是原始值
        pred = pred * mask
        target = target * mask

        # 根據計算dice的公式獲取dice損失
        # https://zhuanlan.zhihu.com/p/348832716
        a = torch.sum(pred * target, 1) + smooth
        b = torch.sum(pred * pred, 1) + smooth
        c = torch.sum(target * target, 1) + smooth
        d = (2 * a) / (b + c)
        # d會是dice值，這個值越大表示契合程度越高，使用1去減就可以作為損失值
        return 1 - d

    def ohem_img(self, text_score, gt_text, gt_mask):
        """Sample the top-k maximal negative samples and all positive samples.

        Args:
            text_score (Tensor): The text score of size :math:`(H, W)`.
            gt_text (Tensor): The ground truth text mask of size
                :math:`(H, W)`.
            gt_mask (Tensor): The effective region mask of size :math:`(H, W)`.

        Returns:
            Tensor: The sampled pixel mask of size :math:`(H, W)`.
        """
        # 已看過，選取前k大的負樣本以及所有的正樣本
        # text_score = text的分數，tensor shape [height, width]
        # gt_text = 有文字的部分會是非0，其他地方為0，非0部分會依據是哪個群的文字會有不同數字，shape [height, width]
        # gt_mask = ignore的部分，1的地方表示沒有ignore的文字，0表示被ignore部分的文字，被ignore的原因可能會是有重疊

        # 檢查一些資料型態
        assert isinstance(text_score, torch.Tensor)
        assert isinstance(gt_text, torch.Tensor)
        assert isinstance(gt_mask, torch.Tensor)
        assert len(text_score.shape) == 2
        assert text_score.shape == gt_text.shape
        assert gt_text.shape == gt_mask.shape

        # 總正樣本數量 = 有被標註的文字格數量 - 有被標註的文字格數量&&有被標註為ignore的網格數量
        pos_num = int(torch.sum(gt_text > 0.5).item()) - int(
            torch.sum((gt_text > 0.5) * (gt_mask <= 0.5)).item())
        # 負樣本數量 = 沒有被標註為文字的部分
        neg_num = int(torch.sum(gt_text <= 0.5).item())
        # 負樣本數量會需要與正樣本成比例關係，否則會導致負樣本過多
        neg_num = int(min(pos_num * self.ohem_ratio, neg_num))

        if pos_num == 0 or neg_num == 0:
            # 如果有發生正樣本或是負樣本為0就會跳出警告
            warnings.warn('pos_num = 0 or neg_num = 0')
            return gt_mask.bool()

        # 獲取沒有被標註成文字部分的text_score
        neg_score = text_score[gt_text <= 0.5]
        # 對neg_score進行排序，這裡我們只會需要排序過後的分數，不需要知道原始index是多少，這裡會是由大到小排序
        neg_score_sorted, _ = torch.sort(neg_score, descending=True)
        # 獲取前neg_num大的部分
        threshold = neg_score_sorted[neg_num - 1]
        # 在tensor.bool時使用(+)表示or，使用(*)表示and
        # 所以這裡會將(大於threshold或是有被標記為文字)且(沒有被ignore標注到的地方)設定為True，否則為False
        sampled_mask = (((text_score >= threshold) + (gt_text > 0.5)) > 0) * (
            gt_mask > 0.5)
        # sampled_mask = tensor.bool shape [height]
        return sampled_mask

    def ohem_batch(self, text_scores, gt_texts, gt_mask):
        """OHEM sampling for a batch of imgs.

        Args:
            text_scores (Tensor): The text scores of size :math:`(H, W)`.
            gt_texts (Tensor): The gt text masks of size :math:`(H, W)`.
            gt_mask (Tensor): The gt effective mask of size :math:`(H, W)`.

        Returns:
            Tensor: The sampled mask of size :math:`(H, W)`.
        """
        # 已看過，使用OHEM計算損失
        # text_scores = 文字分數，tensor shape [batch_size, height, width]
        # gt_text = 標註有文字的部分，tensor shape [batch_size, height, width]
        # gt_mask = 標註哪些部分有被ignore的text，有的部分會是0其他為1

        # 檢查傳入的資料有沒有型態問題
        assert isinstance(text_scores, torch.Tensor)
        assert isinstance(gt_texts, torch.Tensor)
        assert isinstance(gt_mask, torch.Tensor)
        assert len(text_scores.shape) == 3
        assert text_scores.shape == gt_texts.shape
        assert gt_texts.shape == gt_mask.shape

        sampled_masks = []
        # 遍歷batch_size
        for i in range(text_scores.shape[0]):
            # 取出該圖像對應的資料放入到ohem_img當中
            sampled_masks.append(
                self.ohem_img(text_scores[i], gt_texts[i], gt_mask[i]))

        # 將結果進行堆疊，shape [batch_size, height, width]
        # (大於threshold或是有被標記為文字)且(沒有被ignore標注到的地方)設定為True，否則為False
        sampled_masks = torch.stack(sampled_masks)

        return sampled_masks
