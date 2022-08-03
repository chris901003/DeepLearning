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
        """ 已看過，計算PAN的損失值
        """
        super().__init__()
        # 檢查傳入的reduction是否合法
        assert reduction in ['mean', 'sum'], "reduction must in ['mean','sum']"
        # 將傳入的參數進行保存
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
        # 已看過，PAN的損失計算部分
        # preds = 預測出來的特徵圖，tensor shape [batch_size, channel=6, height, width]
        # downsample_ratio = 特徵圖會是原圖多少倍的大小，也就是需要將原圖縮放多少比例後才能對應上特徵圖
        # gt_kernels = 對原始標註圖像進行縮放後的標註圖像，如果不是文字部分會是0，文字部分會依據所處的連通塊有不同的index
        # gt_mask = 如果為0的地方就是需要忽略掉的地方，因為該地方標示不明確，例如有重疊部分，1表示有需要計算損失的部分

        # 檢查gt_kernels是否為BitmapMasks實例對象
        assert check_argument.is_type_list(gt_kernels, BitmapMasks)
        # 檢查gt_mask是否為BitmapMasks實例對象
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        # 檢查downsample_ratio是否為float型態
        assert isinstance(downsample_ratio, float)

        # 獲取預測是否為文字的置信度，作用是預測哪些部分是文本(不會去區分是文本的地方是屬於哪個文字團的)
        pred_texts = preds[:, 0, :, :]
        # 獲取預測核，主要是預測文字團的中間線，文字團會從中間線開始往外擴散，所以會需要知道文字團的中心線
        pred_kernels = preds[:, 1, :, :]
        # 其他特徵圖，論文當中是similar_vector，主要是輔助kernel用的，要讓不同文字團的中心線距離越遠越好，同時非中心地方要越近越好
        inst_embed = preds[:, 2:, :, :]
        # feature_sz = (batch_size, channel=6, height, width)
        feature_sz = preds.size()

        # 保存對應關係
        # gt_kernels = 將標註圖像的標註匡往內縮小獲取我們希望預測出來的文字團中心線
        # gt_masks = 哪些部分被標註為模糊地帶，在模糊地帶我們希望不要計算到損失值，所以在模糊地帶會是0其他會是1
        mapping = {'gt_kernels': gt_kernels, 'gt_mask': gt_mask}
        # 構建gt字典
        gt = {}
        # 遍歷mapping當中的key與value
        for key, value in mapping.items():
            # 將key與value存到gt當中
            gt[key] = value
            # 遍歷其中的資料並且使用rescale進行高寬壓縮
            gt[key] = [item.rescale(downsample_ratio) for item in gt[key]]
            # 將ndarray型態轉成tensor格式，同時因為為了要壓成一個batch可能會有圖像需要padding，在這裡也會對標註圖像進行padding
            # 原先在gt當中一個list當中是一張圖像在不同縮放比例的gt，通過bitmask2tensor後變成一個縮放比例下不同圖像的堆疊
            gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            # 將tensor轉到訓練設備上
            gt[key] = [item.to(preds.device) for item in gt[key]]
        # 進行損失計算，這裡會將兩個不同縮放比例的gt_kernels與inst_embed放入，tensor shape [batch_size]
        # loss_arrgs = 像素聚合損失，文本實例與其kernel的loss
        # loss_discrs = 相似性向量損失，不同文本實例的kernel的loss
        # 文本實例 = 在圖像當中一個完整的文字區域，如果有多個區域表示該圖像當中有多段文字
        # kernel = 某個文字區域經過縮放後得到的區域
        # 一個文本實例可以有多個kernel，像是在PSENet當中會有多個kernel預測一個文本實例，在PANet當中只會有一個kernel對應一個文本實例
        # 這裡詳細的loss計算放到note當中
        loss_aggrs, loss_discrs = self.aggregation_discrimination_loss(
            gt['gt_kernels'][0], gt['gt_kernels'][1], inst_embed)
        # compute text loss，計算文字置信度損失
        # 先獲取sample_mask之後才可以計算損失
        sampled_mask = self.ohem_batch(pred_texts.detach(),
                                       gt['gt_kernels'][0], gt['gt_mask'][0])
        loss_texts = self.dice_loss_with_logits(pred_texts,
                                                gt['gt_kernels'][0],
                                                sampled_mask)

        # compute kernel loss，計算kernel的損失，這裡也是會使用dice損失

        # 獲取有標註為文字部分同時沒有被有問題的標註匡到的位置，符合的位置會是1否則就會是0，tensor shape [batch_size, height, width]
        sampled_masks_kernel = (gt['gt_kernels'][0] > 0.5).float() * (
            gt['gt_mask'][0].float())
        loss_kernels = self.dice_loss_with_logits(pred_kernels,
                                                  gt['gt_kernels'][1],
                                                  sampled_masks_kernel)
        # 這裡的loss shape [batch_size]
        losses = [loss_texts, loss_kernels, loss_aggrs, loss_discrs]
        if self.reduction == 'mean':
            # 如果需要透過均值進行融合會到這裡
            losses = [item.mean() for item in losses]
        elif self.reduction == 'sum':
            # 如果是透過加總進行融合會到這裡
            losses = [item.sum() for item in losses]
        else:
            raise NotImplementedError

        # 每種損失的權重
        coefs = [1, self.alpha, self.beta, self.beta]
        # 乘上權重
        losses = [item * scale for item, scale in zip(losses, coefs)]

        # 最後將損失構建成字典型態，每個對應上就會是一個值的tensor
        results = dict()
        results.update(
            loss_text=losses[0],
            loss_kernel=losses[1],
            loss_aggregation=losses[2],
            loss_discrimination=losses[3])
        # 最後回傳損失字典
        return results

    def aggregation_discrimination_loss(self, gt_texts, gt_kernels,
                                        inst_embeds):
        """Compute the aggregation and discriminative losses.

        Args:
            gt_texts (Tensor): The ground truth text mask of size
                :math:`(N, 1, H, W)`.
            gt_kernels (Tensor): The ground truth text kernel mask of
                size :math:`(N, 1, H, W)`.
            inst_embeds(Tensor): The text instance embedding tensor
                of size :math:`(N, 4, H, W)`.

        Returns:
            (Tensor, Tensor): A tuple of aggregation loss and discriminative
            loss before reduction.
        """
        # 已看過，計算聚合損失和判別損失
        # 有關於Ldis以及Lagg的公式可以看:https://zhuanlan.zhihu.com/p/89889661
        # gt_texts = 原始標註圖像，標示0表示不是文字，標示非0表示有文字，tensor shape [batch_size, channel=1, height, width]
        # gt_kernels = 經過縮放後的標註圖像，標示0表示不是文字，標示非0表示有文字，也就是我們希望預測的文字團中心線位置
        # inst_embeds = 預測出來的，tensor shape [batch_size, channel=4, height, width]
        #               每一個pixel的相似度向量，這裡向量深度就是channel深度

        # 獲取batch_size
        batch_size = gt_texts.size()[0]
        # 將高寬維度進行壓縮，shape [batch_size, height * width]
        gt_texts = gt_texts.contiguous().reshape(batch_size, -1)
        # 將高寬維度進行壓縮，shape [batch_size, height * width]
        gt_kernels = gt_kernels.contiguous().reshape(batch_size, -1)

        # 檢查inst_embeds的channel是否為4
        assert inst_embeds.shape[1] == 4
        # 將高寬進行壓縮，shape [batch_size, channel=4, height * width]
        inst_embeds = inst_embeds.contiguous().reshape(batch_size, 4, -1)

        # 損失保存的地方，記錄下每個標註團的損失值，記錄下Lagg以及Ldis
        loss_aggrs = []
        loss_discrs = []

        # 不同圖像分開處理，遍歷長度為batch_size
        for text, kernel, embed in zip(gt_texts, gt_kernels, inst_embeds):

            # for each image
            # 獲取一張圖像當中總共有多少個標註匡
            text_num = int(text.max().item())
            # 用來記錄一個文字團的Lagg
            loss_aggr_img = []
            # 用來保存一個文字團預測出來的平均相似度向量
            kernel_avgs = []
            # 用來加速debug用的，如果設定只會隨機挑選指定數量的標註匡
            select_num = self.speedup_bbox_thr
            if 0 < select_num < text_num:
                inds = np.random.choice(
                    text_num, select_num, replace=False) + 1
            else:
                # 獲取需要遍歷的標註匡index
                inds = range(1, text_num + 1)

            # 遍歷所有的標註匡
            for i in inds:
                # for each text instance
                # 找出kernel當中指定的標註團，處於該團的地方會是True其他地方會是False
                kernel_i = (kernel == i)  # 0.2ms
                if kernel_i.sum() == 0 or (text == i).sum() == 0:  # 0.2ms
                    # 如過在縮放後的標註面積為0或是原始標註面積為0就會直接continue
                    continue

                # compute G_Ki in Eq (2)
                # 獲取該文字團在預測圖當中的平均置信度，avg shape [4]，也就是獲取該範圍內的預測平均相似度向量
                avg = embed[:, kernel_i].mean(1)  # 0.5ms
                # 將預測結果記錄到kernel_avgs
                kernel_avgs.append(avg)

                # 將被標註為文字部分的預測數字取出，embed_i shape [4, points]
                embed_i = embed[:, text == i]  # 0.6ms
                # ||F(p) - G(K_i)|| - delta_aggregation, shape: nums
                # 將取出的預測值減去縮放後的預測圖均值，之後再透過標準化對於channel=4的地方，最後減去delta_aggregation
                distance = (embed_i - avg.reshape(4, 1)).norm(  # 0.5ms
                    2, dim=0) - self.delta_aggregation
                # compute D(p,K_i) in Eq (2)
                # 獲取distance大於0的地方並且取平方，小於0的地方就直接是0
                hinge = torch.max(
                    distance,
                    torch.tensor(0, device=distance.device,
                                 dtype=torch.float)).pow(2)

                # 將hinge的值全部加1之後取log最後取平均值
                aggr = torch.log(hinge + 1).mean()
                # 記錄到loss_aggr_img當中，這裡的aggr就會是一張圖像當中的一個文字團的Lagg
                loss_aggr_img.append(aggr)

            # 獲取總共有記錄下多少個文字團的損失
            num_inst = len(loss_aggr_img)
            if num_inst > 0:
                # 將每個文字團的Lagg進行求和平均
                loss_aggr_img = torch.stack(loss_aggr_img).mean()
            else:
                loss_aggr_img = torch.tensor(
                    0, device=gt_texts.device, dtype=torch.float)
            # 進行保存，保存一張圖像的Lagg
            loss_aggrs.append(loss_aggr_img)

            loss_discr_img = 0
            # 透過標準庫當中的itertools的combinations可以獲取傳入的list的排列數，後面的2表示長度為2的所有排列方式
            # 這裡是用kernel_avgs進行排列，所以可以獲取任意兩倆之間的差值，avg_i與avg_j的shape [4]，計算Ldis
            for avg_i, avg_j in itertools.combinations(kernel_avgs, 2):
                # delta_discrimination - ||G(K_i) - G(K_j)||，獲取avg_i與avg_j的距離
                distance_ij = self.delta_discrimination - (avg_i -
                                                           avg_j).norm(2)
                # D(K_i,K_j)，如果距離小於0就會是0，否則就會是距離值的平方
                D_ij = torch.max(
                    distance_ij,
                    torch.tensor(
                        0, device=distance_ij.device,
                        dtype=torch.float)).pow(2)
                # 將計算出的D_ij加上一後取log之後累加到loss_discr_img當中
                loss_discr_img += torch.log(D_ij + 1)

            if num_inst > 1:
                # 如果有超過一個標註匡就會到這裡，這裡是公式的計算
                loss_discr_img /= (num_inst * (num_inst - 1))
            else:
                loss_discr_img = torch.tensor(
                    0, device=gt_texts.device, dtype=torch.float)
            if num_inst == 0:
                # 如果過濾後沒有半個標註匡就會跳出警告
                warnings.warn('num of instance is 0')
            # 記錄下logg_discr_img
            loss_discrs.append(loss_discr_img)
        # 最後將batch進行stack
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
