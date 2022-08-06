# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmdet.core import BitmapMasks
from torch import nn

from mmocr.models.builder import LOSSES
from mmocr.utils import check_argument


@LOSSES.register_module()
class TextSnakeLoss(nn.Module):
    """The class for implementing TextSnake loss. This is partially adapted
    from https://github.com/princewang1994/TextSnake.pytorch.

    TextSnake: `A Flexible Representation for Detecting Text of Arbitrary
    Shapes <https://arxiv.org/abs/1807.01544>`_.

    Args:
        ohem_ratio (float): The negative/positive ratio in ohem.
    """

    def __init__(self, ohem_ratio=3.0):
        """ 已看過，TextSnakeLoss計算損失值的初始化
        Args:
            ohem_ratio: 在ohem當中正負樣本的比例
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        # 保存傳入的參數
        self.ohem_ratio = ohem_ratio

    def balanced_bce_loss(self, pred, gt, mask):
        """ 已看過，計算預測該為至是否為文本損失值
        Args:
            pred: 預測圖，shape tensor [batch_size, height, width]
            gt: 標註圖像，shape tensor [batch_size, height, width]
            mask: 為1的地方表示需要計算損失值，0表示不用計算損失值，shape tensor [batch_size, height, width]
        """

        # 檢查傳入的資訊
        assert pred.shape == gt.shape == mask.shape
        # 獲取正樣本位置，正樣本部分會是1負樣本會是0
        positive = gt * mask
        # 獲取負樣本位置，負樣本會是1正樣本會是0
        negative = (1 - gt) * mask
        # 計算總共正樣本數量
        positive_count = int(positive.float().sum())
        # 將gt轉成float型態
        gt = gt.float()
        if positive_count > 0:
            # 如果有正樣本會到這裡
            # 透過torch官方的BCE計算損失值，reduction使用的是none回傳就會與傳入的shape相同
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            # 獲取正樣本的loss，將loss乘上正樣本的mask，會將負樣本位置loss變成0
            positive_loss = torch.sum(loss * positive.float())
            # 獲取負樣本的loss，將loss乘上負樣本的mask，會將正樣本位置loss變成0
            negative_loss = loss * negative.float()
            # 獲取負樣本數量，如果負樣本數量超過正樣本的ohem_ratio倍，就會限制
            negative_count = min(
                int(negative.float().sum()),
                int(positive_count * self.ohem_ratio))
        else:
            # 如果沒有正樣本會到這裡
            # 將正樣本的損失值設定成0
            positive_loss = torch.tensor(0.0, device=pred.device)
            # 計算負樣本損失
            loss = F.binary_cross_entropy(pred, gt, reduction='none')
            negative_loss = loss * negative.float()
            # 將負樣本數量採100
            negative_count = 100
        # 獲取前negative_count的作為負樣本損失，shape tensor [negative_count]
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        # 將正樣本與負樣本相加最後取平均
        balance_loss = (positive_loss + torch.sum(negative_loss)) / (
            float(positive_count + negative_count) + 1e-5)

        # 回傳損失值
        return balance_loss

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
        # 已看過，將BitmapMasks轉成tensor格式
        # bitmasks = list[BitmapMasks實例對象]
        # target_sz = 轉換後的圖像大小

        # 檢查傳入的資料是否符合
        assert check_argument.is_type_list(bitmasks, BitmapMasks)
        assert isinstance(target_sz, tuple)

        # 獲取batch_size
        batch_size = len(bitmasks)
        # 獲取總共有多少個mask
        num_masks = len(bitmasks[0])

        # 最終結果保存的空間
        results = []

        # 遍歷所有的mask
        for level_inx in range(num_masks):
            # kernel的保存空間
            kernel = []
            # 遍歷整個batch的資料
            for batch_inx in range(batch_size):
                # 將當中的mask資料轉成tensor格式
                mask = torch.from_numpy(bitmasks[batch_inx].masks[level_inx])
                # hxw
                mask_sz = mask.shape
                # left, right, top, bottom，構建padding的大小
                pad = [
                    0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]
                ]
                # 進行padding
                mask = F.pad(mask, pad, mode='constant', value=0)
                kernel.append(mask)
            # 將結果堆疊起來
            kernel = torch.stack(kernel)
            results.append(kernel)

        # 最後回傳results
        # 會將原始最外層的list表示一個batch當中不同圖像的資料，變成不同kernel的資料
        # 也就是第一層list原始長度表示batch_size變成kernel的數量
        return results

    def forward(self, pred_maps, downsample_ratio, gt_text_mask,
                gt_center_region_mask, gt_mask, gt_radius_map, gt_sin_map,
                gt_cos_map):
        """
        Args:
            pred_maps (Tensor): The prediction map of shape
                :math:`(N, 5, H, W)`, where each dimension is the map of
                "text_region", "center_region", "sin_map", "cos_map", and
                "radius_map" respectively.
            downsample_ratio (float): Downsample ratio.
            gt_text_mask (list[BitmapMasks]): Gold text masks.
            gt_center_region_mask (list[BitmapMasks]): Gold center region
                masks.
            gt_mask (list[BitmapMasks]): Gold general masks.
            gt_radius_map (list[BitmapMasks]): Gold radius maps.
            gt_sin_map (list[BitmapMasks]): Gold sin maps.
            gt_cos_map (list[BitmapMasks]): Gold cos maps.

        Returns:
            dict:  A loss dict with ``loss_text``, ``loss_center``,
            ``loss_radius``, ``loss_sin`` and ``loss_cos``.
        """
        # 已看過，計算TextSnake的損失函數
        # pred_maps = 預測的結果，tensor shape [batch_size, channel=5, height, width]
        # downsample_ratio = 需要將gt縮放多少比例，這裡會是1，預測出來的特徵圖的高寬與輸入時相同
        # gt_text_mask = 如果是文字的地方就會是1否則就會是0
        # gt_center_region_mask = 如果是在文字中心的地方會是1否則會是0，會是gt_text_mask經過縮放後的結果
        # gt_mask = 如果該座標有包括被ignore的文字團的話就會是0否則都會是1，會根據gt_mask決定該點是否計算損失
        # gt_radius_map = 期望預測出來在該座標的半徑值
        # gt_sin_map = 期望預測出來在該座標的sin值
        # gt_cos_map = 期望預測出來在該座標的cos值

        # 檢查傳入的參數是否符合指定型態
        assert isinstance(downsample_ratio, float)
        assert check_argument.is_type_list(gt_text_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_center_region_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_mask, BitmapMasks)
        assert check_argument.is_type_list(gt_radius_map, BitmapMasks)
        assert check_argument.is_type_list(gt_sin_map, BitmapMasks)
        assert check_argument.is_type_list(gt_cos_map, BitmapMasks)

        # 將預測出來的channel=0的地方作為判斷是否為文字的置信度分數
        pred_text_region = pred_maps[:, 0, :, :]
        # 將預測出來的channel=1的地方作為判斷是否為文字中心區的置信度分數
        pred_center_region = pred_maps[:, 1, :, :]
        # 將預測出來的channel=2的地方作為該座標的sin值
        pred_sin_map = pred_maps[:, 2, :, :]
        # 將預測出來的channel=3的地方作為該座標的cos值
        pred_cos_map = pred_maps[:, 3, :, :]
        # 將預測出來的channel=4的地方作為該座標的半徑值
        pred_radius_map = pred_maps[:, 4, :, :]
        # 獲取預測的shape (batch_size, channel=5, height, width)
        feature_sz = pred_maps.size()
        # 當前運行的設備
        device = pred_maps.device

        # bitmask 2 tensor
        # 構建一個對應關係表，方便進行遍歷
        mapping = {
            'gt_text_mask': gt_text_mask,
            'gt_center_region_mask': gt_center_region_mask,
            'gt_mask': gt_mask,
            'gt_radius_map': gt_radius_map,
            'gt_sin_map': gt_sin_map,
            'gt_cos_map': gt_cos_map
        }
        # 保存gt相關資料
        gt = {}
        # 遍歷mapping當中的資料
        for key, value in mapping.items():
            # 取出當中的資料
            gt[key] = value
            # 將BitmapMasks轉成tensor格式
            if abs(downsample_ratio - 1.0) < 1e-2:
                # 如果下採樣倍率小於1.02就會到這裡
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
            else:
                # 其他的會到這裡
                gt[key] = [item.rescale(downsample_ratio) for item in gt[key]]
                gt[key] = self.bitmasks2tensor(gt[key], feature_sz[2:])
                if key == 'gt_radius_map':
                    gt[key] = [item * downsample_ratio for item in gt[key]]
            # 將資料轉移到訓練設備上面
            gt[key] = [item.to(device) for item in gt[key]]

        # 透過預測的sin與cos獲取之後對sin與cos的縮放比例
        scale = torch.sqrt(1.0 / (pred_sin_map**2 + pred_cos_map**2 + 1e-8))
        # 將sin與cos乘上縮放比例
        pred_sin_map = pred_sin_map * scale
        pred_cos_map = pred_cos_map * scale

        # 計算預測是否為文本區域的損失，tensor shape [1]，平均損失值
        loss_text = self.balanced_bce_loss(
            torch.sigmoid(pred_text_region), gt['gt_text_mask'][0],
            gt['gt_mask'][0])

        # 獲取文本的mask，將有文字mask與不計算loss的mask取and，要計算損失的地方會是True否則就會是False
        text_mask = (gt['gt_text_mask'][0] * gt['gt_mask'][0]).float()
        # 獲取文本中心的損失，將預測圖與標註圖透過BCE計算損失圖，這裡reduction是none所以會是圖
        loss_center_map = F.binary_cross_entropy(
            torch.sigmoid(pred_center_region),
            gt['gt_center_region_mask'][0].float(),
            reduction='none')
        if int(text_mask.sum()) > 0:
            # 如果需要計算loss的部分大於0就會到這裡
            # 這裡是計算loss平均值
            loss_center = torch.sum(
                loss_center_map * text_mask) / torch.sum(text_mask)
        else:
            # 如果全部都不需要計算loss到這裡，直接將loss設定成0
            loss_center = torch.tensor(0.0, device=device)

        # 獲取文本中心的mask，透過center的圖乘上不需要計算損失的部分，需要計算損失的會是True不需要計算的會是False
        center_mask = (gt['gt_center_region_mask'][0] *
                       gt['gt_mask'][0]).float()
        if int(center_mask.sum()) > 0:
            # 如果需要計算loss的center格子數大於0就會到這裡
            # 獲取半徑的shape (batch_size, height, width)
            map_sz = pred_radius_map.size()
            # 構建一個全為1且shape與pred_radius_map相同的tensor
            ones = torch.ones(map_sz, dtype=torch.float, device=device)
            # 計算radius損失
            loss_radius = torch.sum(
                # 使用smooth_l1_loss計算損失，這裡將預測的除以標註的，當值越接近一時表示預測與標註越接近
                # 所以與ones進行損失計算，之後還會在乘上mask只獲取需要計算損失的loss，最後取平均值
                F.smooth_l1_loss(
                    pred_radius_map / (gt['gt_radius_map'][0] + 1e-2),
                    ones,
                    reduction='none') * center_mask) / torch.sum(center_mask)
            # 計算sin損失，這裡與radius相同
            loss_sin = torch.sum(
                F.smooth_l1_loss(
                    pred_sin_map, gt['gt_sin_map'][0], reduction='none') *
                center_mask) / torch.sum(center_mask)
            # 計算cos損失，這裡與radius相同
            loss_cos = torch.sum(
                F.smooth_l1_loss(
                    pred_cos_map, gt['gt_cos_map'][0], reduction='none') *
                center_mask) / torch.sum(center_mask)
        else:
            # 如果沒有中心線就不會有radius與sin與cos的損失需要計算，直接設定成0
            loss_radius = torch.tensor(0.0, device=device)
            loss_sin = torch.tensor(0.0, device=device)
            loss_cos = torch.tensor(0.0, device=device)

        # 將損失打包成dict
        results = dict(
            loss_text=loss_text,
            loss_center=loss_center,
            loss_radius=loss_radius,
            loss_sin=loss_sin,
            loss_cos=loss_cos)

        # 回傳results
        return results
