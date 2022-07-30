# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES

from . import BaseTextDetTargets


@PIPELINES.register_module()
class PANetTargets(BaseTextDetTargets):
    """Generate the ground truths for PANet: Efficient and Accurate Arbitrary-
    Shaped Text Detection with Pixel Aggregation Network.

    [https://arxiv.org/abs/1908.05900]. This code is partially adapted from
    https://github.com/WenmuZhou/PAN.pytorch.

    Args:
        shrink_ratio (tuple[float]): The ratios for shrinking text instances.
        max_shrink (int): The maximum shrink distance.
    """

    def __init__(self, shrink_ratio=(1.0, 0.5), max_shrink=20):
        """ 已看過，對原始標註圖像進行縮放，為了可以算出不同層輸出的損失值
        Args:
            shrink_ratio: 縮放比例
            max_shrink: 最大縮放距離
        """
        # 將傳入的值進行保存
        self.shrink_ratio = shrink_ratio
        self.max_shrink = max_shrink

    def generate_targets(self, results):
        """Generate the gt targets for PANet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """
        # 已看過，產生PANet所需要的標註圖像資料，透過將最終標註圖像進行範圍縮放獲取

        # 檢查resutls必須是dict格式
        assert isinstance(results, dict)

        # 獲取masks標註訊息
        polygon_masks = results['gt_masks'].masks
        # 獲取不合法的masks標註訊息
        polygon_masks_ignore = results['gt_masks_ignore'].masks

        # 獲取圖像高寬
        h, w, _ = results['img_shape']
        # 保存不同縮減比例的標註圖像
        gt_kernels = []
        # 遍歷縮減比例
        for ratio in self.shrink_ratio:
            # 透過generate_kernels獲取經過縮減後的標註圖像
            # mask = ndarray shape [height, width]，假設該圖像有5個標註匡就會在對應的位置上標註成遍歷到的index+1，其他都會是0
            mask, _ = self.generate_kernels((h, w),
                                            polygon_masks,
                                            ratio,
                                            max_shrink=self.max_shrink,
                                            ignore_tags=None)
            # 將結果保存下來
            gt_kernels.append(mask)
        # 透過generate_effective_mask處理ignore的mask，gt_mask會在ignore標註位置為0其餘為1，shape [height, width]
        gt_mask = self.generate_effective_mask((h, w), polygon_masks_ignore)

        # 將mask_fields資訊清空，原先裡面裝的是標註哪些key與mask有關係
        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        if 'bbox_fields' in results:
            # 清空bbox_fields資訊
            results['bbox_fields'].clear()
        # 清空一系列results當中的資訊
        results.pop('gt_labels', None)
        results.pop('gt_masks', None)
        results.pop('gt_bboxes', None)
        results.pop('gt_bboxes_ignore', None)

        # 構建gt_kernels與gt_mask的字典
        mapping = {'gt_kernels': gt_kernels, 'gt_mask': gt_mask}
        # 遍歷mapping
        for key, value in mapping.items():
            # 如果value不是list格式就變成list格式
            value = value if isinstance(value, list) else [value]
            # 更新對應的key且value會是BitmapMasks實例對象
            results[key] = BitmapMasks(value, h, w)
            # 將該key保存到與mask相關的mask_fields當中
            results['mask_fields'].append(key)

        # 回傳更新好的results
        return results
