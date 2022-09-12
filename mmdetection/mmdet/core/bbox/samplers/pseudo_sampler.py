# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


@BBOX_SAMPLERS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        # 已看過，目前不確定作用
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, *args, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        # 已看過
        # assign_result = 配對過後將資料放入到一個class構建的實例對象
        # bboxes = 預測匡資料，shape [num_pred, 4]
        # gt_bboxes = 真實匡資料，shape [num_gt, 4]

        # 透過nonzero可以獲取哪些部分不是0的index
        # pos_inds正樣本的index，表示出哪些預測匡有匡到物體
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        # neg_inds負樣本的index，表示哪些匡是匡到背景
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        # 構建一個shape [num_pred]且全為0的tensor
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        # 將結果用SamplingResults包裝
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        # 回傳包裝後的實例化對象
        return sampling_result
