# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.utils import util_mixins


class SamplingResult(util_mixins.NiceRepr):
    """Bbox sampling result.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_bboxes': torch.Size([12, 4]),
            'neg_inds': tensor([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
            'num_gts': 4,
            'pos_assigned_gt_inds': tensor([], dtype=torch.int64),
            'pos_bboxes': torch.Size([0, 4]),
            'pos_inds': tensor([], dtype=torch.int64),
            'pos_is_gt': tensor([], dtype=torch.uint8)
        })>
    """

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        """ 已看過
        Args:
            pos_inds: 預測匡中正樣本的index
            neg_inds: 預設匡中負樣本的index
            bboxes: 預測匡資料shape [num_pred, 4]
            gt_bboxes: 標註匡資料shape [num_gt, 4]
            assign_result: AssignResult的實例化對象
        """

        # 保存輸入資料
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        # 獲取正樣本的預測匡資訊shape [num_gt, 4]
        self.pos_bboxes = bboxes[pos_inds]
        # 獲取副樣本的預測匡資訊shape [num_pred - num_gt, 4]
        self.neg_bboxes = bboxes[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        # 有多少個標註匡
        self.num_gts = gt_bboxes.shape[0]
        # 獲取預測匡對應上的gt匡的index，這裡又將1扣回來
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # 如果沒有gt匡就會在這裡
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            # 記錄下預測匡對應上的真實匡資訊
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long(), :]

        if assign_result.labels is not None:
            # 記錄下預測匡對應上的類別資訊
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        return torch.cat([self.pos_bboxes, self.neg_bboxes])

    def to(self, device):
        """Change the device of the data inplace.

        Example:
            >>> self = SamplingResult.random()
            >>> print(f'self = {self.to(None)}')
            >>> # xdoctest: +REQUIRES(--gpu)
            >>> print(f'self = {self.to(0)}')
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, torch.Tensor):
                _dict[key] = value.to(device)
        return self

    def __nice__(self):
        data = self.info.copy()
        data['pos_bboxes'] = data.pop('pos_bboxes').shape
        data['neg_bboxes'] = data.pop('neg_bboxes').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_bboxes': self.pos_bboxes,
            'neg_bboxes': self.neg_bboxes,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }

    @classmethod
    def random(cls, rng=None, **kwargs):
        """
        Args:
            rng (None | int | numpy.random.RandomState): seed or state.
            kwargs (keyword arguments):
                - num_preds: number of predicted boxes
                - num_gts: number of true boxes
                - p_ignore (float): probability of a predicted box assigned to \
                    an ignored truth.
                - p_assigned (float): probability of a predicted box not being \
                    assigned.
                - p_use_label (float | bool): with labels or not.

        Returns:
            :obj:`SamplingResult`: Randomly generated sampling result.

        Example:
            >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
            >>> self = SamplingResult.random()
            >>> print(self.__dict__)
        """
        from mmdet.core.bbox import demodata
        from mmdet.core.bbox.assigners.assign_result import AssignResult
        from mmdet.core.bbox.samplers.random_sampler import RandomSampler
        rng = demodata.ensure_rng(rng)

        # make probabalistic?
        num = 32
        pos_fraction = 0.5
        neg_pos_ub = -1

        assign_result = AssignResult.random(rng=rng, **kwargs)

        # Note we could just compute an assignment
        bboxes = demodata.random_boxes(assign_result.num_preds, rng=rng)
        gt_bboxes = demodata.random_boxes(assign_result.num_gts, rng=rng)

        if rng.rand() > 0.2:
            # sometimes algorithms squeeze their data, be robust to that
            gt_bboxes = gt_bboxes.squeeze()
            bboxes = bboxes.squeeze()

        if assign_result.labels is None:
            gt_labels = None
        else:
            gt_labels = None  # todo

        if gt_labels is None:
            add_gt_as_proposals = False
        else:
            add_gt_as_proposals = True  # make probabalistic?

        sampler = RandomSampler(
            num,
            pos_fraction,
            neg_pos_ub=neg_pos_ub,
            add_gt_as_proposals=add_gt_as_proposals,
            rng=rng)
        self = sampler.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        return self
