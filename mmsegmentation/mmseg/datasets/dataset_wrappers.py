# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import collections
import copy
from itertools import chain

import mmcv
import numpy as np
from mmcv.utils import build_from_cfg, print_log
from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .builder import DATASETS, PIPELINES
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    support evaluation and formatting results

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
        separate_eval (bool): Whether to evaluate the concatenated
            dataset results separately, Defaults to True.
    """

    def __init__(self, datasets, separate_eval=True):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE
        self.separate_eval = separate_eval
        assert separate_eval in [True, False], \
            f'separate_eval can only be True or False,' \
            f'but get {separate_eval}'
        if any([isinstance(ds, CityscapesDataset) for ds in datasets]):
            raise NotImplementedError(
                'Evaluating ConcatDataset containing CityscapesDataset'
                'is not supported!')

    def evaluate(self, results, logger=None, **kwargs):
        """Evaluate the results.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]]): per image
                pre_eval results or predict segmentation map for
                computing evaluation metric.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str: float]: evaluate results of the total dataset
                or each separate
            dataset if `self.separate_eval=True`.
        """
        assert len(results) == self.cumulative_sizes[-1], \
            ('Dataset and results have different sizes: '
             f'{self.cumulative_sizes[-1]} v.s. {len(results)}')

        # Check whether all the datasets support evaluation
        for dataset in self.datasets:
            assert hasattr(dataset, 'evaluate'), \
                f'{type(dataset)} does not implement evaluate function'

        if self.separate_eval:
            dataset_idx = -1
            total_eval_results = dict()
            for size, dataset in zip(self.cumulative_sizes, self.datasets):
                start_idx = 0 if dataset_idx == -1 else \
                    self.cumulative_sizes[dataset_idx]
                end_idx = self.cumulative_sizes[dataset_idx + 1]

                results_per_dataset = results[start_idx:end_idx]
                print_log(
                    f'\nEvaluateing {dataset.img_dir} with '
                    f'{len(results_per_dataset)} images now',
                    logger=logger)

                eval_results_per_dataset = dataset.evaluate(
                    results_per_dataset, logger=logger, **kwargs)
                dataset_idx += 1
                for k, v in eval_results_per_dataset.items():
                    total_eval_results.update({f'{dataset_idx}_{k}': v})

            return total_eval_results

        if len(set([type(ds) for ds in self.datasets])) != 1:
            raise NotImplementedError(
                'All the datasets should have same types when '
                'self.separate_eval=False')
        else:
            if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                    results, str):
                # merge the generators of gt_seg_maps
                gt_seg_maps = chain(
                    *[dataset.get_gt_seg_maps() for dataset in self.datasets])
            else:
                # if the results are `pre_eval` results,
                # we do not need gt_seg_maps to evaluate
                gt_seg_maps = None
            eval_results = self.datasets[0].evaluate(
                results, gt_seg_maps=gt_seg_maps, logger=logger, **kwargs)
            return eval_results

    def get_dataset_idx_and_sample_idx(self, indice):
        """Return dataset and sample index when given an indice of
        ConcatDataset.

        Args:
            indice (int): indice of sample in ConcatDataset

        Returns:
            int: the index of sub dataset the sample belong to
            int: the index of sample in its corresponding subset
        """
        if indice < 0:
            if -indice > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            indice = len(self) + indice
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, indice)
        if dataset_idx == 0:
            sample_idx = indice
        else:
            sample_idx = indice - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        """format result for every sample of ConcatDataset."""
        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        ret_res = []
        for i, indice in enumerate(indices):
            dataset_idx, sample_idx = self.get_dataset_idx_and_sample_idx(
                indice)
            res = self.datasets[dataset_idx].format_results(
                [results[i]],
                imgfile_prefix + f'/{dataset_idx}',
                indices=[sample_idx],
                **kwargs)
            ret_res.append(res)
        return sum(ret_res, [])

    def pre_eval(self, preds, indices):
        """do pre eval for every sample of ConcatDataset."""
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]
        ret_res = []
        for i, indice in enumerate(indices):
            dataset_idx, sample_idx = self.get_dataset_idx_and_sample_idx(
                indice)
            res = self.datasets[dataset_idx].pre_eval(preds[i], sample_idx)
            ret_res.append(res)
        return sum(ret_res, [])


@DATASETS.register_module()
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        """
        :param dataset: Dataset實例對象，這個是已經針對選定數據集實例化的Dataset
        :param times: 重複次數
        """
        # 已看過

        # 將一些資料保存下來
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        # 原始圖像總共有多少張
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item from original dataset."""
        # 已看過，數據讀入的入口，從這個__getitem__函數開始
        # pytorch的Dataloader會給出一個隨機的idx，我們使用這個index獲取對應的圖片以及標註圖像
        # 因為給定__len__的方式與以epoch為主的有所不同，所以這裡我們要用idx模訓練集的圖像張數才會是我們真正要的
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """The length is multiplied by ``times``"""
        # 已看過，這裡是以iter為主，所以我們是用times*ort_len作為__getitem__的idx會獲得的隨機數的範圍
        return self.times * self._ori_len


@DATASETS.register_module()
class MultiImageMixDataset:
    """A wrapper of multiple images mixed dataset.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup. For the augmentation pipeline of mixed image data,
    the `get_indexes` method needs to be provided to obtain the image
    indexes, and you can set `skip_flags` to change the pipeline running
    process.


    Args:
        dataset (:obj:`CustomDataset`): The dataset to be mixed.
        pipeline (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        skip_type_keys (list[str], optional): Sequence of type string to
            be skip pipeline. Default to None.
    """

    def __init__(self, dataset, pipeline, skip_type_keys=None):
        assert isinstance(pipeline, collections.abc.Sequence)
        if skip_type_keys is not None:
            assert all([
                isinstance(skip_type_key, str)
                for skip_type_key in skip_type_keys
            ])
        self._skip_type_keys = skip_type_keys

        self.pipeline = []
        self.pipeline_types = []
        for transform in pipeline:
            if isinstance(transform, dict):
                self.pipeline_types.append(transform['type'])
                transform = build_from_cfg(transform, PIPELINES)
                self.pipeline.append(transform)
            else:
                raise TypeError('pipeline must be a dict')

        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self.num_samples = len(dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        results = copy.deepcopy(self.dataset[idx])
        for (transform, transform_type) in zip(self.pipeline,
                                               self.pipeline_types):
            if self._skip_type_keys is not None and \
                    transform_type in self._skip_type_keys:
                continue

            if hasattr(transform, 'get_indexes'):
                indexes = transform.get_indexes(self.dataset)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                mix_results = [
                    copy.deepcopy(self.dataset[index]) for index in indexes
                ]
                results['mix_results'] = mix_results

            results = transform(results)

            if 'mix_results' in results:
                results.pop('mix_results')

        return results

    def update_skip_type_keys(self, skip_type_keys):
        """Update skip_type_keys.

        It is called by an external hook.

        Args:
            skip_type_keys (list[str], optional): Sequence of type
                string to be skip pipeline.
        """
        assert all([
            isinstance(skip_type_key, str) for skip_type_key in skip_type_keys
        ])
        self._skip_type_keys = skip_type_keys
