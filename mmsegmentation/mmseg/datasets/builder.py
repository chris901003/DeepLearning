# Copyright (c) OpenMMLab. All rights reserved.
import copy
import platform
import random
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg, digit_version
from torch.utils.data import DataLoader

from .samplers import DistributedSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def _concat_dataset(cfg, default_args=None):
    """Build :obj:`ConcatDataset by."""
    from .dataset_wrappers import ConcatDataset
    img_dir = cfg['img_dir']
    ann_dir = cfg.get('ann_dir', None)
    split = cfg.get('split', None)
    # pop 'separate_eval' since it is not a valid key for common datasets.
    separate_eval = cfg.pop('separate_eval', True)
    num_img_dir = len(img_dir) if isinstance(img_dir, (list, tuple)) else 1
    if ann_dir is not None:
        num_ann_dir = len(ann_dir) if isinstance(ann_dir, (list, tuple)) else 1
    else:
        num_ann_dir = 0
    if split is not None:
        num_split = len(split) if isinstance(split, (list, tuple)) else 1
    else:
        num_split = 0
    if num_img_dir > 1:
        assert num_img_dir == num_ann_dir or num_ann_dir == 0
        assert num_img_dir == num_split or num_split == 0
    else:
        assert num_split == num_ann_dir or num_ann_dir <= 1
    num_dset = max(num_split, num_img_dir)

    datasets = []
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        if isinstance(img_dir, (list, tuple)):
            data_cfg['img_dir'] = img_dir[i]
        if isinstance(ann_dir, (list, tuple)):
            data_cfg['ann_dir'] = ann_dir[i]
        if isinstance(split, (list, tuple)):
            data_cfg['split'] = split[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets, separate_eval)


def build_dataset(cfg, default_args=None):
    """Build datasets."""
    # 已看過
    # 構建dataset
    # cfg = 構建dataset的詳細參數，每一個dict表示一種處理的方式，cfg本身也是dict格式
    from .dataset_wrappers import (ConcatDataset, MultiImageMixDataset,
                                   RepeatDataset)

    if isinstance(cfg, (list, tuple)):
        # 當cfg的型態是list或是tuple時表示，我們會用多個資料集，所以需要進行合併(目前先不去看細節)
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        # 當我們的type為RepeatDataset會往這裡，目前只有看到這個先從這裡開始
        # 將構建dataset詳細內容傳入以及default_args以及要重複多少次，由RepeatDataset構建
        # 在進入RepeatDataset前會先再次進入build_dataset當中一次
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'MultiImageMixDataset':
        cp_cfg = copy.deepcopy(cfg)
        cp_cfg['dataset'] = build_dataset(cp_cfg['dataset'])
        cp_cfg.pop('type')
        dataset = MultiImageMixDataset(**cp_cfg)
    elif isinstance(cfg.get('img_dir'), (list, tuple)) or isinstance(
            cfg.get('split', None), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        # 當cfg['type']都不符合上面的條件就是type是資料集名稱
        # 使用build_from_cfg構建資料集
        # 這裡的DATASETS支援20種資料集
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     persistent_workers=True,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        persistent_workers (bool): If True, the data loader will not shutdown
            the worker processes after a dataset has been consumed once.
            This allows to maintain the workers Dataset instances alive.
            The argument also has effect in PyTorch>=1.7.0.
            Default: True
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    # dataset = MMCV格式的資料集
    # samples_per_gpu = 這個跟batch_size是一樣的東西
    # worker_per_gpu = 會使用幾個cpu去加載數據集，這裡跟num_workers相同
    # num_gpus = 使用多少個gpu進行訓練
    # dist = 是否啟用分布式訓練
    # shuffle = 每個epoch是否進行重排數據集
    # seed = 亂數種子
    # drop_last = 當最後一組數據量不到一個batch_size時，是否拋棄該組數據
    # pin_memory = 開啟後可以提高速度
    # persistent_workers = 如果開啟後dataloader就不會被關閉
    # return = 這裡會回傳一個pytorch格式的dataloader

    # 已看過

    # 獲取分布式訓練的一些資訊
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle, seed=seed)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        # 沒有使用分布式訓練會從這裡
        sampler = None
        # 總共的batch_size = 使用到的gpu數量 * 每個gpu一次訓練的圖片數量
        batch_size = num_gpus * samples_per_gpu
        # 總共的num_workers = 使用的gpu數量 * 每個gpu需要用到的cpu數量
        num_workers = num_gpus * workers_per_gpu

    # 透過worker_init_fn設定種子碼
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    # 根據torch版本會有不同的方式創建dataloader
    if digit_version(torch.__version__) >= digit_version('1.8.0'):
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            **kwargs)
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=pin_memory,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            drop_last=drop_last,
            **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """
    # 已看過
    # 主要是用來設定種子碼的

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
