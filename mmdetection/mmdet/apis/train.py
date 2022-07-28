# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_runner,
                         get_dist_info)

from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint, get_root_logger)


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    # Get flag from config
    if ('auto_scale_lr' not in cfg) or \
            (not cfg.auto_scale_lr.get('enable', False)):
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get base batch size from config
    base_batch_size = cfg.auto_scale_lr.get('base_batch_size', None)
    if base_batch_size is None:
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = len(range(world_size))
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    samples_per_gpu = cfg.data.train_dataloader.samples_per_gpu
    batch_size = num_gpus * samples_per_gpu
    logger.info(f'Training with {num_gpus} GPU(s) with {samples_per_gpu} '
                f'samples per GPU. The total batch size is {batch_size}.')

    if batch_size != base_batch_size:
        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / base_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')
        cfg.optimizer.lr = scaled_lr
    else:
        logger.info('The batch size match the '
                    f'base batch size: {base_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    """ 已看過，準備進行模型訓練，會先構建一些鉤子函數
    Args:
        model: 模型本身
        dataset: 資料集
        cfg: config文件內容
        distributed: 是否有啟用分布式訓練
        validate: 是否需要進行驗證
        timestamp: 時間戳
        meta: 保存訓練過程資訊的
    """

    # 調整config文件的兼容性，主要是對於分布式訓練以及多gpu的會需要經過調整
    cfg = compat_cfg(cfg)
    # logger = 紀錄使用的
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders，如果dataset不是list型態就會在外面包上list
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # 獲取以什麼方式計算一個iter，在object detection都會以epoch作為底
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    # 構建DataLoader的基本參數
    train_dataloader_default_args = dict(
        # samples_per_gpu = 一個gpu一次會讀入幾張圖像，就是batch_size的大小
        samples_per_gpu=2,
        # 會用多少個進程去讀取圖像，越大的話讀取速度會越快，上限會是batch_size大小
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        # 總共有多少個gpu同時訓練
        num_gpus=len(cfg.gpu_ids),
        # 是否使用分布式訓練
        dist=distributed,
        # 隨機種子碼
        seed=cfg.seed,
        # iter的基礎
        runner_type=runner_type,
        # 是否讓workers佔住cpu資源
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    # 構建DataLoader
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    # 構建學習率調整
    auto_scale_lr(cfg, distributed, logger)
    # 構建優化器
    optimizer = build_optimizer(model, cfg.optimizer)

    # 構建runner，裡面會有4個有用的api
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    # 獲取時間戳，用來檔案命名的
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        # 獲取優化器的config
        optimizer_config = cfg.optimizer_config

    # register hooks，將鉤子函數添加上去
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        # 如果是分布式訓練就會需要多一個鉤子函數
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # 如果有驗證模式，這裡會建立驗證資料的DataLoader
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # 當一個gpu的batch_size大於1就會進來
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        # 構建驗證集的dataset
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        # 構建驗證集的DataLoader
        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        # 添加鉤子函數
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        # 是否需要接續上次的訓練繼續訓練
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        # 是否需要接續上次的訓練繼續訓練
        cfg.resume_from = resume_from

    if cfg.resume_from:
        # 接續上次的訓練，會載入優化器以及學習率
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        # 載入預訓練權重，不會載入優化器以及學習率
        runner.load_checkpoint(cfg.load_from)
    # 開始進行訓練
    runner.run(data_loaders, cfg.workflow)
