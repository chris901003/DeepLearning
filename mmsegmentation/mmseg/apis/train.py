# Copyright (c) OpenMMLab. All rights reserved.
import os
import random
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         build_runner, get_dist_info)
from mmcv.utils import build_from_cfg

from mmseg import digit_version
from mmseg.core import DistEvalHook, EvalHook, build_optimizer
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import (build_ddp, build_dp, find_latest_checkpoint,
                         get_root_logger)


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
    # 已看過
    # 設定本次訓練的亂數種子

    if seed is not None:
        # 如果在訓練時有給seed就會直接回傳
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    # 在多gpu下要確保每個gpu的亂數種子要相同，不然有機率發生不可預測的錯誤
    rank, world_size = get_dist_info()
    # seed = 隨機種子碼
    seed = np.random.randint(2**31)
    if world_size == 1:
        # 如果是單gpu就直接回傳種子碼
        return seed

    # 多gpu就會需要同步種子碼
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
    # 已看過
    # 設定所有相關的隨機種子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Launch segmentor training.
# 訓練開始的地方，從這裡進入訓練流程
def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """
    :param model: 整個網路模型
    :param dataset: 資料集的讀入以及處理
    :param cfg: 配置文檔
    :param distributed: 是否啟用分布式訓練
    :param validate: 是否啟用驗證
    :param timestamp: 時間戳，用來記錄用的
    :param meta: 紀錄一些重要的資訊，例如系統環境或是隨機種子碼，總之就是跟log有關的內容，這裡對我們來說並不是很重要
    :return:
    """
    # 已看過

    # 保存log檔用的東西
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    # 如果dataset最外層不是list型態就轉成list型態，通常會已經是list型態了
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # The default loader config
    # loader的配置
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True)
    # The overall dataloader settings
    # 整體數據加載設置
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })

    # The specific dataloader settings
    # 載入一些特定的加載方式
    train_loader_cfg = {**loader_cfg, **cfg.data.get('train_dataloader', {})}
    # 原先dataset當中為mmseg.datasets型態，通過build_dataloader會變成torch.utils.dataloader.Dataloader型態
    # 這裡會將上面的train_loader_cfg傳入
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on devices
    if distributed:
        # 使用分布式訓練會到這裡來
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # DDP wrapper
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        # 使用單機單gpu或是多gpu或是cpu都會到這裡
        if not torch.cuda.is_available():
            # 如果要使用cpu進行訓練要將MMCV版本升級到1.4.4以上
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        # 將model放到設備上，通過後model外層會多MMDataParallel，原先的model會放在裡面
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build runner
    # 構建優化器，將模型以及優化器配置傳入
    # optimizer = 優化器實例對象，如果是pytorch官方實現的就會直接是torch.optim的實例對象
    optimizer = build_optimizer(model, cfg.optimizer)

    if cfg.get('runner') is None:
        # 檢查cfg當中有沒有runner，如果沒有這裡會添加上去
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        # 順便提出一個警告，在新版本當中需要runner這個參數
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    # 構建runner，回傳的是一個runner實例對象，裏面實現了四個api
    # run(), train(), val(), save_checkpoint()
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            batch_processor=None,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # 在訓練以及驗證的過程當中，整體流程大多都是用鉤子函數進行控制的
    # register hooks
    # 這裡開始在runner上面加上一些常用的鉤子函數，可以在訓練的時候使用
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        # 如果有進行分布式訓練才會進來
        # when distributed training by epoch, using`DistSamplerSeedHook` to set
        # the different seed to distributed sampler for each epoch, it will
        # shuffle dataset at each epoch and avoid overfitting.
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # an ugly walkaround to make the .log and .log.json filenames the same
    # 將時間戳放入到runner當中
    runner.timestamp = timestamp

    # register eval hooks，創建驗證模式會需要用到的鉤子函數
    if validate:
        # val_dataset = 驗證集的dataset，這裡與創建train_dataset相同
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # The specific dataloader settings
        # val_loader_cfg = 一些設定dataloader的相關參數
        val_loader_cfg = {
            **loader_cfg,
            'samples_per_gpu': 1,
            'shuffle': False,  # Not shuffle by default
            **cfg.data.get('val_dataloader', {}),
        }
        # val_dataloader = 官方實現的Dataloader，透過上面的dataset以及設定檔形成的
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)
        # 獲取evaluation內容
        eval_cfg = cfg.get('evaluation', {})
        # 如果不是IterBasedRunner表示是以一個epoch當作一次迭代
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        # 依據是否使用分布式學習決定eval_hook
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        # 如果有自定義的鉤子函數會進去，大多數時候不會有自定義的鉤子
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    # 一下就是載入預訓練權重或是接續上次訓練到一半的參數(這個部分晚點再來研究如何使用)
    if cfg.resume_from is None and cfg.get('auto_resume'):
        # 透過auto_resume將resume資訊放到resume_from當中
        resume_from = find_latest_checkpoint(cfg.work_dir)
        if resume_from is not None:
            cfg.resume_from = resume_from
    if cfg.resume_from:
        # 如果有指定resume_from就會從這裡加載
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        # 這裡也是加載權重
        runner.load_checkpoint(cfg.load_from)
    # 透過runner的run這個api正式開始模型訓練，同時如果有開啟驗證也會進行驗證
    # 也就是完整的一整套訓練流程
    runner.run(data_loaders, cfg.workflow)
