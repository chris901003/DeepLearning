# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info)
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import build_dataloader, build_dataset

from mmocr import digit_version
from mmocr.apis.utils import (disable_text_recog_aug_test,
                              replace_image_to_tensor)
from mmocr.utils import get_root_logger


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    """ 已看過，訓練文字偵測模型
    Args:
        model: 模型本身
        dataset: 訓練集的Dataset
        cfg: config文件
        distributed: 是否啟用分布式訓練
        validate: 是否需要進行驗證
        timestamp: 時間戳
        meta: 保存訓練過程資料的
    """

    # logger = 製作log的東西
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    # 如果傳入的dataset不是list型態就會轉成list
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # step 1: give default values and override (if exist) from cfg.data
    # 構建DataLoader的config資料
    default_loader_cfg = {
        **dict(
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.get('seed'),
            drop_last=False,
            persistent_workers=False),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
    }
    # update overall dataloader(for train, val and test) setting
    # 將一些其他資料更新到config當中
    default_loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })

    # step 2: cfg.data.train_dataloader has highest priority
    # cfg當中的train_dataloader有最高優先權，如果有相同的key會以cfg當中的為主
    train_loader_cfg = dict(default_loader_cfg,
                            **cfg.data.get('train_dataloader', {}))

    # 根據傳入的config以及dataset進行構建DataLoader實例化對象
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        # 如果使用分布式學習會到這裡
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        if not torch.cuda.is_available():
            # 如果使用cpu進行訓練需要將MMCV版本升級到1.4.4以上
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        # 將模型放到MMDataParallel
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    # build runner
    # 構建優化器部分
    optimizer = build_optimizer(model, cfg.optimizer)

    # 構建runner需要使用到的config資料
    if 'runner' not in cfg:
        cfg.runner = {
            # 以epoch為基底的計算方式
            'type': 'EpochBasedRunner',
            # 設定總epoch數量
            'max_epochs': cfg.total_epochs
        }
        # 會跳出警告，這裡希望在config文件當中已經配置好runner資料
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            # 檢查config設定的total_epochs與max_epochs是否相同
            assert cfg.total_epochs == cfg.runner.max_epochs

    # 構建runner實例對象，之後訓練模型以及驗證模型都是使用裏面的api進行
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    # 獲取當前時間戳
    runner.timestamp = timestamp

    # fp16 setting，gpu相關設定
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        # 多gpu相關設定
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks，添加鉤子函數
    runner.register_training_hooks(
        # 學習率
        cfg.lr_config,
        # 優化器
        optimizer_config,
        # checkpoint
        cfg.checkpoint_config,
        # log相關
        cfg.log_config,
        cfg.get('momentum_config', None),
        # 其他自定義鉤子
        custom_hooks_config=cfg.get('custom_hooks', None))
    if distributed:
        # 如果啟用分布式訓練就會多一個鉤子函數
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # 如果有需要驗證就會構建與驗證相關的東西
        # 獲取一個gpu一次會驗證多少張圖像，如果沒有特別指定就會是1
        val_samples_per_gpu = (cfg.data.get('val_dataloader', {})).get(
            'samples_per_gpu', cfg.data.get('samples_per_gpu', 1))
        if val_samples_per_gpu > 1:
            # Support batch_size > 1 in test for text recognition
            # by disable MultiRotateAugOCR since it is useless for most case
            # 如果驗證時一個batch輸入多張圖像就會將MultiRotateAugOCR關閉，不過該功能對多數case不起作用所以沒有關係
            cfg = disable_text_recog_aug_test(cfg)
            cfg = replace_image_to_tensor(cfg)

        # 構建驗證集的dataset
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        # 構建驗證集DataLoader的相關配置
        val_loader_cfg = {
            **default_loader_cfg,
            **dict(shuffle=False, drop_last=False),
            **cfg.data.get('val_dataloader', {}),
            **dict(samples_per_gpu=val_samples_per_gpu)
        }

        # 構建驗證集的DataLoader實例對象
        val_dataloader = build_dataloader(val_dataset, **val_loader_cfg)

        # 獲取evaluation相關資料
        eval_cfg = cfg.get('evaluation', {})
        # 如果是以epoch為主by_epoch就會是True
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        # 根據是否有使用分布式訓練挑選鉤子函數
        eval_hook = DistEvalHook if distributed else EvalHook
        # 添加驗證的鉤子函數
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        # 如果需要載入整個模型的預訓練權重就會在這裡載入
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        # 如果需要從上次訓練到的地方繼續訓練就從這裡載入
        runner.load_checkpoint(cfg.load_from)
    # 開始進行模型訓練
    runner.run(data_loaders, cfg.workflow)


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed. If the seed is None, it will be replaced by a
    random number, and then broadcasted to all processes.

    Args:
        seed (int, Optional): The seed.
        device (str): The device where the seed will be put on.

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
