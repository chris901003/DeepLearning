# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer, get_dist_info)
from mmcv.runner.hooks import Fp16OptimizerHook

from ..core import (DistEvalHook, EvalHook, OmniSourceDistSamplerSeedHook,
                    OmniSourceRunner)
from ..datasets import build_dataloader, build_dataset
from ..utils import (PreciseBNHook, build_ddp, build_dp, default_device,
                     get_root_logger)
from .test import multi_gpu_test


def init_random_seed(seed=None, device=default_device, distributed=True):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
        distributed (bool): Whether to use distributed training.
            Default: True.
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

    if distributed:
        dist.broadcast(random_num, src=0)
    return random_num.item()


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                test=dict(test_best=False, test_last=False),
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (:obj:`Dataset`): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        test (dict): The testing option, with two keys: test_last & test_best.
            The value is True or False, indicating whether to test the
            corresponding checkpoint.
            Default: dict(test_best=False, test_last=False).
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    # 已看過，準備進行訓練，這裡會將模型做最後處理並且添加上一些鉤子函數，在訓練時可以透過鉤子函數在對應時間呼叫
    # model = 模型本身
    # dataset = 訓練的dataset，這裡還需要將其放入到Dataloader當中
    # cfg = config資料
    # distributed = 是否有使用分布式訓練
    # validate = 是否需要進行驗證模式
    # test = 在哪些時候需要test的設定
    # timestamp = 啟動程序時的時間戳
    # meta = 保存訓練過程中的參數

    # 構建logger用來打印訓練時的資訊
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    # 如果傳入的dataset不是list或是tuple格式，就會將其轉成list格式
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    # 構建Dataloader時需要的設定
    dataloader_setting = dict(
        # 設定每個gpu的batch_size，如果config當中沒有設定就會是1
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        # 設定每個gpu讀取資料時可以使用多少進程，如果config當中沒有設定就會是1
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        # 是否要將讀取資料的進程一直保留
        persistent_workers=cfg.data.get('persistent_workers', False),
        # 使用的gpu數量
        num_gpus=len(cfg.gpu_ids),
        # 是否有啟用分布式訓練
        dist=distributed,
        # 種子碼
        seed=cfg.seed)
    # 將config當中設定的其他關於dataloader的config資料放入
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    if cfg.omnisource:
        # 如果有使用omnisource就會到這裡
        # The option can override videos_per_gpu
        train_ratio = cfg.data.get('train_ratio', [1] * len(dataset))
        omni_videos_per_gpu = cfg.data.get('omni_videos_per_gpu', None)
        if omni_videos_per_gpu is None:
            dataloader_settings = [dataloader_setting] * len(dataset)
        else:
            dataloader_settings = []
            for videos_per_gpu in omni_videos_per_gpu:
                this_setting = cp.deepcopy(dataloader_setting)
                this_setting['videos_per_gpu'] = videos_per_gpu
                dataloader_settings.append(this_setting)
        data_loaders = [
            build_dataloader(ds, **setting)
            for ds, setting in zip(dataset, dataloader_settings)
        ]

    else:
        # 沒有使用omnisource就會到這裡，直接構建Dataloader實例對象
        data_loaders = [
            build_dataloader(ds, **dataloader_setting) for ds in dataset
        ]

    # put model on gpus
    if distributed:
        # 如果是分布式訓練就會到這裡
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        # 構建DDP模型
        model = build_ddp(
            model,
            default_device,
            default_args=dict(
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters))
    else:
        # 其他就會到這裡，構建DP模型
        model = build_dp(
            model, default_device, default_args=dict(device_ids=cfg.gpu_ids))

    # build runner
    # 構建優化器實例對象
    optimizer = build_optimizer(model, cfg.optimizer)

    # 選擇使用的Runner類
    Runner = OmniSourceRunner if cfg.omnisource else EpochBasedRunner
    # 構建Runner實例化對象
    runner = Runner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    # 將時間戳給到runner當中
    runner.timestamp = timestamp

    # fp16 setting
    # 獲取fp16的設定
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        # 如果有設定fp16_cfg就會到這裡
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        # 如果是分布式訓練且optimizer_config當中沒有type就會到這裡
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        # 其他
        optimizer_config = cfg.optimizer_config

    # register hooks
    # 將鉤子函數添加到runner當中，這樣就可以在對應的時間呼叫要使用的鉤子函數
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    # multigrid setting
    # 獲取多重網格相關設定
    multigrid_cfg = cfg.get('multigrid', None)
    if multigrid_cfg is not None:
        # 如果有設定multigrid_cfg就會到這裡
        from mmaction.utils.multigrid import LongShortCycleHook
        multigrid_scheduler = LongShortCycleHook(cfg)
        runner.register_hook(multigrid_scheduler)
        logger.info('Finish register multigrid hook')

        # subbn3d aggregation is HIGH, as it should be done before
        # saving and evaluation
        from mmaction.utils.multigrid import SubBatchNorm3dAggregationHook
        subbn3d_aggre_hook = SubBatchNorm3dAggregationHook()
        runner.register_hook(subbn3d_aggre_hook, priority='VERY_HIGH')
        logger.info('Finish register subbn3daggre hook')

    # precise bn setting
    if cfg.get('precise_bn', False):
        # 如果有設定precise_bn就會到這裡
        precise_bn_dataset = build_dataset(cfg.data.train)
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=1,  # save memory and time
            persistent_workers=cfg.data.get('persistent_workers', False),
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed)
        data_loader_precise_bn = build_dataloader(precise_bn_dataset,
                                                  **dataloader_setting)
        precise_bn_hook = PreciseBNHook(data_loader_precise_bn,
                                        **cfg.get('precise_bn'))
        runner.register_hook(precise_bn_hook, priority='HIGHEST')
        logger.info('Finish register precisebn hook')

    if distributed:
        # 如果是分布式訓練就會到這裡
        if cfg.omnisource:
            runner.register_hook(OmniSourceDistSamplerSeedHook())
        else:
            runner.register_hook(DistSamplerSeedHook())

    if validate:
        # 如果有設定驗證就會到這裡，主要是構建驗證的資料集以及添加上鉤子函數
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook(val_dataloader, **eval_cfg) if distributed \
            else EvalHook(val_dataloader, **eval_cfg)
        runner.register_hook(eval_hook, priority='LOW')

    if cfg.resume_from:
        # 如果有需要從上次訓練到的地方繼續訓練就會到這裡
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        # 如果有先訓練好的權重可以到這裡載入
        runner.load_checkpoint(cfg.load_from)
    runner_kwargs = dict()
    if cfg.omnisource:
        # 如果有啟用omnisource就會到這裡
        runner_kwargs = dict(train_ratio=train_ratio)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, **runner_kwargs)

    if distributed:
        dist.barrier()
    time.sleep(5)

    if test['test_last'] or test['test_best']:
        best_ckpt_path = None
        if test['test_best']:
            ckpt_paths = [x for x in os.listdir(cfg.work_dir) if 'best' in x]
            ckpt_paths = [x for x in ckpt_paths if x.endswith('.pth')]
            if len(ckpt_paths) == 0:
                runner.logger.info('Warning: test_best set, but no ckpt found')
                test['test_best'] = False
                if not test['test_last']:
                    return
            elif len(ckpt_paths) > 1:
                epoch_ids = [
                    int(x.split('epoch_')[-1][:-4]) for x in ckpt_paths
                ]
                best_ckpt_path = ckpt_paths[np.argmax(epoch_ids)]
            else:
                best_ckpt_path = ckpt_paths[0]
            if best_ckpt_path:
                best_ckpt_path = osp.join(cfg.work_dir, best_ckpt_path)

        test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        gpu_collect = cfg.get('evaluation', {}).get('gpu_collect', False)
        tmpdir = cfg.get('evaluation', {}).get('tmpdir',
                                               osp.join(cfg.work_dir, 'tmp'))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            persistent_workers=cfg.data.get('persistent_workers', False),
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('test_dataloader', {}))

        test_dataloader = build_dataloader(test_dataset, **dataloader_setting)

        names, ckpts = [], []

        if test['test_last']:
            names.append('last')
            ckpts.append(None)
        if test['test_best'] and best_ckpt_path is not None:
            names.append('best')
            ckpts.append(best_ckpt_path)

        for name, ckpt in zip(names, ckpts):
            if ckpt is not None:
                runner.load_checkpoint(ckpt)

            outputs = multi_gpu_test(runner.model, test_dataloader, tmpdir,
                                     gpu_collect)
            rank, _ = get_dist_info()
            if rank == 0:
                out = osp.join(cfg.work_dir, f'{name}_pred.pkl')
                test_dataset.dump_results(outputs, out)

                eval_cfg = cfg.get('evaluation', {})
                for key in [
                        'interval', 'tmpdir', 'start', 'gpu_collect',
                        'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers'
                ]:
                    eval_cfg.pop(key, None)

                eval_res = test_dataset.evaluate(outputs, **eval_cfg)
                runner.logger.info(f'Testing results of the {name} checkpoint')
                for metric_name, val in eval_res.items():
                    runner.logger.info(f'{metric_name}: {val:.04f}')
