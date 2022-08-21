# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         get_dist_info)
from mmcv.utils import digit_version

from mmpose.core import DistEvalHook, EvalHook, build_optimizers
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.utils import get_root_logger

try:
    from mmcv.runner import Fp16OptimizerHook
except ImportError:
    warnings.warn(
        'Fp16OptimizerHook from mmpose will be deprecated from '
        'v0.15.0. Please install mmcv>=1.1.4', DeprecationWarning)
    from mmpose.core import Fp16OptimizerHook


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


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    # 準備進行訓練模式
    # model = 模型實例化對象
    # dataset = dataset實例化對象
    # cfg = 對於訓練的config資料
    # distributed = 是否為分布式訓練
    # validate = 是否進行驗證
    # timestamp = 當前時間戳
    # meta = 保存訓練過程參數

    # 創建logger
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    # 如果傳入的dataset不是list或是tuple就用list進行包裝
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    # step 1: give default values and override (if exist) from cfg.data
    # 構建DataLoader的實例化參數
    loader_cfg = {
        **dict(
            # 種子碼
            seed=cfg.get('seed'),
            # 是否需要將不滿足一個batch的資料丟棄
            drop_last=False,
            # 是否為分布式訓練
            dist=distributed,
            # 使用的gpu數量
            num_gpus=len(cfg.gpu_ids)),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'samples_per_gpu',
                   'workers_per_gpu',
                   'shuffle',
                   'seed',
                   'drop_last',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }

    # step 2: cfg.data.train_dataloader has highest priority
    # 將額外設定的config資料添加上去
    train_loader_cfg = dict(loader_cfg, **cfg.data.get('train_dataloader', {}))

    # 構建DataLoader實例化對象
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # determine whether use adversarial training precess or not
    # 使否產生對抗性訓練，這裡現在都會是False
    use_adverserial_train = cfg.get('use_adversarial_train', False)

    # put model on gpus
    if distributed:
        # 如果是分布式訓練就會到這裡設定
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        if use_adverserial_train:
            # Use DistributedDataParallelWrapper for adversarial training
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        # 非分布式訓練就會到這裡設定
        if digit_version(mmcv.__version__) >= digit_version(
                '1.4.4') or torch.cuda.is_available():
            model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        else:
            warnings.warn(
                'We recommend to use MMCV >= 1.4.4 for CPU training. '
                'See https://github.com/open-mmlab/mmpose/pull/1157 for '
                'details.')

    # build runner
    # 構建優化器實例對象
    optimizer = build_optimizers(model, cfg.optimizer)

    # 構建以epoch為主的runner
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    # 將當前時間戳放到runner當中
    runner.timestamp = timestamp

    if use_adverserial_train:
        # The optimizer step process is included in the train_step function
        # of the model, so the runner should NOT include optimizer hook.
        optimizer_config = None
    else:
        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

    # 獲取用戶自定義的鉤子函數
    custom_hooks_cfg = cfg.get('custom_hooks', None)
    if custom_hooks_cfg is None:
        # 如果有設定就會到這裡
        custom_hooks_cfg = cfg.get('custom_hooks_config', None)
        if custom_hooks_cfg is not None:
            warnings.warn(
                '"custom_hooks_config" is deprecated, please use '
                '"custom_hooks" instead.', DeprecationWarning)

    # register hooks
    # 將鉤子函數添加上去
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=custom_hooks_cfg)

    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # 如果有需要進行驗證就會到這裡，將驗證資料放入
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            drop_last=False,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        # 如果有需要載入上次未訓練完的結果就會到這裡
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        # 如果是要單純加載模型參數就會到這裡
        runner.load_checkpoint(cfg.load_from)
    # 開始進行訓練
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
