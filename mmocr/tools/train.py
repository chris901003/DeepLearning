#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash

from mmocr import __version__
from mmocr.apis import init_random_seed, train_detector
from mmocr.datasets import build_dataset
from mmocr.models import build_detector
from mmocr.utils import (collect_env, get_root_logger, is_2dlist,
                         setup_multi_processes)


def parse_args():
    # 已看過，文字辨識OCR的參數
    parser = argparse.ArgumentParser(description='Train a detector.')
    # config文件選擇，會根據指定的config文件構建模型以及訓練過程
    parser.add_argument('config', help='Train config file path.')
    # 訓練過程資料保存位置，如果沒有指定會自動找地方放
    parser.add_argument('--work-dir', help='The dir to save logs and models.')
    # 預訓練權重檔案位置，只會加載模型權重不會加載優化器以及學習率的資料
    parser.add_argument(
        '--load-from', help='The checkpoint file to load from.')
    # 從上次訓練到的地方繼續訓練，會加載優化器狀態以及學習率資料
    parser.add_argument(
        '--resume-from', help='The checkpoint file to resume from.')
    # 不進行驗證
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Whether not to evaluate the checkpoint during training.')
    # 多gpu相關設定
    group_gpus = parser.add_mutually_exclusive_group()
    # 使用多少個gpu進行訓練
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training).')
    # 指定的gpu ids
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    # 指定的gpu id
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    # 種子碼，可以透過指定種子碼將每次訓練結果復現
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    # 當使用分布式訓練時是否在不同的種子碼
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    # 是否指定CUDNN的backend
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Whether to set deterministic options for CUDNN backend.')
    # 其他額外設定
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    # 對於config的額外設定
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be of the form of either '
        'key="[a,b]" or key=a,b .The argument also allows nested list/tuple '
        'values, e.g. key="[(a,b),(c,d)]". Note that the quotation marks '
        'are necessary and that no white space is allowed.')
    # 使用的主要底層
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Options for job launcher.')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    # 多gpu相關的東西
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        # 不能同時設定options與cfg_options
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        # 新版本已經不支援options設定
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    # 已看過，訓練OCR模型的主函數
    # 獲取args內容
    args = parse_args()

    # 解析指定的config文件
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    setup_multi_processes(cfg)

    # set cudnn_benchmark，如果輸入到模型當中的圖像是單尺度開啟後可以加快模型訓練速度
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # 如果有設定保存位置就會直接給到config當中
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # 如果沒有特別指定就會自動找地方存放
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        # 如果有設定加載的預訓練權重就會給到config當中
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        # 如果有設定加載上次訓練到一半的訓練內容就會給到config當中
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        # 設定gpu
        cfg.gpu_ids = range(1)
        # 如果直接使用train.py進行訓練只支持單gpu或是cpu進行訓練
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        # 使用的gpu ids
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        # gpu相關設定
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        # 不啟用分布式訓練
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    # 檢查保存資料資料夾是否存在，如果不存在就會創建一個
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # 將config內容寫入到保存資料夾當中
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    # 獲取當前時間作為檔案命名使用
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log檔案的名稱
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # logger用來紀錄訓練過程的資訊
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # meta = 訓練中的一些資料都會存在裏面
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    # 獲取種子碼，如果有預先進行設定就會使用設定好的
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    # 設定種子碼
    set_random_seed(seed, deterministic=args.deterministic)
    # 保存種子碼
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # 構建模型
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    # 對模型進行初始化權重，如果需要加載預訓練權重也會在這裡進行
    model.init_weights()

    # 構建訓練使用的dataset
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        # 如果workflow當中同時有訓練以及驗證就會進來
        val_dataset = copy.deepcopy(cfg.data.val)
        if cfg.data.train.get('pipeline', None) is None:
            if is_2dlist(cfg.data.train.datasets):
                train_pipeline = cfg.data.train.datasets[0][0].pipeline
            else:
                train_pipeline = cfg.data.train.datasets[0].pipeline
        elif is_2dlist(cfg.data.train.pipeline):
            train_pipeline = cfg.data.train.pipeline[0]
        else:
            train_pipeline = cfg.data.train.pipeline

        if val_dataset['type'] in ['ConcatDataset', 'UniformConcatDataset']:
            for dataset in val_dataset['datasets']:
                dataset.pipeline = train_pipeline
        else:
            val_dataset.pipeline = train_pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.get('checkpoint_config', None) is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        # 如果有設定checkpoint_config就會到這裡設定checkpoint的設定
        cfg.checkpoint_config.meta = dict(
            mmocr_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    # 將CLASSES資訊放到model當中
    model.CLASSES = datasets[0].CLASSES
    # 準備開始進行訓練
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
