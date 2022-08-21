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

from mmpose import __version__
from mmpose.apis import init_random_seed, train_model
from mmpose.datasets import build_dataset
from mmpose.models import build_posenet
from mmpose.utils import collect_env, get_root_logger, setup_multi_processes


def parse_args():
    # 關節點偵測訓練參數
    parser = argparse.ArgumentParser(description='Train a pose model')
    # 選擇模型配置文件
    parser.add_argument('config', help='train config file path')
    # 訓練過程資料保存位置
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    # 載入先前訓練到一半的權重資料，會將優化器以及學習率同時載入
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    # 如果不需要進行驗證就設定成True
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    # 這裡gpus與gpu_ids只能選擇一個進行設定
    group_gpus = parser.add_mutually_exclusive_group()
    # 使用gpu的數量
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    # 指定使用哪幾個gpu進行訓練
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    # 指定使用哪個gpu進行訓練
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    # 種子碼設定
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    # 使否需要在不同gpu上使用不同的種子碼
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    # CUDNN的相關設定
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # 添加其他config資料在原先讀取的config資料上
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    # 分布式訓練會用到的東西
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # 分布式訓練相關東西
    parser.add_argument('--local_rank', type=int, default=0)
    # 在多gpu時會根據gpu數量調整學習率，通常batch_size越大或是多gpu時會將學習率調高
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    # 打包參數
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        # 多gpu設定
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # 將設定參數回傳
    return args


def main():
    # 訓練關節點檢測模型
    # 獲取預先設定參數
    args = parse_args()

    # 將模型config資料進行讀入
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        # 如果在args當中有需要額外添加config資料就會透過merge_from_dict將設定資料放入cfg當中
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    # 設定多線程資料
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    # 如果有設定開啟cudnn_benchmark就會到這裡
    if cfg.get('cudnn_benchmark', False):
        # 找到最佳的卷積計算方式，如果輸入圖像大小固定會加速推理
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        # 如果有指定work_dir就會將設定的路徑放到cfg當中保存
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        # 如果沒有設定work_dir同時cfg當中也沒有就對到這裡，使用預設的路徑作為保存的地方
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        # 如果有設定resume_from資料就會到這裡，將資料放到cfg當中
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        # 這裡只有支援單gpu或是cpu，所以即使設定多gpu也只會使用單gpu進行訓練
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        # 這裡只有支援單gpu或是cpu，所以即使設定多gpu也只會使用單gpu進行訓練
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        # 這裡只有支援單gpu或是cpu，所以即使設定多gpu也只會使用單gpu進行訓練
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # 如果有設定根據gpu數量調整學習率就會到這裡
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        # 如果launcher是none表示不啟用分布式訓練
        distributed = False
        if len(cfg.gpu_ids) > 1:
            # 如果設定成多gpu，這裡會跳出警告，表示這裡只支援單gpu訓練
            warnings.warn(
                f'We treat {cfg.gpu_ids} as gpu-ids, and reset to '
                f'{cfg.gpu_ids[0:1]} as gpu-ids to avoid potential error in '
                'non-distribute training time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        # 如果有設定分布式訓練相關資料就會到這裡
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    # 檢查指定的work_dir是否存在，如果不存在就會構建一個
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # init the logger before other steps
    # 獲取當前時間作為檔案的名稱
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # 設定log檔案的檔案名稱
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # 構建logger實例化對象
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # 構建meta字典，主要是保留訓練當中的參數值
    meta = dict()
    # log env info
    # 保存環境參數，保括訓練設備
    env_info_dict = collect_env()
    # 下面是logger保存的資料
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    # 將亂數種子設定，如果有指定的種子碼就可以在之後將結果復現
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    # 構建模型實例對象
    model = build_posenet(cfg.model)
    # 構建dataset資料
    datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # 如果有需要載入預訓練權重就會到裡面進行加載
    if cfg.checkpoint_config is not None:
        # save mmpose version, config file content
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmpose_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text,
        )
    # 開始訓練模型
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
