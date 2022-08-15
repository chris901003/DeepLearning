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

from mmaction import __version__
from mmaction.apis import init_random_seed, train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import (collect_env, get_root_logger,
                            register_module_hooks, setup_multi_processes)


def parse_args():
    # 構建影片理解的輸入參數
    parser = argparse.ArgumentParser(description='Train a recognizer')
    # 指定的訓練模型config文件路徑
    parser.add_argument('config', help='train config file path')
    # 訓練過程資料保存資料夾路徑
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    # 從上次訓練到的地方繼續訓練，這裡會將優化器以及學習率都進行設定
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    # 在訓練過程中是否需要進行驗證
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    # 在最後一個訓練結束後是否進行測試
    parser.add_argument(
        '--test-last',
        action='store_true',
        help='whether to test the checkpoint after training')
    # 在結束訓練後是否需要將訓練過程中最好的權重進行測試
    parser.add_argument(
        '--test-best',
        action='store_true',
        help=('whether to test the best checkpoint (if applicable) after '
              'training'))
    # 構建一個群組，在這群組當中的參數只能設定一個
    group_gpus = parser.add_mutually_exclusive_group()
    # 使用多少個gpu進行訓練
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    # 可以使用的gpu在系統上是哪些index
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    # 隨機種子碼，如果固定種子碼可以讓結果復現
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    # 在不同的gpu上是否需要使用不同的種子碼
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    # 對於CUDNN的設定，設定為True時可以讓每次模型訓練結果相同，會將每次返回的卷積算法固定
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # 添加額外的config資料
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    # 底層的框架選擇
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # 分布式訓練會用到的參數
    parser.add_argument('--local_rank', type=int, default=0)
    # 將參數打包成args
    args = parser.parse_args()
    # 分布式訓練相關東西
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # 將打包後的參數回傳
    return args


def main():
    # 獲取啟動時傳入的參數資訊
    args = parse_args()

    # 解析config文件內容
    cfg = Config.fromfile(args.config)

    # 如果有額外添加的config資訊就會到這裡進行融合
    cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    # 設定多線程相關資料
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    # 如果有需要開啟cudnn_benchmark就會開啟，會在訓練前找到比較好計算卷積的方式，如果輸入的圖像大小都固定可以提升訓練速度
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority:
    # CLI > config file > default (base filename)
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        # 如果有傳入work_dir就會到這裡指定訓練過程輸出
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        # 如果沒有指定輸出位址且config當中也沒有指定就會到這裡，使用默認的位置
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        # 如果有傳入上次訓練到一半的保存權重檔案就會到這裡
        cfg.resume_from = args.resume_from

    if args.gpu_ids is not None or args.gpus is not None:
        # 如果有設定gpu_ids或是gpus就會到這裡
        # 官方強烈推薦使用分布式訓練，如果使用分布式訓練就不會有gpu_ids與gpus參數
        warnings.warn(
            'The Args `gpu_ids` and `gpus` are only used in non-distributed '
            'mode and we highly encourage you to use distributed mode, i.e., '
            'launch training with dist_train.sh. The two args will be '
            'deperacted.')
        if args.gpu_ids is not None:
            # 如果非分布式訓練就只能使用單gpu或是cpu進行訓練
            warnings.warn(
                'Non-distributed training can only use 1 gpu now. We will '
                'use the 1st one in gpu_ids. ')
            cfg.gpu_ids = [args.gpu_ids[0]]
        elif args.gpus is not None:
            # 如果非分布式訓練就只能使用單gpu或是cpu進行訓練
            warnings.warn('Non-distributed training can only use 1 gpu now. ')
            cfg.gpu_ids = range(1)

    # init distributed env first, since logger depends on the dist info.
    # 配置分布式訓練的環境設置
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # The flag is used to determine whether it is omnisource training
    # 設定是否使用omnisource訓練
    cfg.setdefault('omnisource', False)

    # The flag is used to register module's hooks
    # 用來註冊模型鉤子函數
    cfg.setdefault('module_hooks', [])

    # create work_dir
    # 檢查指定的work_dir是否存在，如果不存在就會進行創建
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # 將設定資料與config資料進行保存，這裡就會保存到work_dir當中
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    # 獲取當前時間戳，這裡主要是用在檔案命名上面使用
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # 構建log的檔案名稱
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # 創建logger實例對象
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # 創建meta字典，meta主要是保存一些訓練過程參數資料
    meta = dict()
    # log env info
    # 獲取當前環境相關資訊
    env_info_dict = collect_env()
    # 將環境相關資訊變成str格式
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    # 排版用的
    dash_line = '-' * 60 + '\n'
    # 保存到logger當中
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    # 也保存到meta當中
    meta['env_info'] = env_info

    # log some basic info
    # 一些基礎的log資訊
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config: {cfg.pretty_text}')

    # set random seeds
    # 進行種子碼設定
    seed = init_random_seed(args.seed, distributed=distributed)
    # 如果希望在不同gpu上有不同的種子碼就會在設定的種子碼上添加gpu的rank
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    # 設定種子碼
    set_random_seed(seed, deterministic=args.deterministic)

    # 將種子碼資訊保存到config當中
    cfg.seed = seed
    # 也保存到meta當中
    meta['seed'] = seed
    meta['config_name'] = osp.basename(args.config)
    meta['work_dir'] = osp.basename(cfg.work_dir.rstrip('/\\'))

    # 構建模型實例化對象
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        # 如果有設定模型鉤子就會到這裡進行註冊
        register_module_hooks(model, cfg.module_hooks)

    if cfg.omnisource:
        # If omnisource flag is set, cfg.data.train should be a list
        # 如果有使用omnisource就會到這裡表示傳入的訓練dataset是由多個dataset組合成的
        assert isinstance(cfg.data.train, list)
        datasets = [build_dataset(dataset) for dataset in cfg.data.train]
    else:
        # 單一個訓練dataset
        datasets = [build_dataset(cfg.data.train)]

    if len(cfg.workflow) == 2:
        # For simplicity, omnisource is not compatible with val workflow,
        # we recommend you to use `--validate`
        # 如果workflow當中有指定val就會到這裡，不過官方建議使用--validate進行訓練過程中的驗證，這裡疑似是因為有部分內容沒有實作完成
        assert not cfg.omnisource
        if args.validate:
            warnings.warn('val workflow is duplicated with `--validate`, '
                          'it is recommended to use `--validate`. see '
                          'https://github.com/open-mmlab/mmaction2/pull/123')
        # 複製一份訓練的dataset資料
        val_dataset = copy.deepcopy(cfg.data.val)
        # 構建val用的dataset
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmaction version, config file content and class names in
        # checkpoints as meta data
        # 設定checkpoint相關資訊
        cfg.checkpoint_config.meta = dict(
            mmaction_version=__version__ + get_git_hash(digits=7),
            config=cfg.pretty_text)

    # 設定測試相關的option，這裡會有是否在最後進行測試以及最後測試訓練過程中最佳的權重
    test_option = dict(test_last=args.test_last, test_best=args.test_best)
    # 開始進行訓練
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=args.validate,
        test=test_option,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
