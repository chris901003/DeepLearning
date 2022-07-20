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
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)


def parse_args():
    # 已看過
    # 初始化設定超參數
    parser = argparse.ArgumentParser(description='Train a segmentor')
    # 配置要使用的網路模型文件
    parser.add_argument('config', help='train config file path')
    # 訓練過程以及結果保存位置
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    # *暫時不確定*
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    # *暫時不確定*
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    # *暫時不確定*
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    # *暫時不確定*
    group_gpus = parser.add_mutually_exclusive_group()

    # 這裡跟gpu相關的設定都會無效，最後都只會使用一塊gpu進行訓練，默認會使用第一塊gpu進行訓練
    # 要用多少個gpu進行訓練
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    # 指定哪幾塊gpu
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    # 要使用電腦當中第幾塊gpu
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    # 亂數種子，固定亂數種子可以讓訓練復現
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    # *暫時不確定*
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    # *暫時不確定*
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # *暫時不確定*
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    # *暫時不確定*
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    # *暫時不確定*
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # *暫時不確定*
    parser.add_argument('--local_rank', type=int, default=0)
    # *暫時不確定*
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    # 已看過
    # 獲取相關超參數設定
    args = parse_args()

    # 將給定的config文件轉換成dict格式
    # cfg會有[filename, pretty_text, text]可以直接使用，剩下來的就是被保護的內容
    cfg = Config.fromfile(args.config)

    # 預設會是None
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    # 預設是不會啟用的，如果輸入網路的圖像大小都會是相同的可以開啟，這樣可以提升模型訓練速度
    # 如果是多尺度的網路模型就不建議開啟，會導致速度降低，詳細可以到網路上查詢
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    # 如果有指定保存訓練資料位置，就會在這裡進行指定
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        # 會將args指定的位置給cfg，這樣就可以知道要儲存在哪裡
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # 如果沒有指定保存位置同時cfg當中也沒有寫，就會直接在當前目錄下的work_dirs進行保存
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # 將一些args裏面的超參數移植到cfg當中，因為之後是透過cfg來構建完整訓練過程
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        # 這裡不論設定多少個gpu都會只使用單gpu，因為這個腳本就只會負責單gpu訓練
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        # 這裡只會使用單gpu且使用默認使用第一塊gpu
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    # 如果都沒有設定gpu相關內容，這裡會預設使用第一塊gpu
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # 將args內容放到cfg當中
    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    # 這裡預設是None
    if args.launcher == 'none':
        # 也就是不會啟動分布式訓練
        distributed = False
    else:
        # 多gpu才會用到這裡的東西
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # 檢查work_dir是否存在，如果不存在就會創建一個
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config，將配置文件詳細內容寫到work_dir的檔案當中
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps，記錄下當下時間並且格式化
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # log_file = log檔案名稱，名稱透過當前時間來命名
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # logger會是logger型態，這是python官方的logging模組的東西，主要就是用來記錄log的
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings，設定多線程的相關配置
    setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    # meta = 紀錄一些重要的資訊，例如系統環境或是隨機種子碼，總之就是跟log有關的內容，這裡對我們來說並不是很重要
    meta = dict()
    # log env info
    # env_info_dict = 訓練環境相關的設定，包含設備以及程式的版本號，這裡會存成dict格式
    env_info_dict = collect_env()
    # 將dict的內容透過換行符變成str格式
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    # 分隔線
    dash_line = '-' * 60 + '\n'
    # 將設備以及版本訊息放到logger當中
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    # 同時meta也會紀錄下這些資訊
    meta['env_info'] = env_info

    # log some basic info，將一些資訊也放入到logger當中
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    # cfg.device = 獲得訓練的設備
    cfg.device = get_device()
    # 設定種子碼，如果在args裏面有傳入就會直接使用args裏面的設定
    seed = init_random_seed(args.seed, device=cfg.device)
    # 單gpu下不會有任何變化
    seed = seed + dist.get_rank() if args.diff_seed else seed
    # 將seed資訊放入到logger當中記錄下來
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    # 將所有會用到隨機的部分都將種子碼設定好
    set_random_seed(seed, deterministic=args.deterministic)
    # 將種子碼也放到cfg當中
    cfg.seed = seed
    # meta記錄下種子碼訊息
    meta['seed'] = seed
    # 紀錄config副檔名
    meta['exp_name'] = osp.basename(args.config)

    # 這裡就是開始構建模型，算是MMCV的精髓
    # 傳入的會有cfg的model部分以及train_cfg和test_cfg
    # 我們這裡train_cfg以及test_cfg為None
    # model = 完整模型的實例化對象，從encoder到decoder到解碼頭到最後的loss計算全部都在model當中
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    # 將模型進行初始化，透過遞歸方式一層一層結構往下初始化
    model.init_weights()

    # SyncBN is not support for DP
    if not distributed:
        # SyncBN只有在多gpu時可以使用，所以如果是單gpu會透過revert_sync_batchnorm將SyncBN轉成BN
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        # 將模型內的SyncBN轉成BN
        model = revert_sync_batchnorm(model)

    # 將model的內容記錄下來放到logger當中
    logger.info(model)

    # datasets = 會是list型態，裏面就是完整的dataset的實例對象
    # 如果是RepeatDataset裡面會有(CLASSES, PALETTE, dataset, times)分別是(分類類別名稱, 要用哪種顏色表示, 圖像資料, 重複次數)
    # 細節部分就用Debug模式自己下去看
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        # 當workflow長度為2時表示有驗證集
        # 這裡我們拷貝一份訓練集的設定資料
        val_dataset = copy.deepcopy(cfg.data.val)
        # 將驗證集的資料處理流變成與訓練時圖片處理流相同
        val_dataset.pipeline = cfg.data.train.pipeline
        # 透過build_dataset進行構建dataset
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        # 如果有使用checkpoint_config就會在這裡保存一些訊息
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    # 將分類類別名稱添加到model當中，這樣在可視化上面可以更方便
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    # 下面就開始訓練(又是一個地獄)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
