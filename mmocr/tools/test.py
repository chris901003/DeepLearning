#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import multi_gpu_test

from mmocr.apis.test import single_gpu_test
from mmocr.apis.utils import (disable_text_recog_aug_test,
                              replace_image_to_tensor)
from mmocr.datasets import build_dataloader, build_dataset
from mmocr.models import build_detector
from mmocr.utils import revert_sync_batchnorm, setup_multi_processes


def parse_args():
    # 已看過，驗證模式下的參數配置
    parser = argparse.ArgumentParser(
        description='MMOCR test (and eval) a model.')
    # 選取config文件
    parser.add_argument('config', help='Test config file path.')
    # 設定訓練完成的模型權重檔案位置
    parser.add_argument('checkpoint', help='Checkpoint file.')
    # 輸出結果的檔案位置，需要是pickle格式
    parser.add_argument('--out', help='Output result file in pickle format.')
    # 是否需要將conv與bn進行融合，透過融合可以加速模型訓練，因為權重不會再改變所以可以透過數學將兩層合併為一層
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed.')
    # 要使用的gpu的id
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    # 是否需要將輸出資料格式化
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without performing evaluation. It is'
        'useful when you want to format the results to a specific format and '
        'submit them to the test server.')
    # 驗證指標，如果是文字區域檢測可以使用hmean-iou
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='The evaluation metrics. Options: \'hmean-ic13\', \'hmean-iou'
        '\' for text detection tasks, \'acc\' for text recognition tasks, and '
        '\'macro-f1\' for key information extraction tasks.')
    # 是否需要展示預測結果
    parser.add_argument('--show', action='store_true', help='Show results.')
    # 將預測結果保存在哪裡
    parser.add_argument(
        '--show-dir', help='Directory where the output images will be saved.')
    # 閾值，當預測值小於閾值的會被忽略掉，預設為0.3
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='Score threshold (default: 0.3).')
    # 是否要使用gpu收集results
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='Whether to use gpu to collect results.')
    # 暫存檔案位置，用來搜集從多線程預測出的結果
    parser.add_argument(
        '--tmpdir',
        help='The tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified.')
    # 其他額外添加的config
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into the config file. If the value '
        'to be overwritten is a list, it should be of the form of either '
        'key="[a,b]" or key=a,b. The argument also allows nested list/tuple '
        'values, e.g. key="[(a,b),(c,d)]". Note that the quotation marks '
        'are necessary and that no white space is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='Custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='Custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Options for job launcher.')
    # 多gpu相關資料
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options.')
        args.eval_options = args.options
    # 將args資料回傳
    return args


def main():
    # 已看過，獲取設定資料
    args = parse_args()

    # 以下幾種驗證內容至少需要選擇一個否則驗證是沒有作用的
    assert (
        args.out or args.eval or args.format_only or args.show
        or args.show_dir), (
            'Please specify at least one operation (save/eval/format/show the '
            'results / save the results) with the argument "--out", "--eval"'
            ', "--format-only", "--show" or "--show-dir".')

    if args.eval and args.format_only:
        # eval與format_only只能兩個選一個使用
        raise ValueError('--eval and --format_only cannot be both specified.')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        # 如果有設定out參數則out檔案需要是以pkl或是pickle為副檔名
        raise ValueError('The output file must be a pkl file.')

    # 讀取config文件當中內容
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        # 如果有需要特別添加額外config內容就會在這裡進行加入
        cfg.merge_from_dict(args.cfg_options)
    # 設定多線程驗證的內容
    setup_multi_processes(cfg)

    # set cudnn_benchmark，開啟後可以增加模型推理速度
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # 如果在config文件當中backbone有需要載入預訓練權重會先進行關閉，因為會載入整個模型的權重
    if cfg.model.get('pretrained'):
        cfg.model.pretrained = None
    # 處理neck模塊的預訓練權重，將其設定成None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated，獲取一個gpu的batch_size
    samples_per_gpu = (cfg.data.get('test_dataloader', {})).get(
        'samples_per_gpu', cfg.data.get('samples_per_gpu', 1))
    if samples_per_gpu > 1:
        # 多gpu相關設定
        cfg = disable_text_recog_aug_test(cfg)
        cfg = replace_image_to_tensor(cfg)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # 構建驗證使用的dataset
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    # 構建基本DataLoader配置參數
    default_loader_cfg = {
        # 設定種子碼以及是否將最後部分去除以及分布式訓練資訊
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           ))
    }
    # 其他config參數載入進去
    default_loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    # 最終創建DataLoader參數
    test_loader_cfg = {
        **default_loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **cfg.data.get('test_dataloader', {}),
        **dict(samples_per_gpu=samples_per_gpu)
    }

    # 構建DataLoader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # 構建模型實例對象
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    # 將模型當中所有的syncBN轉成BN
    model = revert_sync_batchnorm(model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # 載入訓練好的模型權重
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        # 如果不是使用分布式驗證就會到這裡
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        is_kie = cfg.model.type in ['SDMGR']
        # 進行單gpu或是cup的驗證
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  is_kie, args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
