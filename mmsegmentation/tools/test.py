# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg import digit_version
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes


# 用來驗證模型的好壞用的，可以打印出一些指標

def parse_args():
    # 初始超參數設定
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    # 選擇config文件
    parser.add_argument('config', help='test config file path')
    # 訓練好的模型權重檔案位置
    parser.add_argument('checkpoint', help='checkpoint file')
    # 保留驗證的資料
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    # 如果啟用就會使用多尺度驗證，不啟用就是單尺度驗證
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    # 用來調整輸出的格式用的，可以讓輸出更好看
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    # 設定需要驗證的指標
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    # 是否展示結果
    parser.add_argument('--show', action='store_true', help='show results')
    # 展示出畫好的圖會在哪裡展示
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
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
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # 透明度
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        # 多gpu相關內容
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        # options與cfg_options不可以同時使用且options已經要被淘汰
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        # options已經被淘汰
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    # 獲取超參數
    args = parse_args()
    # 至少要指定一種展示的方式否則驗證會沒有輸出
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        # eval與format_only只能選一個
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        # 輸出檔案的副檔名需要是.pkl
        raise ValueError('The output file must be a pkl file.')

    # 解析config文件，這裡與train相同
    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        # 如果這裡還有特殊自訂的config就會在這裡加進去
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings，多進程的相關設定
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        # 如果模型的輸入圖像大小固定，這裡就可以設定成True可以加快正向推理的速度
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        # 如果有設定aug_test這裡就會在圖像處理部分多上不同縮放比例
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        # 同時會開啟翻轉
        cfg.data.test.pipeline[1].flip = True
    # 將pretrained關閉，之後會直接統一加載傳入的訓練好的權重
    cfg.model.pretrained = None
    # 將模型狀態表示成驗證狀態
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        # 設定多gpu相關的內容
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        # 如果有設定保存的地址就會進來，rand=0表示主進程
        # 檢查檔案位置是否存在，如果檔案不存在就創建一個
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        # 獲取當前時間，用來命名檔案
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            # 如果有啟用aug_test表示有啟用多尺度的驗證
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            # 如果沒有啟用aug_test表示只有單尺度驗證
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0:
        # 如果沒有設定保存位置就會到這裡來，rank=0表示主進程
        # 這裡會默認給出一個保存資料的位置
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        # 如果該檔案位置不存在就會創建一個
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        # 用來作為檔案名稱
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            # 如果有啟用aug_test表示啟用多尺度模式
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    # 根據test的構建方式創建dataset，這裡的cfg.data.test與test當中的dataset會相同
    dataset = build_dataset(cfg.data.test)
    # The default loader config，構建預設的dataloader相關參數
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings，覆寫一些參數
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    # 構建dataloader的超參數，這裡因為是在測試所以我們會將batch_size設定成1，這裡是透過samples_per_gpu進行控制
    # 同時我們也會將shuffle關閉
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader，構建dataloader實例對象
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    # 構建模型
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # 讀取訓練好的權重
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        # 獲取CLASSES資訊並且放到model當中
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        # 如果在訓練權重上沒有紀錄就會到就會到資料集的class上面找
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        # 獲取調色版並且放到model當中
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        # 如果沒有找到就會到資料集的class上面找
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated，efficient_test已經被棄用了
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    # cityscapes好像有一種特殊的驗證方式
    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        # 跟cityscapes資料及有關係的
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    # 獲取當前驗證設備
    cfg.device = get_device()
    if not distributed:
        # 如果沒有用分布式驗證就會到這裡來，這邊會給出經告說不能用SyncBN
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        if not torch.cuda.is_available():
            # mmcv的版本要夠新才能使用cpu驗證
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        # 將模型當中的SyncBN換成普通BN
        model = revert_sync_batchnorm(model)
        # 這裡通過build_dp後外面會多包一層MMDataParallel
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        # results = list[tuple]，list長度就會是總共有多少個batch，tuple裡面存的就會是一些相交的資料，詳細可以進入到裡面看
        results = single_gpu_test(
            model,
            data_loader,
            args.show,
            args.show_dir,
            False,
            args.opacity,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        results = multi_gpu_test(
            model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            False,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        # rank=0表示在主進程上，只有在主進程才會進行記錄
        if args.out:
            # 如果有指定輸出就會到這裡來
            warnings.warn(
                'The behavior of ``args.out`` has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        if args.eval:
            # 如果有指定需要的驗證類型就會進來
            eval_kwargs.update(metric=args.eval)
            # 透過evaluate對於我們指定的計算內容進行計算
            # metric裡面會有總體的正確率以及總體的iou以及個別的正確率以及個別的iou資料
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            # 將結果以及配置文件轉成json的樣子並且輸出
            mmcv.dump(metric_dict, json_file, indent=4)
            if tmpdir is not None and eval_on_format_results:
                # remove tmp dir when cityscapes evaluation
                shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()
