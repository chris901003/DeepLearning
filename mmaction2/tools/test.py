# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model

from mmaction.datasets import build_dataloader, build_dataset
from mmaction.models import build_model
from mmaction.utils import (build_ddp, build_dp, default_device,
                            register_module_hooks, setup_multi_processes)

# TODO import test functions from mmcv and delete them from mmaction2
try:
    from mmcv.engine import multi_gpu_test, single_gpu_test
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from mmaction.apis import multi_gpu_test, single_gpu_test


def parse_args():
    # 已看過，測試模式下的傳入參數部分
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    # 指定使用的config文件檔案路徑
    parser.add_argument('config', help='test config file path')
    # 預訓練權重位置
    parser.add_argument('checkpoint', help='checkpoint file')
    # 測試後結果的檔案輸出位置，這裡可以支援3中輸出格式
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    # 是否需要將卷積與標準化層進行融合，這樣可以稍微提升測試速度
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    # 驗證的方式，這裡會根據不同的資料集有不同的驗證選項，可以一次選擇多種
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    # 是否要使用gpu來蒐集結果
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    # 暫存資料夾位置，從多個gpu收集的結果暫時存放的地方
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    # 其他額外的options會在dataset的evaluate當中添加上去，因為這裡是test，所以不會加到train上
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    # 這裡也是會額外加到dataset的evaluate上
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    # 添加到config文件當中的額外參數
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    # 平均測試剪輯時的平均類型
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    # 底層使用的模組
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # 分布式測試相關的東西
    parser.add_argument('--local_rank', type=int, default=0)
    # onnx格式的東西
    parser.add_argument(
        '--onnx',
        action='store_true',
        help='Whether to test with onnx model or not')
    # 是否使用TensorRT引擎
    parser.add_argument(
        '--tensorrt',
        action='store_true',
        help='Whether to test with TensorRT engine or not')
    # 將args蒐集起來
    args = parser.parse_args()
    # 跟多gpu相關的設定
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        # options與eval_options不可以同時設定
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        # 在這裡如果要設定options就設定到eval_options上，因為options主要是在訓練時使用
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    # 已看過，將cfg當中的pretrained資訊變成None，也就是不去載入預訓練權重

    if 'pretrained' in cfg:
        # 如果cfg當中有pretrained就將其設定成None
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        # 使用遞歸的方式往下看哪些地方是dict
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)


def inference_pytorch(args, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""
    # 已看過，透過pytorch模型獲取預測結果
    # args = 啟動時輸入的設定資料
    # cfg = config的設定資料
    # distributed = 是否啟用分布式測試
    # data_loader = 測試資料的Dataloader實例化對象

    if args.average_clips is not None:
        # 如果有設定average_clips就會到這裡
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # remove redundant pretrain steps for testing
    # 這裡原先會缺少gpu_ids的設定，我在這邊添加上gpu_ids
    cfg.gpu_ids = 0
    # 將pretrained關閉
    turn_off_pretrained(cfg.model)

    # build the model and load checkpoint
    # 構建模型本身
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    if len(cfg.module_hooks) > 0:
        # 如果又設定module_hooks就會到這裡來，將鉤子函數添加上去
        register_module_hooks(model, cfg.module_hooks)

    # 獲取fp16_cfg資訊
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        # 如果fp16_cfg不是None就會到這裡
        wrap_fp16_model(model)
    # 載入訓練好的權重
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        # 如果有設定將卷積層與標準化層合併就會到這裡
        model = fuse_conv_bn(model)

    if not distributed:
        # 如果不是分布式訓練就會到這裡
        # 將模型放到DP上面
        model = build_dp(
            model, default_device, default_args=dict(device_ids=cfg.gpu_ids))
        # 進行推理
        outputs = single_gpu_test(model, data_loader)
    else:
        model = build_ddp(
            model,
            default_device,
            default_args=dict(
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False))
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    # 最後將推理結果返回
    return outputs


def inference_tensorrt(ckpt_path, distributed, data_loader, batch_size):
    """Get predictions by TensorRT engine.

    For now, multi-gpu mode and dynamic tensor shape are not supported.
    """
    assert not distributed, \
        'TensorRT engine inference only supports single gpu mode.'
    import tensorrt as trt
    from mmcv.tensorrt.tensorrt_utils import (torch_device_from_trt,
                                              torch_dtype_from_trt)

    # load engine
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(ckpt_path, mode='rb') as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    # For now, only support fixed input tensor
    cur_batch_size = engine.get_binding_shape(0)[0]
    assert batch_size == cur_batch_size, \
        ('Dataset and TensorRT model should share the same batch size, '
         f'but get {batch_size} and {cur_batch_size}')

    context = engine.create_execution_context()

    # get output tensor
    dtype = torch_dtype_from_trt(engine.get_binding_dtype(1))
    shape = tuple(context.get_binding_shape(1))
    device = torch_device_from_trt(engine.get_location(1))
    output = torch.empty(
        size=shape, dtype=dtype, device=device, requires_grad=False)

    # get predictions
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        bindings = [
            data['imgs'].contiguous().data_ptr(),
            output.contiguous().data_ptr()
        ]
        context.execute_async_v2(bindings,
                                 torch.cuda.current_stream().cuda_stream)
        results.extend(output.cpu().numpy())
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return results


def inference_onnx(ckpt_path, distributed, data_loader, batch_size):
    """Get predictions by ONNX.

    For now, multi-gpu mode and dynamic tensor shape are not supported.
    """
    assert not distributed, 'ONNX inference only supports single gpu mode.'

    import onnx
    import onnxruntime as rt

    # get input tensor name
    onnx_model = onnx.load(ckpt_path)
    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    assert len(net_feed_input) == 1

    # For now, only support fixed tensor shape
    input_tensor = None
    for tensor in onnx_model.graph.input:
        if tensor.name == net_feed_input[0]:
            input_tensor = tensor
            break
    cur_batch_size = input_tensor.type.tensor_type.shape.dim[0].dim_value
    assert batch_size == cur_batch_size, \
        ('Dataset and ONNX model should share the same batch size, '
         f'but get {batch_size} and {cur_batch_size}')

    # get predictions
    sess = rt.InferenceSession(ckpt_path)
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        imgs = data['imgs'].cpu().numpy()
        onnx_result = sess.run(None, {net_feed_input[0]: imgs})[0]
        results.extend(onnx_result)
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():
    # 已看過，測試模式從這裡開始
    # 獲取傳入的設定參數
    args = parse_args()

    if args.tensorrt and args.onnx:
        # 如果同時設定tensorRT與onnx就會報錯，兩個只能選擇一個設定
        raise ValueError(
            'Cannot set onnx mode and tensorrt mode at the same time.')

    # 將指定的config文件讀取進來
    cfg = Config.fromfile(args.config)

    # 將額外設定的config資料添加上去
    cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    # 設定多線程資料
    setup_multi_processes(cfg)

    # Load output_config from cfg
    # 獲取config當中的output_config相關資料
    output_config = cfg.get('output_config', {})
    if args.out:
        # 如果有設定out就會到這裡
        # Overwrite output_config from args.out
        # 將out資料放到output_config當中
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    # 獲取config當中的eval_config資料
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # 將args當中的eval設定融合進去
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.eval_options:
        # 如果有額外設定eval_options就會到這裡，將eval_options融合到eval_config當中
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    # 如果沒有設定output_config又沒有設定eval_config就會報錯，至少需要選擇一種輸出結果的方式，如果都沒有設定即使程式跑完也不會有任何結果
    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    # 獲取測試的dataset的樣式
    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        # 如果output_config當中有out就會到這裡
        if 'output_format' in output_config:
            # 如果output_config當中有output_format就會到這裡
            # ugly workround to make recognition and localization the same
            # 這裡會跳出警告
            warnings.warn(
                'Skip checking `output_format` in localization task.')
        else:
            # 如果當中沒有output_format就會到這裡
            # 獲取output_config當中的out
            out = output_config['out']
            # make sure the dirname of the output path exists
            # 檢查輸出的檔案位置是否存在，如果不存在就會創建一個
            mmcv.mkdir_or_exist(osp.dirname(out))
            # 將out字串進行分割
            _, suffix = osp.splitext(out)
            if dataset_type == 'AVADataset':
                # 如果當前使用的dataset_type是AVA就會到這裡，如果out的副檔名不是csv就會報錯，因為該dataset就只能輸出csv格式
                assert suffix[1:] == 'csv', ('For AVADataset, the format of '
                                             'the output file should be csv')
            else:
                # 其他dataset就會到這裡，如果有在file_handlers當中就沒有問題
                assert suffix[1:] in file_handlers, (
                    'The format of the output '
                    'file should be json, pickle or yaml')

    # set cudnn benchmark
    # 如果有設定cudnn benchmark就會到這裡，將其打開
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        # 如果launcher是none就會將分布式訓練設定成False
        distributed = False
    else:
        # 否則就會將分布式訓練設定成True
        distributed = True
        # 進行初始化
        init_dist(args.launcher, **cfg.dist_params)

    # The flag is used to register module's hooks
    # 設定預設的鉤子
    cfg.setdefault('module_hooks', [])

    # build the dataloader
    # 構建訓練用的dataset
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # 構建dataloader需要的config資料
    dataloader_setting = dict(
        # 設定每個gpu的batch_size
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        # 設定每個gpu會使用多少個進程進行讀取影片
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        # 是否使用分布式測試
        dist=distributed,
        # 是否需要隨機排序
        shuffle=False)
    # 將config當中的dataloader設定添加上去
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    # 根據設定好的dataloader_setting實例化Dataloader
    data_loader = build_dataloader(dataset, **dataloader_setting)

    if args.tensorrt:
        # 如果使用tensorRT就會到這裡
        outputs = inference_tensorrt(args.checkpoint, distributed, data_loader,
                                     dataloader_setting['videos_per_gpu'])
    elif args.onnx:
        # 如果是要onnx就會到這裡
        outputs = inference_onnx(args.checkpoint, distributed, data_loader,
                                 dataloader_setting['videos_per_gpu'])
    else:
        # 兩個都沒有就會到這裡
        outputs = inference_pytorch(args, cfg, distributed, data_loader)

    # 檢查當前進程
    rank, _ = get_dist_info()
    if rank == 0:
        # 只有主進程才會進行計算eval結果
        if output_config.get('out', None):
            # 如果有需要將結果寫入檔案就會到這裡
            out = output_config['out']
            print(f'\nwriting results to {out}')
            # 透過dump_results將結果寫入檔案
            dataset.dump_results(outputs, **output_config)
        if eval_config:
            # 如果有設定eval_config就會到這裡，透過dataset中的evaluate進行評估
            eval_res = dataset.evaluate(outputs, **eval_config)
            for name, val in eval_res.items():
                # 最後打印結果
                print(f'{name}: {val:.04f}')


if __name__ == '__main__':
    main()
