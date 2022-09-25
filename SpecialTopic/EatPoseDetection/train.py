import argparse
import torch
from torch.utils.data import DataLoader
import torch
from torch import nn
from utils_fit import fit_one_epoch
from SpecialTopic.ST.net.lr_scheduler import get_lr_scheduler_yolox, set_optimizer_lr_yolox
from SpecialTopic.ST.build import build_detector, build_dataset


def args_parser():
    parser = argparse.ArgumentParser('Detect eating pose')
    # 模型預訓練權重資料
    parser.add_argument('--models-path', type=str, default='none')
    # 訓練資料標註文件
    parser.add_argument('--train-annotation', type=str, default='/Users/huanghongyan/Documents/DeepLearning/mmaction2/'
                                                                'data/kinetics400/kinetics400_train_list_videos.txt')
    # 驗證資料標註文件
    parser.add_argument('--eval-annotation', type=str, default='/Users/huanghongyan/Documents/DeepLearning/mmaction2/'
                                                                'data/kinetics400/kinetics400_val_list_videos.txt')
    # 分類類別數
    parser.add_argument('--num-classes', type=int, default=400)
    # 一個batch的大小，如果gpu的ram爆開請將此條小
    parser.add_argument('--batch-size', type=int, default=2)
    # 最大學習率
    parser.add_argument('--Init-lr', type=float, default=1e-2)
    # 總訓練次數
    parser.add_argument('--epoch', type=int, default=300)
    # 多少個Epoch後會保存權重資料
    parser.add_argument('--save-period', type=int, default=1)
    # 保存權重路徑
    parser.add_argument('--save-dir', type=str, default='./checkpoint')
    # 多少次訓練epoch後進行驗證
    parser.add_argument('--eval-period', type=int, default=1)
    # 優化器選擇
    parser.add_argument('--optimizer-type', type=str, default='sgd')
    # 學習率調整方式
    parser.add_argument('--lr-decay-type', type=str, default='cos')
    # 是否使用雙精度訓練
    parser.add_argument('--fp16', action='store_true')
    # 在DataLoader時使用的cpu核心數
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_cfg = {
        'type': 'Recognizer3D',
        'backbone': {
            'type': 'ResNet3d',
            'pretrained2d': True,
            'pretrained': '/Users/huanghongyan/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth',
            'depth': 50,
            'conv1_kernel': (5, 7, 7),
            'conv1_stride_t': 2,
            'pool1_stride_t': 2,
            'conv_cfg': {'type': 'Conv3d'},
            'norm_eval': False,
            'inflate': ((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
            'zero_init_residual': False
        },
        'cls_head': {
            'type': 'I3DHead',
            'num_classes': args.num_classes,
            'in_channels': 2048,
            'spatial_type': 'avg',
            'dropout_ratio': 0.5,
            'init_std': 0.01
        },
        'train_cfg': {},
        'test_cfg': {
            'average_clips': 'prob'
        }
    }
    model = build_detector(model_cfg)
    train_dataset_cfg = {
        'type': 'VideoDataset',
        'ann_file': args.train_annotation,
        'data_prefix': '/Users/huanghongyan/Documents/DeepLearning/mmaction2/data/kinetics400/videos_train',
        'pipeline': [
            {'type': 'PyAVInit'},
            {'type': 'SampleFrames', 'clip_len': 32, 'frame_interval': 2, 'num_clips': 1},
            {'type': 'PyAVDecode'},
            {'type': 'Recognizer3dResize', 'scale': (-1, 256)},
            {'type': 'MultiScaleCrop', 'input_size': 224, 'scales': (1, 0.8), 'random_crop': False,
             'max_wh_scale_gap': 0},
            {'type': 'Recognizer3dResize', 'scale': (224, 224), 'keep_ratio': False},
            {'type': 'Flip', 'flip_ratio': 0.5},
            {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_bgr': False},
            {'type': 'FormatShape', 'input_format': 'NCTHW'},
            {'type': 'Collect', 'keys': ['imgs', 'label']},
            {'type': 'ToTensor', 'keys': ['imgs', 'label']}
        ]
    }
    train_dataset = build_dataset(train_dataset_cfg)
    train_dataloader_cfg = {
        'dataset': train_dataset,
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': False,
        'collate_fn': train_dataset.custom_collate_fn
    }
    train_dataloader = DataLoader(**train_dataloader_cfg)
    eval_dataset_cfg = {
        'type': 'VideoDataset',
        'ann_file': args.eval_annotation,
        'data_prefix': '/Users/huanghongyan/Documents/DeepLearning/mmaction2/data/kinetics400/videos_val',
        'pipeline': [
            {'type': 'PyAVInit'},
            {'type': 'SampleFrames', 'clip_len': 32, 'frame_interval': 2, 'num_clips': 2, 'test_mode': True},
            {'type': 'PyAVDecode'},
            {'type': 'Recognizer3dResize', 'scale': (-1, 256)},
            {'type': 'ThreeCrop', 'crop_size': 256},
            {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_bgr': False},
            {'type': 'FormatShape', 'input_format': 'NCTHW'},
            {'type': 'Collect', 'keys': ['imgs', 'label']},
            {'type': 'ToTensor', 'keys': ['imgs', 'label']}
        ]
    }
    eval_dataset = build_dataset(eval_dataset_cfg)
    eval_dataloader_cfg = {
        'dataset': eval_dataset,
        'batch_size': 1,
        'shuffle': False,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': False,
        'collate_fn': train_dataset.custom_collate_fn
    }
    eval_dataloader = DataLoader(**eval_dataloader_cfg)
    if args.eval_annotation != 'none':
        pass
    if args.fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    Init_lr = args.Init_lr
    Min_lr = Init_lr * 0.01
    batch_size = args.batch_size
    nbs = 64
    lr_limit_max = 1e-3 if args.optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if args.optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    pg0, pg1, pg2 = list(), list(), list()
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm3d) or 'bn' in k:
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {
        'adam': torch.optim.Adam(pg0, Init_lr_fit, betas=(0.937, 0.999)),
        'sgd': torch.optim.SGD(pg0, Init_lr_fit, momentum=0.937, nesterov=True)
    }[args.optimizer_type]
    optimizer.add_param_group({'params': pg1, 'weight_decay': 5e-4})
    optimizer.add_param_group({'params': pg2})
    lr_scheduler_func = get_lr_scheduler_yolox(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.epoch)
    Init_Epoch = 0
    if args.models_path != 'none':
        print('Loading Pretrained')
        pretrained_dict = torch.load(args.models_path, map_location=device)
        model_pretrain_dict = pretrained_dict['model_state_dict']
        optimizer_pretrain_dict = pretrained_dict['optimizer_state_dict']
        Init_Epoch = pretrained_dict['epoch']
        model.load_state_dict(model_pretrain_dict)
        optimizer.load_state_dict(optimizer_pretrain_dict)
    for epoch in range(Init_Epoch, args.epoch):
        set_optimizer_lr_yolox(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model, device, optimizer, epoch, train_dataloader, eval_dataloader, args.epoch, args.fp16, scaler,
                      args.save_period, args.save_dir, args.eval_period)


if __name__ == '__main__':
    main()
    print('Finish')
