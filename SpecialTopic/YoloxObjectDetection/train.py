import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import get_classes
from ST.build import build_detector, build_loss, build_dataset
from ST.net.weight_init import weights_init_yolox
import numpy as np
from torch.backends import cudnn
from ST.net.lr_scheduler import get_lr_scheduler_yolox, set_optimizer_lr_yolox
from utils_fit import fit_one_epoch


def parse_args():
    parser = argparse.ArgumentParser('YoloX Training')
    # 比較常需要調整的部分
    parser.add_argument('--models-path', type=str, default='/Users/huanghongyan/Downloads/best_weight.pth')
    parser.add_argument('--phi', type=str, default='l')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--classes-path', default='/Users/huanghongyan/Downloads/food_data_flag/classes.txt', type=str)
    parser.add_argument('--train-annotation-path', default='/Users/huanghongyan/Downloads/food_data_flag/'
                                                           '2012_train.txt', type=str)
    parser.add_argument('--val-annotation-path', default='/Users/huanghongyan/Downloads/food_data_flag/2012_train.txt',
                        type=str)
    parser.add_argument('--fp16', action='store_true')

    # 與訓練過程相關
    parser.add_argument('--Init-Epoch', type=int, default=0)
    parser.add_argument('--Freeze_Epoch', type=int, default=50)
    parser.add_argument('--UnFreeze_Epoch', type=int, default=300)
    parser.add_argument('--Freeze-Train', action='store_false')
    parser.add_argument('--Init-lr', type=float, default=1e-2)
    parser.add_argument('--optimizer_type', type=str, default='sgd')
    parser.add_argument('--momentum', type=float, default=0.937)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--lr-decay-type', type=str, default='cos')

    # 保存資料以及訓練流程相關設定
    parser.add_argument('--save-period', type=int, default=10)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    parser.add_argument('--eval-period', type=int, default=10)
    parser.add_argument('--coco-json-file', type=str,
                        default='/Users/huanghongyan/Downloads/food_data_flag/val2017.json')
    parser.add_argument('--num-workers', type=int, default=1)

    # 比較不會需要調整的部分
    parser.add_argument('--Cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--sync-bn', action='store_true')
    parser.add_argument('--input-shape', default=[640, 640], nargs='+', type=int)
    parser.add_argument('--mosaic', action='store_false')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        raise NotImplementedError('目前暫未支持分布式訓練')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
    class_names, num_classes = get_classes(args.classes_path)
    model_cfg = {
        'type': 'YoloBody',
        'phi': args.phi,
        'backbone_cfg': {
            'type': 'YOLOPAFPN'
        },
        'head_cfg': {
            'type': 'YOLOXHead',
            'num_classes': num_classes
        }
    }
    model = build_detector(model_cfg)
    weights_init_yolox(model)
    if args.models_path != 'none':
        if local_rank == 0:
            print(f'Load weights {args.models_path}')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.models_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    loss_cfg = {
        'type': 'YOLOLoss',
        'num_classes': num_classes,
        'fp16': args.fp16
    }
    yolo_loss = build_loss(loss_cfg)
    if args.fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    model_train = model.train()
    if args.sync_bn and ngpus_per_node > 1 and args.distributed:
        raise NotImplementedError('分布式訓練或是多卡訓練目前不支援')
    elif args.sync_bn:
        raise NotImplementedError('分布式訓練或是多卡訓練目前不支援')
    if args.Cuda:
        if args.distributed:
            raise NotImplementedError('分布式訓練或是多卡訓練目前不支援')
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    Freeze_batch_size = args.batch_size
    Unfreeze_batch_size = args.batch_size // 2
    Init_lr = args.Init_lr
    Min_lr = Init_lr * 0.01
    UnFreeze_flag = False
    if args.Freeze_Train:
        for param in model.backbone.parameters():
            param.require_grad = False
    batch_size = Freeze_batch_size if args.Freeze_Train else Unfreeze_batch_size
    nbs = 64
    lr_limit_max = 1e-3 if args.optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if args.optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {
        'adam': torch.optim.Adam(pg0, Init_lr_fit, betas=(args.momentum, 0.999)),
        'sgd': torch.optim.SGD(pg0, Init_lr_fit, momentum=args.momentum, nesterov=True)
    }[args.optimizer_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": args.weight_decay})
    optimizer.add_param_group({"params": pg2})
    lr_scheduler_func = get_lr_scheduler_yolox(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.UnFreeze_Epoch)
    assert os.path.isfile(args.train_annotation_path), '需提供標註文件'
    if args.val_annotation_path == 'none':
        print('未指定驗證標註文件，這裡使用訓練文件作為代替')
        args.val_annotation_path = args.train_annotation_val
    with open(args.train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(args.val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    train_dataset_cfg = {
        'type': 'YoloDataset',
        'annotation_lines': train_lines,
        'mosaic': True,
        'pipeline_cfg': [
            {'type': 'LoadInfoFromAnno', 'key': 'annotation_lines'},
            {'type': 'Mosaic', 'input_shape': args.input_shape},
            {'type': 'Collect', 'keys': ['image', 'bboxes']}
        ]
    }
    train_dataset = build_dataset(train_dataset_cfg)
    train_dataloader_cfg = {
        'dataset': train_dataset,
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': False,
        'collate_fn': train_dataset.custom_collate_fn
    }
    train_dataloader = DataLoader(**train_dataloader_cfg)
    val_dataset_cfg = {
        'type': 'YoloDataset',
        'annotation_lines': val_lines,
        'mosaic': False,
        'train': False,
        'pipeline_cfg': [
            {'type': 'LoadInfoFromAnno', 'key': 'annotation_lines'},
            {'type': 'Resize', 'input_shape': args.input_shape, 'save_info': True},
            {'type': 'Collect', 'keys': ['image', 'ori_size', 'keep_ratio', 'images_path']}
        ]
    }
    val_dataset = build_dataset(val_dataset_cfg)
    val_dataloader_cfg = {
        'dataset': val_dataset,
        'batch_size': 1,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': False,
        'collate_fn': val_dataset.custom_collate_fn_val
    }
    val_dataloader = DataLoader(**val_dataloader_cfg)
    for epoch in range(args.Init_Epoch, args.UnFreeze_Epoch):
        if epoch > args.Freeze_Epoch and not UnFreeze_flag and args.Freeze_Train:
            batch_size = Unfreeze_batch_size
            nbs = 64
            lr_limit_max = 1e-3 if args.optimizer_type == 'adam' else 5e-2
            lr_limit_min = 3e-4 if args.optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            lr_scheduler_func = get_lr_scheduler_yolox(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.UnFreeze_Epoch)
            for param in model.backbone.parameters():
                param.requires_grad = True
            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError('Training data to small')
            if args.distributed:
                raise NotImplementedError
            train_dataloader_cfg['batch_size'] = batch_size
            train_dataloader = DataLoader(**train_dataloader_cfg)
            UnFreeze_flag = True
        train_dataloader.dataset.epoch_now = epoch
        set_optimizer_lr_yolox(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, epoch_step, epoch_step_val, train_dataloader,
                      val_dataloader, args.UnFreeze_Epoch, args.Cuda, args.fp16, scaler, args.save_period,
                      args.save_dir, num_classes, local_rank, args.eval_period, args.coco_json_file)


if __name__ == '__main__':
    main()
    print('Finish')
