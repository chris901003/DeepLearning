import argparse
import copy
import torch
import os
from torch.utils.data import DataLoader
from SpecialTopic.SegmentationNet.utils_fit import fit_one_epoch
from SpecialTopic.ST.utils import get_classes, get_model_cfg, get_logger
from SpecialTopic.ST.build import build_detector, build_dataset
from SpecialTopic.ST.net.lr_scheduler import build_lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    # 選用模型主幹，目前支援[Segformer]
    parser.add_argument('--model-type', type=str, default='Segformer')
    # 模型大小，根據不同模型會有不同可以使用的大小
    parser.add_argument('--phi', type=str, default='m')
    # 一個batch的大小，如果顯存不夠就將這裡條小
    parser.add_argument('--batch-size', type=int, default=2)
    # 多少個batch會進行權重更新，可以透過這種方式模擬大batch size的情況，通常可以增加正確率
    parser.add_argument('--optimizer-step-period', type=int, default=1)
    # 預訓練權重，這裡給的會是主幹的預訓練權重
    parser.add_argument('--pretrained', type=str, default='/Users/huanghongyan/Downloads/segformer_mit-b2_512x512_16'
                                                          '0k_ade20k_20220620_114047-64e4feca.pth')
    # 如果要從上次訓練斷掉的地方繼續訓練就將權重文件放到這裡
    parser.add_argument('--load-from', type=str, default='none')
    # 分類類別文件
    parser.add_argument('--classes-path', type=str, default='./classes.txt')
    # 訓練圖像資料的前綴路徑，為了可以將標註文件內容寫成相對路徑所使用
    parser.add_argument('--data-prefix', type=str, default='')
    # 訓練使用的標註文件
    parser.add_argument('--train-annotation-path', type=str, default='./train_annotation.txt')
    # 驗證使用的標註文件，如果沒有找到該標註文件就會使用訓練文件當作驗證文件
    parser.add_argument('--eval-annotation-path', type=str, default='./eval_annotation.txt')
    # 自動使用fp16，如果沒有關閉就會在使用gpu訓練時自動開啟，開啟後可以節省一半的顯存
    parser.add_argument('--auto-fp16', action='store_false')
    # 自動使用cudnn，當檢測到使用gpu訓練時會自動開啟cudnn，如果模型當中沒有使用到卷積層就不會有效果
    parser.add_argument('--auto-cudnn', action='store_false')

    # 起始的Epoch數
    parser.add_argument('--Init-Epoch', type=int, default=0)
    # 在多少個Epoch前會將主幹進行凍結，只會訓練分類頭部分
    parser.add_argument('--Freeze-Epoch', type=int, default=50)
    # 總共會經過多少個Epoch
    parser.add_argument('--Total-Epoch', type=int, default=100)
    # 最大學習率
    parser.add_argument('--Init-lr', type=int, default=1e-2)
    # 優化器選擇類型
    parser.add_argument('--optimizer-type', type=str, default='sgd')
    # 學習率下降方式
    parser.add_argument('--lr-decay-type', type=str, default='cos')

    # 多少個Epoch後會強制保存權重
    parser.add_argument('--save-period', type=int, default=10)
    # 是否要保存訓練時的最小loss權重
    parser.add_argument('--best-train-loss', action='store_true')
    # 是否要保存驗證時的最小loss權重
    parser.add_argument('--best-eval-loss', action='store_false')
    # 是否需要將優化器同時保存，為了之後可以繼續訓練
    parser.add_argument('--save-optimizer', action='store_true')
    # 檔案保存路徑
    parser.add_argument('--save-path', type=str, default='./checkpoint')
    # 給保存的權重命名，比較好分類
    parser.add_argument('--weight-name', type=str, default='auto')

    # DataLoader中要使用的cpu核心數
    parser.add_argument('--num-workers', type=int, default=1)
    # 是否需要將訓練過程傳送email
    parser.add_argument('--send-email', action='store_true')
    # 要使用哪個電子郵件發送
    parser.add_argument('--email-sender', type=str, default='none')
    # 發送密碼
    parser.add_argument('--email-key', type=str, default='none')
    # 要發送的對象，這裡可以是多個人
    parser.add_argument('--send-to', type=str, default=[], nargs='+')
    # 多少個Epoch後會將log資料進行保存
    parser.add_argument('--save-log-period', type=int, default=5)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fp16 = torch.cuda.is_available() if args.auto_fp16 else False
    if args.auto_cudnn and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    _, num_classes = get_classes(args.classes_path)
    model_cfg = get_model_cfg(model_type=args.model_type, phi=args.phi)
    model_cfg['pretrained'] = args.pretrained
    model_cfg['decode_head']['num_classes'] = num_classes
    model = build_detector(model_cfg)
    model = model.to(device)
    # image = torch.rand((2, 3, 512, 512))
    # target = torch.randint(0, 9, size=(2, 1, 512, 512))
    # output = model(image, target)
    batch_size = args.batch_size
    Freeze_batch_size = batch_size
    Unfreeze_batch_size = batch_size // 2
    Init_lr = args.Init_lr
    Min_lr = Init_lr * 0.01
    UnFreeze_flag = args.Init_Epoch < args.Freeze_Epoch
    if args.Init_Epoch < args.Freeze_Epoch:
        for param in model.backbone.parameters():
            param.require_grad = False
    batch_size = Freeze_batch_size if UnFreeze_flag else Unfreeze_batch_size
    nbs = 64
    lr_limit_max = 1e-3 if args.optimizer_type == 'adamW' else 5e-2
    lr_limit_min = 3e-4 if args.optimizer_type == 'adamW' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    optimizer = {
        'adamW': torch.optim.AdamW(model.parameters(), Init_lr_fit),
        'sgd': torch.optim.SGD(model.parameters(), Init_lr_fit, momentum=0.937, nesterov=True)
    }[args.optimizer_type]
    for opti in optimizer.param_groups:
        opti['initial_lr'] = Init_lr_fit
    if args.load_from != 'none' and os.path.exists(args.load_from):
        print('Loading weight from previous state')
        pretrained_dict = torch.load(args.load_from, map_location=device)
        if 'model_weights' in pretrained_dict.keys():
            model_weights = pretrained_dict['model_weights']
        else:
            model_weights = pretrained_dict
        if 'optimizer_weights' in pretrained_dict.keys():
            optimizer_weights = pretrained_dict['optimizer_weights']
            assert 'last_epoch' in pretrained_dict.keys(), '需提供最後訓練時的Epoch'
            args.Init_Epoch = pretrained_dict['last_epoch']
        else:
            optimizer_weights = None
        model.load_state_dict(model_weights)
        if optimizer_weights is not None:
            optimizer.load_state_dict(optimizer_weights)
    last_epoch = args.Init_Epoch if args.Init_Epoch != 0 else -1
    lr_scheduler_cfg = None
    if lr_scheduler_cfg is not None:
        lr_scheduler = build_lr_scheduler(model, lr_scheduler_cfg)
    else:
        lr_scheduler = None
    train_dataset_cfg = {
        'type': 'SegformerDataset',
        'annotation_file': args.train_annotation_path,
        'data_name': 'ADE20KDataset',
        'data_prefix': args.data_prefix,
        'pipeline': [
            {'type': 'LoadImageFromFileSegformer'},
            {'type': 'LoadAnnotationsSegformer', 'reduce_zero_label': True},
            {'type': 'ResizeMMlab', 'img_scale': (2048, 512), 'ratio_range': (0.5, 2.0)},
            {'type': 'RandomCropMMlab', 'crop_size': (512, 512), 'cat_max_ratio': 0.75},
            {'type': 'RandomFlipMMlab', 'prob': 0.5},
            {'type': 'PhotoMetricDistortionSegformer'},
            {'type': 'NormalizeMMlab', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True},
            {'type': 'PadMMlab', 'size': (512, 512), 'pad_val': 0, 'seg_pad_val': 255},
            {'type': 'Collect', 'keys': ['img', 'gt_sematic_seg']}
        ]
    }
    train_dataset = build_dataset(train_dataset_cfg)
    train_dataloader_cfg = {
        'dataset': train_dataset,
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': True if batch_size == 2 else False,
        'collate_fn': train_dataset.train_collate_fn
    }
    train_dataloader = DataLoader(**train_dataloader_cfg)
    if not os.path.exists(args.eval_annotation_path):
        print('使用訓練集同時作為驗證集')
        args.eval_annotation_path = args.train_annotation_path
    eval_dataset_cfg = copy.deepcopy(train_dataset_cfg)
    eval_dataset_cfg['annotation_file'] = args.eval_annotation_path
    eval_dataloader_cfg = copy.deepcopy(train_dataloader_cfg)
    eval_dataset = build_dataset(eval_dataset_cfg)
    eval_dataloader_cfg['dataset'] = eval_dataset
    eval_dataloader = DataLoader(**eval_dataloader_cfg)
    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    training_state = dict(train_loss=10000, val_loss=10000)
    best_train_loss = args.best_train_loss
    best_val_loss = args.best_eval_loss
    save_optimizer = args.save_optimizer
    save_info = {
        'train_loss': list(), 'val_loss': list(), 'train_acc': list(), 'val_acc': list()
    }
    if args.send_email:
        logger = get_logger(save_info=save_info, email_sender=args.email_sender, email_key=args.email_key)
    else:
        logger = get_logger(save_info=save_info)
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    optimizer_step_period = args.optimizer_step_period
    assert optimizer_step_period >= 1, '多少個Epoch進行學習率調整需要至少大於1'
    for epoch in range(args.Init_Epoch, args.Total_Epoch):
        if epoch > args.Freeze_Epoch and not UnFreeze_flag:
            batch_size = Unfreeze_batch_size
            nbs = 64
            lr_limit_max = 1e-3 if args.optimizer_type == 'adamW' else 5e-2
            lr_limit_min = 3e-4 if args.optimizer_type == 'adamW' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            last_epoch = epoch
            lr_scheduler_cfg = None
            if lr_scheduler_cfg is not None:
                lr_scheduler = build_lr_scheduler(model, lr_scheduler_cfg)
            else:
                lr_scheduler = None
            for param in model.parameters():
                param.require_grad = True
            train_dataloader_cfg['batch_size'] = batch_size
            train_dataloader = DataLoader(**train_dataloader_cfg)
        fit_one_epoch(model, device, optimizer, optimizer_step_period, epoch, args.Total_Epoch, train_dataloader,
                      eval_dataloader, fp16, scaler, args.save_period, save_path, training_state, best_train_loss,
                      best_val_loss, save_optimizer, args.weight_name, logger, args.send_to, args.save_log_period)
        if lr_scheduler is not None:
            lr_scheduler.step()


if __name__ == '__main__':
    main()
