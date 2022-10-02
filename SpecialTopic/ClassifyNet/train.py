import argparse
import torch
from torch import nn
import os
from torch.utils.data import DataLoader
from utils_fit import fit_one_epoch
from SpecialTopic.ST.utils import get_classes
from SpecialTopic.ST.net.lr_scheduler import get_lr_scheduler_yolox, set_optimizer_lr_yolox
from SpecialTopic.ST.build import build_detector, build_dataset


def parse_args():
    parser = argparse.ArgumentParser('使用分類網路進行剩餘量判斷')
    # 選擇使用的模型主幹
    parser.add_argument('--model-type', type=str, default='ResNet')
    # 使用的模型大小，這裡支援的尺寸會與使用的模型主幹有關
    parser.add_argument('--phi', type=str, default='m')
    # batch size，盡量調整到超過4，這樣BN層才不會出問題
    parser.add_argument('--batch-size', type=int, default=2)
    # 加載預訓練權重，這裡指的是在ImageNet上預訓練權重
    parser.add_argument('--pretrained', type=str, default='./pretrained.pth')
    # 這裡是可以加載上次訓練到一半的權重
    parser.add_argument('--load-from', type=str, default='none')
    # 提供類別文件
    parser.add_argument('--classes-path', type=str, default='./classes.txt')
    # 如果在標註文件中的圖像路徑使用的是相對路徑，可以透過data-prefix變成絕對路徑
    parser.add_argument('-data-prefix', type=str, default='')
    # 訓練標註文件位置
    parser.add_argument('--train-annotation-path', type=str, default='./train_annotation.txt')
    # 驗證標註文件位置，如果設定none就會自動拿train同時作為驗證
    parser.add_argument('--val-annotation-path', type=str, default='none')
    # 如果有檢測到gpu訓練就會自動啟動fp16，這樣可以節省兩倍記憶體空間
    parser.add_argument('--auto-fp16', action='store_false')

    # 起始Epoch數
    parser.add_argument('--Init-Epoch', type=int, default=0)
    # 總共要訓練多少個Epoch
    parser.add_argument('--Total-Epoch', type=int, default=100)
    # 最大學習率
    parser.add_argument('--Init-lr', type=float, default=1e-2)
    # 指定使用優化器類別
    parser.add_argument('--optimizer-type', type=str, default='sgd')
    # 學習率下降曲線
    parser.add_argument('--lr-decay-type', type=str, default='cos')

    # 多少個Epoch會強制保存
    parser.add_argument('--save-period', type=int, default=5)
    # 是否需要保存訓練時最小的loss權重參數
    parser.add_argument('--best-train-loss', action='store_true')
    # 是否需要保存驗證時最小的loss權重參數
    parser.add_argument('--best-val-loss', action='store_false')
    # 是否需要保存優化器狀態
    parser.add_argument('--save-optimizer', action='store_true')
    # 權重保存位置
    parser.add_argument('--save-path', type=str, default='./save')
    # 權重名稱，會變成權重檔案主要名稱，可以提供辨識效果
    parser.add_argument('--weight-name', type=str, default='auto')

    # 在使用DataLoader時的cpu數
    parser.add_argument('--num-workers', type=int, default=1)
    # 最終輸入到網路的圖像大小
    parser.add_argument('--input-shape', type=int, default=[224, 224], nargs='+')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fp16 = torch.cuda.is_available() if args.auto_fp16 else False
    _, num_classes = get_classes(args.classes_path)
    model_cfg = {
        'type': args.model_type,
        'phi': args.phi,
        'num_classes': num_classes,
        'pretrained': args.pretrained
    }
    model = build_detector(model_cfg)
    model = model.to(device)
    # 測試模型輸入
    # imgs = torch.randn((2, 3, 224, 224))
    # labels = torch.LongTensor([1, 2])
    # loss = model(imgs, labels, with_loss=True)
    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    batch_size = args.batch_size
    Init_lr = args.Init_lr
    Min_lr = Init_lr * 0.01
    nbs = 64
    lr_limit_max = 1e-3 if args.optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if args.optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {
        'adam': torch.optim.Adam(pg0, Init_lr_fit, betas=(0.937, 0.999)),
        'sgd': torch.optim.SGD(pg0, Init_lr_fit, momentum=0.937, nesterov=True)
    }[args.optimizer_type]
    optimizer.add_param_group({'params': pg1, 'weight_decay': 5e-4})
    optimizer.add_param_group({'params': pg2})
    if args.load_from != 'none':
        print('加載上次訓練結果...')
        pretrained_dict = torch.load(args.load_from, map_location=device)
        if 'model_weight' in pretrained_dict:
            model_weight = pretrained_dict['model_weight']
        else:
            model_weight = pretrained_dict
        if 'optimizer_weight' in pretrained_dict:
            optimizer_weight = pretrained_dict['optimizer_weight']
            args.Init_Epoch = pretrained_dict['epoch']
        else:
            optimizer_weight = None
        model.load_state_dict(model_weight)
        if optimizer_weight is not None:
            optimizer.load_state_dict(optimizer_weight)
    lr_scheduler_func = get_lr_scheduler_yolox(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.Total_Epoch)
    assert os.path.isfile(args.train_annotation_path), '需提供合法標註文件'
    if args.val_annotation_path == 'none':
        args.val_annotation_path = args.train_annotation_path
        print('使用訓練標註文件作為驗證標註文件，如果有需要指定驗證標註文件請寫入')
    train_dataset_cfg = {
        'type': 'RemainingDataset',
        'annotation_file': args.train_annotation_path,
        'data_prefix': args.data_prefix,
        'pipeline_cfg': [
            {'type': 'LoadRemainingAnnotation', 'key': ['image', 'label']},
            {'type': 'ResizeSingle', 'input_shape': [224, 224], 'save_info': False, 'keep_ratio': True},
            {'type': 'NormalizeSingle', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'to_rgb': True},
            {'type': 'Collect', 'keys': ['image', 'label', 'image_path']}
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
    val_dataset_cfg = {
        'type': 'RemainingDataset',
        'annotation_file': args.val_annotation_path,
        'data_prefix': args.data_prefix,
        'pipeline_cfg': [
            {'type': 'LoadRemainingAnnotation', 'key': ['image', 'label']},
            {'type': 'ResizeSingle', 'input_shape': [224, 224], 'save_info': False, 'keep_ratio': True},
            {'type': 'NormalizeSingle', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'to_rgb': True},
            {'type': 'Collect', 'keys': ['image', 'label', 'image_path']}
        ]
    }
    val_dataset = build_dataset(val_dataset_cfg)
    val_dataloader_cfg = train_dataloader_cfg
    val_dataloader_cfg['dataset'] = val_dataset
    val_dataloader_cfg['shuffle'] = False
    val_dataloader = DataLoader(**val_dataloader_cfg)
    training_state = dict(train_loss=10000, val_loss=10000)
    best_train_loss = args.best_train_loss
    best_val_loss = args.best_val_loss
    save_optimizer = args.save_optimizer
    save_path = args.save_path
    weight_name = args.weight_name
    for epoch in range(args.Init_Epoch, args.Total_Epoch):
        set_optimizer_lr_yolox(optimizer, lr_scheduler_func, epoch)
        fit_one_epoch(model, device, optimizer, epoch, train_dataloader, val_dataloader, args.Total_Epoch, fp16, scaler,
                      args.save_period, save_path, training_state, best_train_loss, best_val_loss, save_optimizer,
                      weight_name)


if __name__ == '__main__':
    main()
    print('Finish')
