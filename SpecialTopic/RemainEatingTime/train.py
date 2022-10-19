import argparse
import torch
import copy
import os
from torch.utils.data import DataLoader
from SpecialTopic.ST.build import build_detector, build_dataset
from SpecialTopic.ST.net.lr_scheduler import build_lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    # 模型大小
    parser.add_argument('--phi', type=str, default='m')
    # 一個batch的大小
    parser.add_argument('--batch-size', type=int, default=2)
    # 預訓練權重位置
    parser.add_argument('--pretrained', type=str, default='none')
    # 加載上次訓練到一半的資料
    parser.add_argument('--load-from', type=str, default='none')
    # 最長傳入的長度
    parser.add_argument('--max-len', type=int, default=-1)
    # 最大剩餘時間
    parser.add_argument('--max-x-axis', type=int, default=-1)
    # 最大剩餘量
    parser.add_argument('--max-y-axis', type=int, default=-1)
    # 訓練資料位置
    parser.add_argument('--train-annotation-path', type=str, default='./train_annotation.pickle')
    # 驗證資料位置
    parser.add_argument('--eval-annotation-path', type=str, default='./eval_annotation.pickle')
    # 是否自動啟動fp16
    parser.add_argument('--auto-fp16', action='store_false')

    # 初始epoch數
    parser.add_argument('--Init-Epoch', type=int, default=0)
    # 最終epoch數
    parser.add_argument('--Total-Epoch', type=int, default=100)
    # 最大學習率
    parser.add_argument('--Init-lr', type=int, default=1e-2)
    # 優化器選擇
    parser.add_argument('--optimizer-type', type=str, default='sgd')
    # 學習率退火曲線
    parser.add_argument('--lr-decay-type', type=str, default='cos')
    # 多少個epoch強制保存權重
    parser.add_argument('--save-period', type=int, default=5)
    # 是否需要將優化器狀態保存
    parser.add_argument('--save-optimizer', action='store_true')
    # 保存路徑
    parser.add_argument('--save-path', type=str, default='./save')
    # 權重名稱
    parser.add_argument('--weight-name', type=str, default='auto')
    # 在Dataloader當中使用
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fp16 = torch.cuda.is_available() if args.auto_fp16 else False
    if args.max_x_axis != -1:
        max_x_axis = args.max_x_axis + 4
    else:
        max_x_axis = -1
    if args.max_y_axis != -1:
        max_y_axis = args.max_y_axis + 4
    else:
        max_y_axis = -1
    if args.max_len != -1:
        max_len = args.max_len + 2
    else:
        max_len = -1
    train_dataset_cfg = {
        'type': 'RemainEatingTimeDataset',
        # annotation當中會有一下資訊，主要訓練資料會放到'datas'當中
        'train_annotation_path': args.train_annotation_path,
        'dataset_variable_setting': {
            'num_remain_classes': 'Default' if max_y_axis == -1 else max_y_axis,
            'num_time_classes': 'Default' if max_x_axis == -1 else max_x_axis,
            'max_len': 'Default' if max_len == -1 else max_len,
            # 這裡的Default就會按照分類類別數從0->0到n->n
            'remain_to_index': 'Default',
            'time_to_index': 'Default',
            # 這裡的Default就會將pad的值設定成類別數+3
            'remain_pad_val': 'Default',
            'time_pad_val': 'Default',
            # 這裡的Default就會將SOS的值設定成類別數+1
            'remain_SOS_val': 'Default',
            'time_SOS_val': 'Default',
            # 這裡的Default就會將EOS的值設定成類別數+2
            'remain_EOS_val': 'Default',
            'time_EOS_val': 'Default'},
        'pipeline_cfg': [
            {'type': 'FormatRemainEatingData',
             'need_variable': {
                'max_len': 'max_len',
                'remain_to_index': 'remain_to_index', 'time_to_index': 'time_to_index',
                'remain_pad_val': 'remain_pad_val', 'time_pad_val': 'time_pad_val',
                'remain_SOS_val': 'remain_SOS_val', 'time_SOS_val': 'time_SOS_val',
                'remain_EOS_val': 'remain_EOS_val', 'time_EOS_val': 'time_EOS_val'}
             },
            {'type': 'Collect', 'keys': ['food_remain_data', 'time_remain_data']}
        ]
    }
    train_dataset = build_dataset(train_dataset_cfg)
    batch_size = args.batch_size
    train_dataloader_cfg = {
        'dataset': train_dataset,
        'batch_size': batch_size,
        'drop_last': False if batch_size > 2 else True,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'collate_fn': train_dataset.collate_fn_train
    }
    train_dataloader = DataLoader(**train_dataloader_cfg)
    if not os.path.exists(args.eval_annotation_path):
        args.eval_annotation_path = args.train_annotation_path
        print('驗證集使用與訓練集相同資料')
    eval_dataset_cfg = copy.deepcopy(train_dataset_cfg)
    eval_dataset = build_dataset(eval_dataset_cfg)
    eval_dataloader_cfg = copy.deepcopy(train_dataloader_cfg)
    eval_dataloader_cfg['dataset'] = eval_dataset
    eval_dataloader_cfg['shuffle'] = False
    eval_dataloader = DataLoader(**eval_dataloader_cfg)
    model_cfg = {
        'type': 'RemainEatingTime',
        'phi': args.phi,
        # 這裡多出來的3分別為[<SOS>, <END>, <PAD>]，原先本來就會需要多一個(0index的關係)
        # train_dataset['dataset_variable']['num_remain_classes']
        'num_remain_classes': train_dataset.dataset_variable['num_remain_classes'],
        # train_dataset['dataset_variable']['num_time_classes']
        'num_time_classes': train_dataset.dataset_variable['num_time_classes'],
        # 這裡多出來的長度會是[<SOS>, <END>]放在前後的
        # train_dataset['dataset_variable']['max_len']
        'max_len': train_dataset.dataset_variable['max_len'],
        # train_dataset['dataset_variable']['remain_pad_val']
        'remain_pad_val': train_dataset.dataset_variable['remain_pad_val'],
        # train_dataset['dataset_variable']['time_pad_val']
        'time_pad_val': train_dataset.dataset_variable['time_pad_val'],
        'pretrained': args.pretrained
    }
    model = build_detector(model_cfg)
    model = model.to(device)
    # 確認模型可以運作
    # import numpy as np
    # remain = np.random.randint(low=0, high=101, size=120)
    # remain_time = np.random.randint(low=0, high=61, size=120)
    # remain = np.concatenate(([101], remain, [102]))
    # remain_time = np.concatenate(([61], remain_time, [62], [63]))
    # remain = torch.from_numpy(remain).unsqueeze(dim=0)
    # remain_time = torch.from_numpy(remain_time).unsqueeze(dim=0)
    # loss = model(remain, remain_time[:, :-1], remain_time[:, 1:], with_loss=True)
    optimizer = {
        'sgd': torch.optim.SGD(model.parameters(), args.Init_lr, momentum=0.937, nesterov=True),
        'adam': torch.optim.Adam(model.parameters(), args.Init_lr)
    }[args.optimizer_type]
    for opti in optimizer.param_groups:
        opti['initial_lr'] = args.Init_lr
    if args.load_from != 'none' and os.path.exists(args.load_from):
        print('Loading weight from previous state')
        pretrained_dict = torch.load(args.load_from, map_location=device)
        if 'model_weights' in pretrained_dict.keys():
            model_weights = pretrained_dict['model_weights']
        else:
            model_weights = pretrained_dict
        if 'optimizer_weights' in pretrained_dict.keys():
            optimizer_weights = pretrained_dict['optimizer_weights']
            assert 'last_epoch' in pretrained_dict.keys()
            args.Init_Epoch = pretrained_dict['last_epoch']
        else:
            optimizer_weights = None
        model.load_state_dict(model_weights)
        if optimizer_weights is not None:
            optimizer.load_state_dict(optimizer_weights)
    last_epoch = args.Init_Epoch if args.Init_Epoch != 0 else -1
    lr_scheduler_cfg = {
        'type': 'StepLR', 'step_size': 5, 'gamma': 0.1, 'last_epoch': last_epoch
    }
    if lr_scheduler_cfg is not None:
        lr_scheduler = build_lr_scheduler(optimizer, lr_scheduler_cfg)
    else:
        lr_scheduler = None
    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    training_state = dict()


if __name__ == '__main__':
    main()
    print('Finish')
