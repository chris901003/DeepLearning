import argparse
import json
import os
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from SpecialTopic.RemainEatingTime.RegressionModel.utils_fit import train_one_epoch, val_one_epoch
from SpecialTopic.ST.build import build_detector
from SpecialTopic.RemainEatingTime.RegressionModel.dataset import RegressionDataset
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def args_parse():
    parser = argparse.ArgumentParser('訓練回歸模型，主要使要計算剩餘時間')

    # 以下的參數與模型相關，可以隨意調整直到獲取喜歡的擬和程度
    # 訓練的總Epoch數量
    parser.add_argument('--epoch', type=int, default=100)
    # 設定學習率，通常不需要去調動
    parser.add_argument('--lr', type=float, default=1e-3)
    # 將每一個剩餘量的值轉換成多少維度的向量
    parser.add_argument('--lstm-input-size', type=int, default=32)
    # lstm當中隱藏層的channel深度
    parser.add_argument('--lstm-hidden-size', type=int, default=64)
    # lstm當中的層結構堆疊數量
    parser.add_argument('--lstm-num-layers', type=int, default=2)

    # 訓練資料集的路徑
    parser.add_argument('--training-dataset-path', type=str, default='regression_dataset.pickle')
    # 驗證資料集的路徑
    parser.add_argument('--val-dataset-path', type=str, default='regression_dataset.pickle')
    # 超參數相關設定資料位置
    parser.add_argument('--setting-path', type=str, default='setting.json')
    # 模型權重保存路徑
    parser.add_argument('--save-path', type=str, default='regression_model.pth')
    # 是否需要將訓練的過程進行可視化，可以更加的清楚當前的訓練情況
    parser.add_argument('--show', action='store_false')
    args = parser.parse_args()
    return args


def parser_setting(setting_path):
    with open(setting_path, 'r') as f:
        return json.load(f)


def main():
    args = args_parse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Epoch = args.epoch
    lr = args.lr
    training_dataset_path = args.training_dataset_path
    val_dataset_path = args.val_dataset_path
    setting_path = args.setting_path
    save_path = args.save_path
    show = args.show
    settings = parser_setting(setting_path)
    model_cfg = {
        'type': 'RemainTimeRegression',
        'input_size': args.lstm_input_size,
        'hidden_size': args.lstm_hidden_size,
        'num_layers': args.lstm_num_layers,
        'remain_time_classes': settings['remain_time_padding_value'] + 1
    }
    model = build_detector(model_cfg)
    model = model.to(device)
    assert os.path.exists(training_dataset_path), '給定的訓練資料集不存在'
    if val_dataset_path is None or not os.path.exists(val_dataset_path):
        print('未找到指定的驗證集，所以使用訓練集作為驗證集')
        val_dataset_path = training_dataset_path
    train_dataset = RegressionDataset(training_dataset_path)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1,
                                  collate_fn=train_dataset.collate_fn, pin_memory=True)
    val_dataset = RegressionDataset(val_dataset_path)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1,
                                collate_fn=val_dataset.collate_fn, pin_memory=True)
    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, Epoch + 1):
        train_one_epoch(model, epoch, Epoch, device, train_dataloader, loss_function, optimizer)
        val_one_epoch(model, epoch, Epoch, device, val_dataloader, show, save_path)


if __name__ == '__main__':
    main()
