import argparse
import torch
from SpecialTopic.ST.build import build_detector


def parse_args():
    parser = argparse.ArgumentParser()
    # 模型大小
    parser.add_argument('--phi', type=str, default='m')
    # 一個batch的大小
    parser.add_argument('--batch-size', type=int, default=8)
    # 預訓練權重位置
    parser.add_argument('--pretrained', type=str, default='none')
    # 加載上次訓練到一半的資料
    parser.add_argument('--load-from', type=str, default='none')
    # 最長傳入的長度
    parser.add_argument('--max-len', type=int, default=120)
    # 最大剩餘時間
    parser.add_argument('--max-x-axis', type=int, default=60)
    # 最大剩餘量
    parser.add_argument('--max-y-axis', type=int, default=100)
    # 訓練資料位置
    parser.add_argument('--train-annotation-path', type=str, default='./train_annotation.pkt')
    # 驗證資料位置
    parser.add_argument('--eval-annotation-path', type=str, default='./eval_annotation.pkt')
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
    max_x_axis, max_y_axis = args.max_x_axis, args.max_y_axis
    model_cfg = {
        'type': 'RemainEatingTime',
        'phi': args.phi,
        # 這裡多出來的3分別為[<SOS>, <END>, <PAD>]，原先本來就會需要多一個(0index的關係)
        'num_remain_classes': max_y_axis + 4,
        'num_time_classes': max_x_axis + 4,
        # 這裡多出來的長度會是[<SOS>, <END>]放在前後的
        'max_len': args.max_len + 2,
        # train_dataset.remain_pad_val
        'remain_pad_val': 103,
        # train_dataset.time_pad_val
        'time_pad_val': 63,
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
    train_dataset_cfg = {
        'train_annotation_path': args.train_annotation_path
    }


if __name__ == '__main__':
    main()
    print('Finish')
