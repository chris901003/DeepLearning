import argparse
import pandas as pd
import os
import math
import pickle
import numpy as np


def args_parser():
    parser = argparse.ArgumentParser()
    # excel檔案路徑
    parser.add_argument('--xlsx-path', type=str, default='remain.xlsx')
    # 一段訓練長度，單位(秒)
    parser.add_argument('--time-range', type=int, default=2 * 60)
    # 保存路徑
    parser.add_argument('--save-train-path', type=str, default='./train_set.pickle')
    parser.add_argument('--save-val-path', type=str, default='./val_set.pickle')
    args = parser.parse_args()
    return args


def parse_excel(xlsx_path, time_range):
    xlsx_info = pd.read_excel(io=xlsx_path, sheet_name='工作表1')
    xlsx_info = xlsx_info.to_dict()
    weight_info = list()
    for xlsx_key, xlsx_value in xlsx_info.items():
        if not xlsx_key.startswith('Weight'):
            continue
        weights = [weight for _, weight in xlsx_value.items()]
        weight_info.append(weights)
    remain_info = weight_to_remain(weight_info, time_range)
    return remain_info


def weight_to_remain(weight_info, time_range):
    remains = list()
    for idx, weights in enumerate(weight_info):
        while math.isnan(weights[-1]):
            weights.pop()
        assert len(weights) >= time_range, f"時間長度不足{time_range}秒"
        min_weight, max_weight = weights[-1], weights[0]
        total_weight = max_weight - min_weight
        remain = [int(round((weight - min_weight) / total_weight * 100)) for weight in weights]
        remains.append(remain)
    return remains


def get_regression_train_data(remain_info, time_range):
    regression_data = list()
    for remains in remain_info:
        remain_length = len(remains)
        for idx in range(0, remain_length - time_range):
            sub_remain = remains[idx: idx + time_range]
            label = remain_length - (idx + time_range) - 1
            regression_data.append({'remain': sub_remain, 'label': label})
    return regression_data


def main():
    args = args_parser()
    xlsx_path = args.xlsx_path
    time_range = args.time_range
    save_train_path = args.save_train_path
    save_val_path = args.save_val_path
    assert os.path.exists(xlsx_path), '給定excel檔案不存在'
    remain_info = parse_excel(xlsx_path, time_range)
    regression_train_data = get_regression_train_data(remain_info, time_range)
    data_size = len(regression_train_data)
    val_set = np.random.randint(0, data_size, int(data_size * 0.3))
    train = np.array(regression_train_data)
    np.random.shuffle(train)
    train = train.tolist()
    train_data = train[int(data_size * 0.3):]
    val_data = train[:int(data_size * 0.3)]

    with open(save_train_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(save_val_path, 'wb') as f:
        pickle.dump(val_data, f)
    print("Finish")


if __name__ == '__main__':
    main()
