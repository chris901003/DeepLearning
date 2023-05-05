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
    # 資料型態[remain, weight] => [剩餘量, 重量]
    parser.add_argument('--data-type', type=str, default='remain')
    # 保存路徑
    parser.add_argument('--save-train-path', type=str, default='./train_set.pickle')
    parser.add_argument('--save-val-path', type=str, default='./val_set.pickle')
    args = parser.parse_args()
    return args


def parse_excel_weight(xlsx_path, time_range):
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


def parse_excel_remain(xlsx_path):
    xlsx_info = pd.read_excel(io=xlsx_path, sheet_name='工作表1')
    xlsx_info = xlsx_info.to_dict()
    remain_info = list()
    for xlsx_key, xlsx_value in xlsx_info.items():
        if not xlsx_key.startswith('Verify'):
            continue
        remain = [rem for _, rem in xlsx_value.items()]
        while math.isnan(remain[-1]):
            remain.pop()
        remain_info.append(remain)
    return remain_info


def get_regression_train_data(remain_info, time_range):
    regression_data = list()
    for remains in remain_info:
        remain_length = len(remains)
        for idx in range(0, remain_length - time_range):
            sub_remain = remains[idx: idx + time_range]
            label = remain_length - (idx + time_range) - 1
            regression_data.append({'remain': sub_remain, 'elapsed_time': idx + time_range, 'label': label})
    return regression_data


def main():
    args = args_parser()
    xlsx_path = args.xlsx_path
    time_range = args.time_range
    save_train_path = args.save_train_path
    save_val_path = args.save_val_path
    data_type = args.data_type
    assert os.path.exists(xlsx_path), '給定excel檔案不存在'
    remain_info = list()
    if data_type == "weight":
        remain_info = parse_excel_weight(xlsx_path, time_range)
    elif data_type == "remain":
        remain_info = parse_excel_remain(xlsx_path)
    # tmp = parse_excel_weight('./remain2.xlsx', time_range)
    # for t in tmp:
    #     remain_info.append(t)
    regression_train_data = get_regression_train_data(remain_info, time_range)
    data_size = len(regression_train_data)
    train = np.array(regression_train_data)
    np.random.shuffle(train)
    train = train.tolist()
    train_data = train[int(data_size * 0.2):]
    val_data = train[:int(data_size * 0.2)]

    with open(save_train_path, 'wb') as f:
        pickle.dump(train_data, f)
    with open(save_val_path, 'wb') as f:
        pickle.dump(val_data, f)
    print("Finish")
    print(f"Train Count {len(train_data)}")
    print(f"Val Count {len(val_data)}")


if __name__ == '__main__':
    main()
