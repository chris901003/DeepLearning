import argparse
import pandas as pd
import os
import numpy as np
import random
import math
import pickle
import json
from SpecialTopic.RemainEatingTime.RegressionModel.GenerateData.RandomDataGenerator import check_data


def args_parser():
    parser = argparse.ArgumentParser()
    # excel檔案路徑
    # 這裡只會讀取excel當中欄位名稱以weight為開頭的資料，也就是會讀取[weight0, weight1, weight2, ...]
    parser.add_argument('--xlsx-path', type=str, default='remain.xlsx')
    # 最大檢測剩餘量時長，也就是最多可以看多少個剩餘量來判斷剩餘時間
    # 這裡我們預設一定不會拋棄已看到過的資料，所以總檢測數量不可以大於該值
    parser.add_argument('--max-length', type=int, default=120)
    # 因真實情況不會有一次就直接到底的剩餘量，所以這裡會隨機去除資料的尾端資料來模擬正在吃的過程應判斷的剩餘時間
    # Ex:
    # 輸入的剩餘量資料[100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]
    # 對應剩餘時間資料[10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    # 正常檢測時收到的資料[100, 90, 80]
    # 為了希望可以檢測出剩餘時間應該要是[10, 9, 8]
    # 所以這裡透過截斷資料就可以模擬出來
    parser.add_argument('--sampling-range', type=float, default=[0.3, 0.5, 0.7, 0.9], nargs='+')
    # 採樣點數，這裡可以選擇要採樣多少個截斷點，如果設定成-1就會隨機
    parser.add_argument('--sampling-point', type=int, default=3)
    # 保存訓練資料的路徑
    parser.add_argument('--save-dataset-path', type=str, default='regression_dataset.pickle')
    # 保存該模型設定檔的路徑
    parser.add_argument('--save-setting-path', type=str, default='setting.json')
    # 是否為檢查資料集正確性，構建好資料集後檢查使用
    parser.add_argument('--check-dataset', type=bool, default=False)
    # 是否要查看已經建立好的資料可視化，構建好資料集後檢查使用
    parser.add_argument('--visualize-dataset', type=bool, default=False)
    args = parser.parse_args()
    return args


def weight_to_remain(xlsx_info):
    """ 將重量資訊轉成剩餘量資訊
    """
    remains = list()
    for weights in xlsx_info:
        min_weight, max_weight = weights[-1], weights[0]
        total_weight = max_weight - min_weight
        remain = [int(round((weight - min_weight) / total_weight * 100, 0)) for weight in weights]
        remains.append(remain)
    return remains


def main():
    args = args_parser()
    xlsx_path = args.xlsx_path
    max_length = args.max_length
    sampling_range = args.sampling_range
    sampling_point = args.sampling_point
    save_dataset_path = args.save_dataset_path
    save_setting_path = args.save_setting_path
    if args.check_dataset:
        check_dataset(save_dataset_path, save_setting_path)
        return
    if args.visualize_dataset:
        visualize_dataset(save_dataset_path, save_setting_path)
        return

    assert os.path.exists(xlsx_path), '給定的excel檔案不存在'
    xlsx_pandas_info = pd.read_excel(io=xlsx_path, sheet_name='Sheet1')
    xlsx_dict_info = xlsx_pandas_info.to_dict()
    xlsx_info = list()
    for dict_info_key, dict_info_value in xlsx_dict_info.items():
        if not dict_info_key.startswith('Weight'):
            continue
        weights = [weight for _, weight in dict_info_value.items()]
        xlsx_info.append(weights)
    xlsx_info = weight_to_remain(xlsx_info)

    # remain相關的編碼
    remain_start_value = 101
    remain_end_value = 102
    remain_padding_value = 103

    # remain time相關編碼
    remain_time_start_value = max_length + 1
    remain_time_end_value = max_length + 2
    remain_time_padding_value = max_length + 3

    # 將最大長度擴展兩個位置，存放開始值以及結尾值
    max_length += 2

    dataset_list = list()
    for remain_info in xlsx_info:
        remain_len = len(remain_info)
        if sampling_point == -1:
            current_sampling_point = np.random.randint(low=0, high=len(sampling_point) + 1)
        else:
            current_sampling_point = sampling_point
        current_sampling_range = random.sample(sampling_range, current_sampling_point)
        current_sampling_range.append(1)

        for scope in current_sampling_range:
            right_index = math.ceil(scope * remain_len)
            right_index = min(remain_len, right_index)
            remain = remain_info[:right_index]
            remain_time = [idx for idx in range(remain_len)][::-1][:right_index]

            # 添加啟動值以及終點值以及padding值
            remain = [remain_start_value] + remain + [remain_end_value]
            remain = remain + [remain_padding_value] * max_length
            remain = remain[:max_length]
            remain_time = [remain_time_start_value] + remain_time + [remain_time_end_value]
            remain_time = remain_time + [remain_time_padding_value] * max_length
            remain_time = remain_time[:max_length]
            data = dict(remain=remain, remain_time=remain_time)
            dataset_list.append(data)
    print('Save Dataset')
    parameters = {
        'max_length': max_length,
        'remain_start_value': remain_start_value,
        'remain_end_value': remain_end_value,
        'remain_padding_value': remain_padding_value,
        'remain_time_start_value': remain_time_start_value,
        'remain_time_end_value': remain_time_end_value,
        'remain_time_padding_value': remain_time_padding_value
    }
    with open(save_dataset_path, 'wb') as f:
        pickle.dump(dataset_list, f)
    with open(save_setting_path, 'w') as f:
        json.dump(parameters, f, indent=4)
    print('Finish Generate Data')


def check_dataset(data_path='regression_dataset.pickle', setting_path='setting.json'):
    check_data(data_path, setting_path)


def visualize_dataset(data_path, setting_path):
    from matplotlib import pyplot as plt
    with open(data_path, 'rb') as f:
        dataset_info = pickle.load(f)
    with open(setting_path, 'r') as f:
        settings = json.load(f)
    remain_end_value = settings.get('remain_end_value', None)
    remain_time_end_value = settings.get('remain_time_end_value', None)
    for info in dataset_info:
        remain = info.get('remain', None)
        remain_time = info.get('remain_time', None)
        assert remain is not None and remain_time is not None, '資料錯誤'
        remain_end_index = remain.index(remain_end_value)
        remain_time_end_index = remain_time.index(remain_time_end_value)
        remain = np.array(remain[:remain_end_index])
        remain_time = np.array(remain_time[:remain_time_end_index])
        plt.gca().invert_xaxis()
        plt.plot(remain, remain_time, 'r-')
        plt.draw()
        plt.pause(0.1)
        plt.cla()


if __name__ == '__main__':
    main()
