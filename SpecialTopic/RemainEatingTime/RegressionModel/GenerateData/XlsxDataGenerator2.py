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
    # 多少秒的資料會農縮在一起做一次剩餘時間的計算，這裡會影響到整個檢測對於剩餘時間的精細度
    # 剩餘時間的精細度會與time-gap相同(單位: 秒)，但是同時也不能保證模型可以正確訓練起來
    # 對於時間較短的可以設定的精細一點，對於時間較長的相反，因為在相同吃飯時間下越精細表示最後分類的類別數越多
    parser.add_argument('--time-gap', type=int, default=5)
    # 需根據time-gap同步設定，這裡要設定的就是分類數量，最小需要設定成[吃飯時長 / time-gap](單位: 秒)
    # 也就是max-length x time-gap需大於最大影片長度，對於同個類別的time-gap與max-length需相同
    # time-gap不相同會導致時間間隔單位不相同，程式依舊可以執行但是剩餘時間不可靠
    # max-length不相同會導致剩餘時間分類數不相同，程式不可執行，因模型架構不同
    # 以預設參數來說就會是最長影片長度支援為[5 x 120](單位: 秒)的影片
    # 在預測的時候會是以5秒為一個單位進行預測
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
    parser.add_argument('--sampling-points', type=int, default=3)
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


def weight_to_remain(weight_info):
    """ 將重量資訊轉成剩餘量資訊
    """
    remains = list()
    for idx, weights in enumerate(weight_info):
        min_weight, max_weight = weights[-1], weights[0]
        total_weight = max_weight - min_weight
        assert total_weight >= 0, f'第{idx + 1}筆資料的重量變化小於等於0，請檢查是否有問題'
        remain = [int(round((weight - min_weight) / total_weight * 100, 0)) for weight in weights]
        remains.append(remain)
    return remains


def compress_time(weight_info, time_gap):
    """ 將重量資料根據time-gap進行壓縮，最終輸出的重量長度會是ceil(長度/time-gap)
    """
    compress_weight = list()
    for weights in weight_info:
        weight_record = list()
        for idx in range(0, len(weights), time_gap):
            range_weight = weights[idx: idx + time_gap]
            avg_weight = sum(range_weight) / len(range_weight)
            weight_record.append(avg_weight)
        compress_weight.append(weight_record)
    return compress_weight


def parse_excel(xlsx_path, time_gap):
    """ 讀取excel當中的資料，並且轉換成剩餘量資料
    """
    xlsx_pandas_info = pd.read_excel(io=xlsx_path, sheet_name='Sheet1')
    xlsx_dict_info = xlsx_pandas_info.to_dict()
    xlsx_info = list()
    for dict_info_key, dict_info_value in xlsx_dict_info.items():
        if not dict_info_key.startswith('Weight'):
            continue
        weights = [weight for _, weight in dict_info_value.items()]
        xlsx_info.append(weights)
    xlsx_info = compress_time(xlsx_info, time_gap)
    xlsx_info = weight_to_remain(xlsx_info)
    return xlsx_info


def main():
    args = args_parser()
    xlsx_path = args.xlsx_path
    time_gap = args.time_gap
    max_length = args.max_length
    sampling_range = args.sampling_range
    sampling_points = args.sampling_points
    save_dataset_path = args.save_dataset_path
    save_setting_path = args.save_setting_path
    if args.check_dataset:
        check_dataset(save_dataset_path, save_setting_path)
        return
    if args.visualize_dataset:
        visualize_dataset(save_dataset_path, save_setting_path)
        return
    assert os.path.exists(xlsx_path), '給定的excel檔案不存在'
    xlsx_info = parse_excel(xlsx_path, time_gap)

    # remain相關編碼
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
        assert remain_len <= max_length - 2, f'設定的長度不夠長，會導致錯誤，這裡檢測到長度為{remain_len}'
        if sampling_points == -1:
            current_sampling_point = np.random.randint(low=0, high=len(sampling_range) + 1)
        else:
            current_sampling_point = sampling_points
        current_sampling_range = random.sample(sampling_range, current_sampling_point)
        current_sampling_range.append(1)

        for scope in current_sampling_range:
            right_index = math.ceil(scope * remain_len)
            right_index = min(remain_len, right_index)
            remain = remain_info[:right_index]
            remain_time = [idx for idx in range(remain_len)][::-1][:right_index]

            # 添加啟動值以及結尾值以及padding值
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
        'time_gap': time_gap,
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
    assert os.path.exists(data_path), '給定的資料集檔案路徑不存在'
    assert os.path.exists(setting_path), '給定的設定檔案不存在'
    with open(data_path, 'rb') as f:
        dataset_info = pickle.load(f)
    with open(setting_path, 'r') as f:
        settings = json.load(f)
    time_gap = settings.get('time_gap', None)
    assert time_gap is not None, '需提供time-gap資料'
    remain_end_value = settings.get('remain_end_value', None)
    assert remain_end_value is not None, '需提供remain-end的值'
    remain_time_end_value = settings.get('remain_time_end_value', None)
    assert remain_time_end_value is not None, '需提供remain-time-end的值'
    for info in dataset_info:
        remain = info.get('remain', None)
        remain_time = info.get('remain_time', None)
        assert remain is not None and remain_time is not None, '需提供remain以及remain-time資料'
        remain_end_index = remain.index(remain_end_value)
        remain_time_end_index = remain_time.index(remain_time_end_value)
        remain = np.array(remain[:remain_end_index])
        # 這裡需要將剩餘時間根據time-gap進行縮放，這樣得出的結果才會是正確的尺度
        remain_time = np.array(remain_time[:remain_time_end_index]) * np.array([time_gap])
        plt.gca().invert_xaxis()
        plt.plot(remain, remain_time, 'r-')
        plt.draw()
        plt.pause(0.1)
        plt.cla()


if __name__ == '__main__':
    main()
