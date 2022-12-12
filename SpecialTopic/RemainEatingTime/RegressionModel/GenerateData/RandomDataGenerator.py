import argparse
import numpy as np
import pickle
import json


def args_parser():
    parser = argparse.ArgumentParser('隨機生成訓練資料')
    # 每筆資料生成的長度，給定一個範圍，會從指定範圍中獲取一種長度
    # 這裡的長度不會包含最前面的開始以及最後面的結束部分
    parser.add_argument('--length-range', type=int, default=[100, 120], nargs='+')
    # 總共需要生成多少資料
    parser.add_argument('--number-of-data', type=int, default=200)
    # 進行去除後面的概率
    parser.add_argument('--cutoff-rate', type=float, default=0.5)
    # 截斷位置，因為需要部分沒有直接到結束的資料，所以這裡可以設定隨機拋棄最後面的幾個剩餘量
    parser.add_argument('--offset-range', type=int, default=[0, 80], nargs='+')
    # 保存生成資料位置
    parser.add_argument('--dataset-save-path', type=str, default='regression_dataset.pickle')
    # 將設定資料進行保存，這裡會需要保存訓練時的設定，因為會透過指定的length決起始的值以及終點值以及padding值
    parser.add_argument('--save-setting-path', type=str, default='setting.json')
    args = parser.parse_args()
    return args


def main():
    args = args_parser()
    length_range = args.length_range
    number_of_data = args.number_of_data
    cutoff_rate = args.cutoff_rate
    offset_range = args.offset_range
    dataset_save_path = args.dataset_save_path
    save_setting_path = args.save_setting_path
    assert length_range[1] >= length_range[0] >= 2, '指定的生成長度至少需要2並且由小到大'

    max_length = length_range[1] + 2

    # remain parameter
    remain_start_value = 101
    remain_end_value = 102
    remain_padding_value = 103

    # remain time parameter
    remain_time_start_value = length_range[1] + 1
    remain_time_end_value = length_range[1] + 2
    remain_time_padding_value = length_range[1] + 3

    dataset_list = list()
    for _ in range(number_of_data):
        length = np.random.randint(low=length_range[0], high=length_range[1] - 1)
        remain = np.random.randint(low=1, high=100, size=length)
        remain = np.append(np.array([100]), remain)
        remain = np.append(remain, np.array([0]))
        remain = np.sort(remain)[::-1]
        remain_time = [length + 2 - idx - 1 for idx in range(length + 2)]
        if np.random.random() < cutoff_rate:
            offset = np.random.randint(low=max(0, offset_range[0]), high=min(length + 1, offset_range[1] + 1))
            remain = remain[:length + 2 - offset]
            remain_time = remain_time[:length + 2 - offset]
            # 這裡的length是真實有數值的長度
            length -= offset - 2
        else:
            length += 2
        remain = np.append(np.array([remain_start_value]), remain)
        remain = np.append(remain, np.array([remain_end_value]))
        remain_time = np.append(np.array([remain_time_start_value]), remain_time)
        remain_time = np.append(remain_time, np.array([remain_time_end_value]))
        remain = np.append(remain, np.array([remain_padding_value] * length_range[1]))[:max_length]
        remain_time = np.append(remain_time, np.array([remain_time_padding_value] * length_range[1]))[:max_length]
        remain = remain.tolist()
        remain_time = remain_time.tolist()
        data = dict(remain=remain, remain_time=remain_time)
        dataset_list.append(data)

    # 保存生成資料
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
    with open(dataset_save_path, 'wb') as f:
        pickle.dump(dataset_list, f)
    with open(save_setting_path, 'w') as f:
        json.dump(parameters, f, indent=4)
    print('Finish Generate Data')


def check_data(data_path='regression_dataset.pickle', setting_path='setting.json'):
    with open(data_path, 'rb') as f:
        datasets = pickle.load(f)
    assert isinstance(datasets, list), 'dataset中的資料需要是list型態'
    with open(setting_path, 'r') as f:
        settings = json.load(f)
    max_length = settings.get('max_length', None)
    assert max_length is not None, '在setting當中缺少max_length'
    remain_start_value = settings.get('remain_start_value', None)
    assert remain_start_value is not None, '在setting當中缺少remain_start_value'
    remain_end_value = settings.get('remain_end_value', None)
    assert remain_end_value is not None, '在setting當中缺少remain_end_value'
    remain_padding_value = settings.get('remain_padding_value', None)
    assert remain_padding_value is not None, '在setting當中缺少remain_padding_value'
    remain_time_start_value = settings.get('remain_time_start_value', None)
    assert remain_time_start_value is not None, '在setting當中缺少remain_time_start_value'
    remain_time_end_value = settings.get('remain_time_end_value', None)
    assert remain_time_end_value is not None, '在setting當中缺少remain_time_end_value'
    remain_time_padding_value = settings.get('remain_time_padding_value', None)
    assert remain_time_padding_value is not None, '在setting當中缺少remain_time_padding_value'
    for data_info in datasets:
        assert isinstance(data_info, dict), 'datasets當中的每個資料需要是dict格式'
        remain = data_info.get('remain', None)
        remain_time = data_info.get('remain_time', None)
        assert remain is not None, '缺少remain資料'
        assert remain_time is not None, '缺少remain_time資料'
        # 先檢查remain的部分
        assert remain[0] == remain_start_value, 'remain起始值與setting當中資料不符'
        assert len(remain) == max_length, 'remain長度有錯誤'
        find_remain_end_value = False
        find_remain_padding_value = False
        for re in remain[1:]:
            if find_remain_padding_value:
                assert re == remain_padding_value, 'remain有值在padding之後不是padding'
            if re == remain_end_value:
                assert not find_remain_padding_value, 'remain padding發生在remain end前'
                find_remain_end_value = True
            elif re == remain_padding_value:
                assert find_remain_end_value, 'remain padding發生在remain end前'
                find_remain_padding_value = True
            else:
                assert re < remain_start_value, 'remain的值需小於remain_start_value'
        # remain time檢查
        assert remain_time[0] == remain_time_start_value, 'remain time起始值與setting當中資料不符'
        assert len(remain_time) == max_length, 'remain time長度有問題'
        find_remain_time_end_value = False
        find_remain_time_padding_value = False
        for re in remain_time[1:]:
            if find_remain_time_padding_value:
                assert re == remain_time_padding_value, 'remain time有值在padding之後不是padding'
            if re == remain_time_end_value:
                assert not find_remain_time_padding_value, 'remain time padding發生在 nd之前'
                find_remain_time_end_value = True
            elif re == remain_time_padding_value:
                assert find_remain_time_end_value, 'remain time padding需要在end之後'
                find_remain_time_padding_value = True
            else:
                assert re < remain_time_start_value, 'remain time中的值需要小於start值'
    print('Dataset All pass')


if __name__ == '__main__':
    print('Generate Regression model random training data')
    main()
