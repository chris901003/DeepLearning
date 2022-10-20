import argparse
import json
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    # 總共要生成多少資料
    parser.add_argument('--num-datas', type=int, default=50000)
    # 在x軸上的數值範圍，也就是剩餘時間的範圍
    # 在生成時會生成max_len的長度，起始點會從x-start-range中挑一個值出來作為開頭
    parser.add_argument('--axis-x-start-range', type=float, default=[0.0, 0.0], nargs='+')
    # 終點會從x-end-range中挑一個值出來
    parser.add_argument('--axis-x-end-range', type=float, default=[0.6, 1.0], nargs='+')
    # 在y軸上的數值範圍，也就是剩餘量的範圍
    parser.add_argument('--axis-y-range', type=int, default=[0, 100], nargs='+')
    # 設定參數都會放到py檔當中，這樣比較好調整數據
    # 這裡可以選擇放入json檔案或是py檔案都可以進行解碼
    # 如果要給的是py檔就需要直接放在當前目錄下並且直接給檔名就可以，同時最外層的dict請取名為setting
    parser.add_argument('--setting-path', type=str, default='setting.py')
    # 對於每個不同剩餘量出現的概率，這裡會透過隨機進行生成，可以指定哪種剩餘量出現的概率
    # 這裡會去抓setting_json當中的資料，如果沒有抓到就會平均分配概率
    parser.add_argument('--y-probability', type=str, default='y_probability')
    # 生成的長度，這裡的長度指的是x軸上的
    parser.add_argument('--len-range', type=int, default=[40, 120], nargs='+')
    # 有需要將哪些參數加到包裝的資料當中，因為在訓練時可以從這裡獲取需要的資料，這樣在訓練時就不用設定
    parser.add_argument('--add-to-datas', type=str, default=['remain_to_index', 'time_to_index', 'remain_pad_val',
                                                             'time_pad_val', 'remain_SOS_val', 'time_SOS_val',
                                                             'remain_EOS_val', 'time_EOS_val'], nargs='+')
    # 是否需要將生成的資料畫成折線圖，如果不需要的話就設定成none
    parser.add_argument('--plot-data-picture-path', type=str, default='none')
    # 資料保存位置
    parser.add_argument('--data-save-path', type=str, default='./train_annotation.pickle')
    args = parser.parse_args()
    return args


def read_file(file_path):
    if os.path.splitext(file_path)[1] == '.json':
        with open(file_path, 'r') as f:
            result = json.load(f)
    elif os.path.splitext(file_path)[1] == '.py':
        module = __import__(os.path.splitext(file_path)[0])
        result = module.setting
    else:
        raise ValueError('目前只支援json或是py檔案讀取')
    return result


def generate_data(num_datas, axis_x_start_range, axis_x_end_range, axis_y_range, y_probability, len_range, setting_dict):
    num_y_classes = axis_y_range[1] - axis_y_range[0] + 1
    y_probability = setting_dict.get(y_probability, None)
    if y_probability is None:
        y_probability = np.ones(num_y_classes) / num_y_classes
    if isinstance(y_probability, list):
        y_probability = np.array(y_probability)
    assert len(y_probability) == num_y_classes, f'給定概率與類別數不相同，y probability: {len(y_probability)}，' \
                                                f'classes: {len(num_y_classes)}'
    assert axis_x_start_range[0] <= axis_x_start_range[1] <= axis_x_end_range[0] <= axis_x_end_range[1]
    assert len_range[0] <= len_range[1], '長度有問題'
    food_remain_list = [index for index in range(axis_y_range[0], axis_y_range[1] + 1)]
    results = list()
    pbar = tqdm(total=num_datas, desc='Generate data', miniters=0.3)
    for _ in range(num_datas):
        random_len = np.random.randint(low=len_range[0], high=len_range[1] + 1 - 2)
        food_remain = np.random.choice(food_remain_list, size=random_len - 2, replace=True, p=y_probability)
        food_remain = food_remain.tolist()
        food_remain = sorted(food_remain, reverse=True)
        if food_remain[0] != axis_y_range[1]:
            food_remain = [axis_y_range[1]] + food_remain
        if food_remain[-1] != axis_y_range[0]:
            food_remain = food_remain + [axis_y_range[0]]
        food_remain_len = len(food_remain)
        time_remain = [food_remain_len - idx - 1 for idx in range(food_remain_len)]
        start_ratio = np.random.uniform(low=axis_x_start_range[0], high=axis_x_start_range[1])
        end_ratio = np.random.uniform(low=axis_x_end_range[0], high=axis_x_end_range[1])
        start_index = int(food_remain_len * start_ratio)
        end_index = int(food_remain_len * end_ratio)
        food_remain_clip = food_remain[start_index: end_index + 1]
        time_remain_clip = time_remain[start_index: end_index + 1]
        # 這裡將完整時段以及經過提取中間部分的都進行回傳
        if np.random.random() > 0.7:
            data = dict(food_remain=food_remain, time_remain=time_remain)
            results.append(data)
        data = dict(food_remain=food_remain_clip, time_remain=time_remain_clip)
        results.append(data)
        pbar.update(1)
    pbar.close()
    return results


def plot_data(outputs, save_path, x_label='Time remain', y_label='Food remain'):
    pbar = tqdm(total=len(outputs), desc='Drawing data', miniters=0.3)
    for index, output in enumerate(outputs):
        plt.gca().invert_xaxis()
        plt.plot(output['time_remain'], output['food_remain'])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        save_path_name = os.path.join(save_path, f'{index}.jpg')
        plt.savefig(save_path_name)
        plt.close('all')
        pbar.update(1)
    pbar.close()


def main():
    args = parse_args()
    setting_path = args.setting_path
    setting_dict = read_file(setting_path)
    axis_x_start_range = args.axis_x_start_range
    axis_x_end_range = args.axis_x_end_range
    axis_y_range = args.axis_y_range
    y_probability = args.y_probability
    len_range = args.len_range
    max_len = len_range[1]
    outputs = generate_data(args.num_datas, axis_x_start_range, axis_x_end_range, axis_y_range, y_probability,
                            len_range, setting_dict)
    if args.plot_data_picture_path != 'none':
        if not os.path.exists(args.plot_data_picture_path):
            os.mkdir(args.plot_data_picture_path)
        plot_data(outputs, args.plot_data_picture_path)
    results = dict(datas=outputs, max_len=max_len)
    add_to_datas = args.add_to_datas
    for add_to_data in add_to_datas:
        info = setting_dict.get(add_to_data, None)
        assert info is not None, f'在setting_json當中沒有{add_to_data}'
        results[add_to_data] = info
    # 這裡的加一都是因為以0-index的關係，後面的三會是[<SOS>, <EOS>, <PAD>]
    results['num_remain_classes'] = axis_y_range[1] - axis_y_range[0] + 1 + 3
    results['num_time_classes'] = max_len + 1 + 3
    with open(args.data_save_path, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
    print('Finish')
