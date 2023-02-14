import argparse
import os
import json
import math
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties as font

font1 = font(fname='./NotoSansTC-Bold.otf')


def parse_args():
    parser = argparse.ArgumentParser()
    # 要分成多少段，這裡將不管個別影片的長短，就只是分段來看每一段的準確度
    parser.add_argument('--num-part', '-n', type=int, default=4)
    # 多個食物預測結果存放的資料夾位置，在此資料夾下應該是要有很多資料夾，每個資料夾表示吃一次食物
    parser.add_argument('--save-folder', '-f', type=str, default='./VerifyResult')
    # 從提取原始資料的那些內容來做平均
    parser.add_argument('--select-infos', '-s', type=str, default=['l1_sec', 'l2_sec'], nargs='+')
    # 在畫圖時的標題
    parser.add_argument('--select-infos-title', type=str, default=['l1', 'l2'])
    # 結果保存根目錄
    parser.add_argument('--result-save-root-folder', type=str, default='./MixResult')
    # 根目錄下的哪個資料夾，最終保存的位置會是[result_save_root_folder\result_folder_name]
    parser.add_argument('--result-folder-name', type=str, default='Test')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    num_part = args.num_part
    save_folder = args.save_folder
    select_infos = args.select_infos
    select_infos_title = args.select_infos_title
    result_save_root_folder = args.result_save_root_folder
    result_folder_name = args.result_folder_name
    result_folder_name = os.path.join(result_save_root_folder, result_folder_name)
    assert os.path.exists(save_folder), f'指定的資料路徑{save_folder}不存在'
    if not os.path.exists(result_save_root_folder):
        os.mkdir(result_save_root_folder)
    if not os.path.exists(result_folder_name):
        os.mkdir(result_folder_name)

    # 將原始資料檔案路徑找出
    raw_info_file_path = list()
    for folder_name in os.listdir(save_folder):
        folder_path = os.path.join(save_folder, folder_name)
        if not os.path.isdir(folder_path):
            print(f'路徑{folder_path}並非資料夾型態')
            continue
        file_path = os.path.join(folder_path, 'raw_info.json')
        if os.path.exists(file_path):
            raw_info_file_path.append(file_path)
        else:
            print(f'路徑{file_path}不存在，該資料夾無提供原始資料檔案')

    # 從指定檔案中獲取需要的資訊
    raw_infos = list()
    for raw_path in raw_info_file_path:
        with open(raw_path, 'r') as f:
            raw_info = json.load(f)
        info_dict = dict()
        for select_info in select_infos:
            info = raw_info.get(select_info, None)
            assert info is not None, f'檔案{raw_path}當中沒有提供{select_info}資訊'
            info_dict[select_info] = info
        raw_infos.append(info_dict)

    # 根據指定的分段數量，將資料進行分段(對一個影片分成指定片段，一段的值是透過該段的資料取平均獲取到)
    # raw_infos = list()
    raw_infos = fragment_by_num_part(raw_infos, num_part, select_infos)

    # 根據損失計算的方式分成一類(原先資料是根據一個影片，會將其變成根據損失計算的方式集合成一類)
    # raw_infos = dict()
    raw_infos = separate_by_loss(raw_infos, select_infos)

    # 將每一個損失計算的方式取平均(對於一個類別不同影片取平均)
    # raw_infos = dict()
    raw_infos = avg_raw_infos(raw_infos)

    # 保存原始資料
    json_file_path = os.path.join(result_folder_name, 'raw_info.json')
    with open(json_file_path, 'w') as f:
        json.dump(raw_infos, f, indent=4)

    # 畫圖
    bar_title = [f'P{idx}' for idx in range(1, num_part + 1)]
    _ = plt.figure(figsize=(11, 7))
    for idx, select_info in enumerate(select_infos):
        plt.subplot(len(select_infos), 1, idx + 1)
        plt.title(select_infos_title[idx], fontproperties=font1)
        plt.bar(x=range(len(raw_infos[select_info])), height=raw_infos[select_info], tick_label=bar_title, color='b')
    plt.tight_layout()
    chart_save_path = os.path.join(result_folder_name, 'chart.jpg')
    plt.savefig(chart_save_path)
    plt.show()


def fragment_by_num_part(raw_infos, num_part, selection_infos):
    """ 根據指定的長度去分段，將不同長度的資料最後變成相同數量的段數
    """
    results = list()
    for raw_info in raw_infos:
        info_dict = dict()
        for selection_info in selection_infos:
            info = raw_info.get(selection_info, None)
            assert info is not None, '正常來說這裡不會有問題，如果有就問題大了'
            assert isinstance(info, list), '資料需要是list型態'
            info_len = len(info)
            part_len = math.ceil(info_len / num_part)
            info = compress_info(info, part_len, selection_info)
            info_dict[selection_info] = info
        results.append(info_dict)
    return results


def compress_info(info, part_len, loss_cal):
    """ 根據指定的長度，將一段長度內的資料做平均
    """
    results = list()
    count = 0
    tmp_record = list()
    for idx, num in enumerate(info):
        tmp_record.append(num)
        count += 1
        if count == part_len or idx == len(info) - 1:
            avg_func = support_avg_function(loss_cal)
            avg = avg_func(tmp_record)
            results.append(avg)
            tmp_record = list()
            count = 0
    return results


def separate_by_loss(raw_infos, select_infos):
    """ 根據損失計算的方式分成一類
    """
    results = dict()
    for select_info in select_infos:
        infos = list()
        for raw_info in raw_infos:
            info = raw_info.get(select_info, None)
            assert info is not None, '正常來說這裡不會有任何問題'
            infos.append(info)
        results[select_info] = infos
    return results


def avg_raw_infos(raw_infos):
    """ 將每一個損失計算的方式取平均(對於一個類別不同影片取平均)
    """
    results = dict()
    for k, infos in raw_infos.items():
        avg_info = list()
        for idx in range(len(infos[0])):
            tmp_info = list()
            for info_idx in range(len(infos)):
                tmp_info.append(infos[info_idx][idx])
            avg = sum(tmp_info) / len(tmp_info)
            avg_info.append(avg)
        results[k] = avg_info
    return results


def support_avg_function(loss_cal):
    support_func = {
        'l1_sec': l1_avg,
        'l2_sec': l2_avg
    }
    func = support_func.get(loss_cal, None)
    assert func is not None, f'尚未提供{loss_cal}取平均的方式'
    return func


def l1_avg(info):
    avg = sum(info) / len(info)
    return avg


def l2_avg(info):
    avg = math.sqrt(sum(info)) / len(info)
    return avg


if __name__ == '__main__':
    main()
    print('Finish')
