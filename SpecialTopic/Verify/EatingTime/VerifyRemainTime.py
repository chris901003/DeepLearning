import argparse
import numpy as np
import os
import math
import json
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties as font

font1 = font(fname='./NotoSansTC-Bold.otf')


def parse_args():
    parser = argparse.ArgumentParser()
    # 需要分成多少段來進行評估
    parser.add_argument('--num-part', '-n', type=int, default=4)
    # 保存預測剩餘量以及真實重量的檔案位置
    parser.add_argument('--save-info-path', '-f', type=str, default='./ResultSave/remain_time.npy')

    # 由於檢測出出來會有多個檔案，所以需要再多指定一個資料夾位置，也就是會保存到[result_save_root_folder/save_folder_name]下
    # 所以save_folder_name會是一個資料夾
    # 最終驗證結果數據存放的根目錄位置
    parser.add_argument('--result-save-root-folder', '-s', type=str, default='./VerifyResult')
    # 根目錄下的資料夾名稱
    parser.add_argument('--save-folder-name', type=str, default='Test')
    args = parser.parse_args()
    return args


def parse_npy(save_info_path):
    return np.load(save_info_path, allow_pickle=True)


def compress_frame(remain_time_info):
    """ 將一秒內的資料壓縮
    """
    result = list()
    last_stopwatch_time = remain_time_info[0]['stopwatch_time']
    tmp_predict_remain_time = list()
    for idx, info in enumerate(remain_time_info):
        stopwatch_time = info.get('stopwatch_time', None)
        predict_remain_time = info.get('predict_remain_time', None)
        assert stopwatch_time is not None and predict_remain_time is not None, \
            '須提供stopwatch_time以及predict_remain_time資料'
        if stopwatch_time == last_stopwatch_time and idx != len(remain_time_info) - 1:
            tmp_predict_remain_time.append(predict_remain_time)
        else:
            if idx == len(remain_time_info) - 1:
                tmp_predict_remain_time.append(predict_remain_time)
            tot_predict = sum(tmp_predict_remain_time)
            avg_predict = tot_predict / len(tmp_predict_remain_time)
            result.append(dict(predict_remain_time=avg_predict, stopwatch_time=last_stopwatch_time))
            tmp_predict_remain_time = list()
            tmp_predict_remain_time.append(predict_remain_time)
            last_stopwatch_time = stopwatch_time
    return result


def recover_init_frame(remain_time_info):
    """
    在剛開始時，predict_remain_time會處於初始化階段，所以都會是-1
    我們將最先預測出來的剩餘時間將這些-1給替代掉
    """
    init_predict_remain_time = -1
    for info in remain_time_info:
        predict_remain_time = info.get('predict_remain_time', None)
        assert predict_remain_time is not None, '須提供predict_remain_time否則無法校正'
        if predict_remain_time != -1:
            init_predict_remain_time = predict_remain_time
            break
    for info in remain_time_info:
        if info['predict_remain_time'] != -1:
            break
        info['predict_remain_time'] = init_predict_remain_time
    return remain_time_info


def change_stopwatch_to_remain_time(remain_time_info):
    """ 將碼表的時間轉成真實剩餘秒數
    """
    result = list()
    end_stopwatch_time = remain_time_info[-1].get('stopwatch_time', None)
    assert end_stopwatch_time is not None
    for info in remain_time_info:
        stopwatch_time = info.get('stopwatch_time', None)
        assert stopwatch_time is not None, '須提供stopwatch_time'
        real_remain_time = end_stopwatch_time - stopwatch_time
        predict_remain_time = info.get('predict_remain_time', None)
        assert predict_remain_time is not None, '須提供預測的剩餘時間'
        result.append(dict(predict_remain_time=predict_remain_time, real_remain_time=real_remain_time,
                           stopwatch_time=stopwatch_time))
    return result


def parse_time_line(time_line):

    def sec_to_min_plus_sec(sec):
        minutes = sec // 60
        sec = sec % 60
        return f"{minutes}'{sec}''"

    results = list()
    for idx in range(len(time_line) - 1):
        left_time = time_line[idx]
        right_time = time_line[idx + 1]
        left_time = sec_to_min_plus_sec(left_time)
        right_time = sec_to_min_plus_sec(right_time)
        results.append(f'{left_time}~{right_time}')
    return results


def cal_l1_diff(record_info, part_time):
    """ 計算出l1損失 
    Args:
        record_info: 剩餘時間紀錄資料
            ['predict_remain_time', 'real_remain_time', 'stopwatch_time']
        part_time: 多少秒會為一個部分
    Returns:
        results_part: 根據指定的段數進行切割，可以大致看出來該段的結果
        results_sec: 每一秒的損失值，可以更細緻的了解每一秒的誤差
    """
    results_part = list()
    results_sec = list()
    tmp_l1_dif = list()
    count = 0
    for idx, info in enumerate(record_info):
        predict_remain_time = info.get('predict_remain_time', None)
        real_remain_time = info.get('real_remain_time', None)
        stopwatch_time = info.get('stopwatch_time', None)
        assert predict_remain_time is not None and real_remain_time is not None and stopwatch_time is not None, \
            '資料不完整無法進行計算，l1計算出錯'
        diff = abs(predict_remain_time - real_remain_time)
        tmp_l1_dif.append(diff)
        results_sec.append(diff)
        count += 1
        if (count == part_time) or (idx == len(record_info) - 1):
            avg_diff = sum(tmp_l1_dif) / len(tmp_l1_dif)
            results_part.append(avg_diff)
            tmp_l1_dif = list()
            count = 0
    total_avg = sum(results_sec) / len(results_sec)
    return results_part, results_sec, total_avg


def cal_l2_diff(record_info, part_time):
    """ 計算出l2損失值
    """
    results_part = list()
    results_sec = list()
    tmp_l2_dif = list()
    count = 0
    for idx, info in enumerate(record_info):
        predict_remain_time = info.get('predict_remain_time', None)
        real_remain_time = info.get('real_remain_time', None)
        stopwatch_time = info.get('stopwatch_time', None)
        assert predict_remain_time is not None and real_remain_time is not None and stopwatch_time is not None, \
            '資料不完整無法進行計算，l2計算出錯'
        diff = (predict_remain_time - real_remain_time) * (predict_remain_time - real_remain_time)
        tmp_l2_dif.append(diff)
        results_sec.append(diff)
        count += 1
        if (count == part_time) or (idx == len(record_info) - 1):
            avg_diff = math.sqrt(sum(tmp_l2_dif)) / len(tmp_l2_dif)
            results_part.append(avg_diff)
            tmp_l2_dif = list()
            count = 0
    total_avg = math.sqrt(sum(results_sec)) / len(results_sec)
    return results_part, results_sec, total_avg


def main():
    args = parse_args()
    num_part = args.num_part
    save_info_path = args.save_info_path
    result_save_root_folder = args.result_save_root_folder
    save_folder_name = args.save_folder_name
    save_folder_name = os.path.join(result_save_root_folder, save_folder_name)
    assert num_part > 0, '至少要分成一段'
    assert os.path.exists(save_info_path), '提供的剩餘時間保存資料不存在'
    if not os.path.exists(result_save_root_folder):
        os.mkdir(result_save_root_folder)
    if not os.path.exists(save_folder_name):
        os.mkdir(save_folder_name)
    remain_time_info = parse_npy(save_info_path)
    remain_time_info = remain_time_info.tolist()
    assert isinstance(remain_time_info, list), '保存資料格式錯誤，外層需要是list格式'
    for info in remain_time_info:
        assert isinstance(info, dict), '保存資料格式錯誤，內部資料需要是dict格式'
    # 將一秒的資料統整起來
    remain_time_info = compress_frame(remain_time_info)
    # 最前面一個frame會是在Init的部分
    remain_time_info = remain_time_info[1:]
    remain_time_info = recover_init_frame(remain_time_info)
    # 將計時器的資料轉換成真實剩餘時間
    remain_time_info = change_stopwatch_to_remain_time(remain_time_info)
    """
    remain_time_info = {
        'predict_remain_time': 預測的剩餘時間
        'real_remain_time': 真實的剩餘時間
        'stopwatch_time': 碼表上的秒數
    }
    """

    # 影片總秒數
    total_sec = len(remain_time_info)
    part_time = math.ceil(total_sec / num_part)
    time_line = [part_time * i for i in range(num_part + 1)]
    time_line[-1] = min(time_line[-1], total_sec)
    time_line = parse_time_line(time_line)

    # 計算剩餘時間損失
    l1_diff_part, l1_diff_sec, l1_tot_avg = cal_l1_diff(remain_time_info, part_time)
    l2_diff_part, l2_diff_sec, l2_tot_avg = cal_l2_diff(remain_time_info, part_time)

    # 提取預測剩餘時間與真實剩餘時間
    predict_remain_time_list = list()
    real_remain_time_list = list()
    for info in remain_time_info:
        predict_remain_time = info.get('predict_remain_time', None)
        real_remain_time = info.get('real_remain_time', None)
        assert predict_remain_time is not None and real_remain_time is not None, \
            '資料缺少請檢查，提取預測以及真實剩餘時間部分錯誤'
        predict_remain_time_list.append(predict_remain_time)
        real_remain_time_list.append(real_remain_time)

    # 將詳細資料保存到Json檔案中，後續要查找原始資料比較方便
    json_save_path = os.path.join(save_folder_name, 'raw_info.json')
    save_info = dict(
        predict_remain_time=predict_remain_time_list,
        real_remain_time=real_remain_time_list,
        l1_sec=l1_diff_sec,
        l2_sec=l2_diff_sec,
        num_part=num_part,
        l1_part=l1_diff_part,
        l2_part=l2_diff_part,
        l1_tot_avg=l1_tot_avg,
        l2_tot_avg=l2_tot_avg
    )
    with open(json_save_path, 'w') as f:
        json.dump(save_info, f, indent=4)

    # 畫圖表
    _ = plt.figure(figsize=(11, 7))
    plt.subplot(611)
    plt.title('剩餘時間', fontproperties=font1)
    plt.plot(predict_remain_time_list, 'b--', label='預估')
    plt.plot(real_remain_time_list, 'r-', label='真實')
    plt.legend(loc='best', prop=font1)
    plt.subplot(612)
    plt.title('L1', fontproperties=font1)
    plt.plot(l1_diff_sec, 'b-')
    plt.subplot(613)
    plt.title('L2', fontproperties=font1)
    plt.plot(l2_diff_sec, 'b-')
    plt.subplot(614)
    plt.title('L1 分段', fontproperties=font1)
    plt.bar(x=range(len(l1_diff_part)), height=l1_diff_part, tick_label=time_line, color='g')
    plt.subplot(615)
    plt.title('L2 分段', fontproperties=font1)
    plt.bar(x=range(len(l2_diff_part)), height=l2_diff_part, tick_label=time_line, color='r')
    plt.subplot(616)
    plt.title('總平均', fontproperties=font1)
    plt.text(0, 0.5, f'L1 Avg: {l1_tot_avg}', fontsize=15, color='blue')
    plt.text(0, 0, f'L2 Avg: {l2_tot_avg}', fontsize=15, color='blue')

    plt.tight_layout()
    chart_save_path = os.path.join(save_folder_name, 'chart.jpg')
    plt.savefig(chart_save_path)
    plt.show()


if __name__ == '__main__':
    main()
