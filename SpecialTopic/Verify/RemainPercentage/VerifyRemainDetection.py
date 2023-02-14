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
    # 需要分成多少段進行評估
    parser.add_argument('--num-part', '-n', type=int, default=4)
    # 當初紀錄畫面時一秒的畫數
    parser.add_argument('--fps', type=int, default=30)
    # 保存預測剩餘量以及真實重量的檔案位置
    parser.add_argument('--save-info-path', '-f', type=str, default='./ResultSave/record.npy')
    # 最終驗證結果數據存放的根目錄位置
    parser.add_argument('--result-save-root-folder', '-s', type=str, default='./VerifyResult')
    # 根目錄下的資料夾名稱
    parser.add_argument('--save-folder-name', type=str, default='Test')
    args = parser.parse_args()
    return args


def parse_npy(save_info_path):
    return np.load(save_info_path, allow_pickle=True)


def compress_data(record_info, fps, mode):
    """ 資料壓縮方式，將很多frame資料進行壓縮
    """
    if mode == 0:
        return compress_by_mean(record_info, fps)


def compress_by_mean(record_info, fps, sec=1):
    """ 將指定秒數內的資料取平均，預設為1秒
    """
    compress_result = list()
    count = 0
    tmp_remain_list = list()
    tmp_weight_list = list()
    for info in record_info:
        remain = info.get('remain', None)
        weight = info.get('weight', None)
        assert remain is not None and weight is not None, '保存資料內容錯誤，需要提供remain以及weight'
        tmp_remain_list.append(remain)
        tmp_weight_list.append(weight)
        count += 1
        if count == fps * sec:
            avg_remain = sum(tmp_remain_list) / len(tmp_remain_list)
            avg_weight = sum(tmp_weight_list) / len(tmp_weight_list)
            compress_result.append(dict(remain=avg_remain, weight=avg_weight))
            tmp_remain_list = list()
            tmp_weight_list = list()
            count = 0
    if count != 0:
        avg_remain = sum(tmp_remain_list) / len(tmp_remain_list)
        avg_weight = sum(tmp_weight_list) / len(tmp_weight_list)
        compress_result.append(dict(remain=avg_remain, weight=avg_weight))
    return compress_result


def change_weight_to_remain(record_info):
    """ 將重量資訊轉成剩餘量資訊
    """
    results = list()
    min_weight = 10000000
    max_weight = 0
    for info in record_info:
        weight = info.get('weight', None)
        assert weight is not None
        min_weight = min(min_weight, weight)
        max_weight = max(max_weight, weight)
    # 如果要使用此方法得出最大重量需要在影片最前面時就是初始重量
    max_weight = record_info[0]['weight']
    weight_diff = max_weight - min_weight
    for info in record_info:
        remain = info.get('remain', None)
        weight = info.get('weight', None)
        real_remain = (weight - min_weight) / weight_diff * 100
        results.append(dict(predict_remain=remain, real_remain=real_remain))
    return results


def cal_l1_diff(record_info, part_time):
    """ 計算l1差距
    """
    results = list()
    results_sec = list()
    count = 0
    tmp_l1_dif = list()
    for info in record_info:
        predict_remain = info.get('predict_remain', None)
        real_remain = info.get('real_remain', None)
        assert predict_remain is not None and real_remain is not None, '變換過程有錯誤'
        diff = abs(predict_remain - real_remain)
        tmp_l1_dif.append(diff)
        results_sec.append(diff)
        count += 1
        if count == part_time:
            avg_diff = sum(tmp_l1_dif) / len(tmp_l1_dif)
            results.append(avg_diff)
            tmp_l1_dif = list()
            count = 0
    if count != 0:
        avg_diff = sum(tmp_l1_dif) / len(tmp_l1_dif)
        results.append(avg_diff)
    total_avg = sum(results_sec) / len(results_sec)
    return results, results_sec, total_avg


def cal_l2_diff(record_info, part_time):
    """ 計算l2差距
    """
    results = list()
    results_sec = list()
    count = 0
    tmp_l2_dif = list()
    for info in record_info:
        predict_remain = info.get('predict_remain', None)
        real_remain = info.get('real_remain', None)
        assert predict_remain is not None and real_remain is not None, '變換過程有錯誤'
        diff = (predict_remain - real_remain) * (predict_remain - real_remain)
        tmp_l2_dif.append(diff)
        results_sec.append(diff)
        count += 1
        if count == part_time:
            avg_diff = math.sqrt(sum(tmp_l2_dif)) / len(tmp_l2_dif)
            results.append(avg_diff)
            tmp_l2_dif = list()
            count = 0
    if count != 0:
        avg_diff = math.sqrt(sum(tmp_l2_dif)) / len(tmp_l2_dif)
        results.append(avg_diff)
    total_avg = math.sqrt(sum(results_sec)) / len(results_sec)
    return results, results_sec, total_avg


def parse_time_line(time_line):
    """ 整理分段時間的顯示內容
    """

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


def main(args=None):
    if args is None:
        args = parse_args()
        num_part = args.num_part
        fps = args.fps
        save_info_path = args.save_info_path
        result_save_root_folder = args.result_save_root_folder
        save_folder_name = args.save_folder_name
    else:
        num_part = args.get('numPart', None)
        fps = args.get('fps', None)
        save_info_path = args.get('saveInfoPath', None)
        result_save_root_folder = args.get('resultSaveRootFolder', None)
        save_folder_name = args.get('saveFolderName', None)
        assert num_part is not None and fps is not None and result_save_root_folder is not None and \
               save_folder_name is not None, '缺少資料'
    save_folder_name = os.path.join(result_save_root_folder, save_folder_name)
    assert num_part > 0, '須至少大於一段'
    assert fps > 0, 'FPS值至少大於0'
    if not os.path.exists(save_info_path):
        raise RuntimeError('無法取得指定路徑的檔案')
    if not os.path.exists(result_save_root_folder):
        os.mkdir(result_save_root_folder)
    if not os.path.exists(save_folder_name):
        os.mkdir(save_folder_name)
    record_info = parse_npy(save_info_path)
    record_info = record_info.tolist()
    assert isinstance(record_info, list), '保存資料格式錯誤，外層需要是list格式'
    for info in record_info:
        assert isinstance(info, dict), '保存資料格式錯誤，內部資料需要是dict格式'
    record_info = compress_data(record_info, fps, 0)
    record_info = change_weight_to_remain(record_info)
    total_time = len(record_info)
    part_time = math.ceil(total_time / num_part)
    time_line = [part_time * i for i in range(num_part + 1)]
    time_line[-1] = min(time_line[-1], total_time)
    time_line = parse_time_line(time_line)
    l1_diff_part, l1_diff_sec, l1_tot_avg = cal_l1_diff(record_info, part_time)
    l2_diff_part, l2_diff_sec, l2_tot_avg = cal_l2_diff(record_info, part_time)

    # 將剩餘量資料提取出來
    predict_remain_list = list()
    real_remain_list = list()
    for info in record_info:
        predict_remain = info.get('predict_remain', None)
        real_remain = info.get('real_remain', None)
        assert predict_remain is not None and real_remain is not None
        predict_remain_list.append(predict_remain)
        real_remain_list.append(real_remain)

    # 將結果保存到Json檔案中
    json_save_path = os.path.join(save_folder_name, 'raw_info.json')
    save_info = dict(
        predict_remain=predict_remain_list,
        real_remain=real_remain_list,
        l1_sec=l1_diff_sec,
        l2_sec=l2_diff_sec,
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
    plt.title('剩餘量', fontproperties=font1)
    plt.plot(predict_remain_list, 'b--', label='預估')
    plt.plot(real_remain_list, 'r-', label='真實')
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
