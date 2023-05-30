import argparse
import os
import json
import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    # 跑完驗證的資料保存資料夾
    parser.add_argument('--source-folder-path', type=str, default='./Result')
    # 最後結果輸出位置
    parser.add_argument('--output-folder-path', type=str, default='./RemainAvgAcc')
    # 切成幾等分來看每等分中的誤差
    parser.add_argument('--num-parts', type=int, default=5)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    source_folder_path = args.source_folder_path
    output_folder_path = args.output_folder_path
    num_parts = args.num_parts
    assert num_parts > 0, "至少需要切成一塊"
    assert os.path.exists(source_folder_path), "原始資料不存在"
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    raw_infos = load_raw_file(source_folder_path)
    part_infos = list()
    for raw_info in raw_infos:
        part_info = get_part_info(raw_info, num_parts)
        part_infos.append(part_info)
    avg_part_diff = get_avg_part_diff(part_infos, num_parts)
    draw_picture(avg_part_diff, num_parts)
    print(avg_part_diff)


# 讀取原始資料
def load_raw_file(source_folder_path):
    raw_infos = list()
    for folder_path in os.listdir(source_folder_path):
        file_path = os.path.join(source_folder_path, folder_path, "raw_info.json")
        if os.path.exists(file_path):
            print(f"獲取{folder_path}資料")
        else:
            print(f"無法取得{folder_path}資料")
            continue
        with open(file_path, "r") as f:
            raw_info = json.load(f)
        raw_infos.append(raw_info)
    return raw_infos


# 獲取經過分段後的結果
def get_part_info(raw_info, num_parts):
    """
        raw_info = ["predict_remain", "real_remain"]
    """
    predict_remain = raw_info["predict_remain"]
    real_remain = raw_info["real_remain"]
    data_len = len(predict_remain)
    per_part_len = data_len // num_parts
    l1_part_avg = list()
    for idx in range(1, num_parts + 1):
        start = (idx - 1) * per_part_len
        end = idx * per_part_len if idx != num_parts else -1
        part_predict_remain = predict_remain[start: end]
        part_real_remain = real_remain[start: end]
        l1_diff = [abs(predict - real) for predict, real in zip(part_predict_remain, part_real_remain)]
        l1_avg = sum(l1_diff) / len(l1_diff)
        l1_part_avg.append(l1_avg)
    return l1_part_avg


# 根據每個部分進行平均
def get_avg_part_diff(part_infos, num_parts):
    parts_avg = list()
    for idx in range(num_parts):
        total = 0
        for info in part_infos:
            total += info[idx]
        avg = total / len(part_infos)
        parts_avg.append(avg)
    return parts_avg


# 根據傳入的資料將柱狀圖畫出
def draw_picture(avg_part_diff, num_parts):
    x_info = [idx for idx in range(1, num_parts + 1)]
    x_info = np.array(x_info)
    y_info = np.array(avg_part_diff)
    plt.bar(x_info, y_info)
    plt.show()


if __name__ == "__main__":
    print("Running Remain Avg Acc")
    main()
