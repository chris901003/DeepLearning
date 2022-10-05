import argparse
import os
import json


def args_parse():
    parser = argparse.ArgumentParser('All starting from here')
    # 設定每個模塊需要接收的資料以及輸出的結果
    parser.add_argument('--working-flow-cfg', type=str,
                        default='./config/working_flow_cfg.json')
    # 提供檢測模型，輸入為整個攝影機抓到的圖，會需要輸出食物的圖像以及是哪個id，也就是需要由該模塊說明該圖像是屬於哪個人的
    # 這裡需要根據不同類別輸出到指定剩餘量檢測模型
    parser.add_argument('--object-detection-cfg', type=str,
                        default='./config/object_detection_cfg.json')
    # 從目標檢測網路預測出的類別要對應放到哪個剩餘量檢測網路
    parser.add_argument('--object-classify-to-remain-classify', type=str,
                        default='./config/object_classify_to_remain_classify_cfg.json')
    # 輸入為透過目標檢測擷取出的圖像以及該圖像是哪個id，輸出會是該圖像式剩餘多少類別
    parser.add_argument('--remain-detection-cfg', type=str,
                        default='./config/remain_detection_cfg.json')
    # 接下來還會有一堆下游工作，目前先處理這兩個的連動
    args = parser.parse_args()
    return args


def parser_cfg(json_file_path):
    assert os.path.exists(json_file_path), f'指定的config文件 {json_file_path} 不存在'
    with open(json_file_path) as f:
        config = json.load(f)
    return config


def main():
    args = args_parse()
    working_flow_cfg = parser_cfg(args.working_flow_cfg)


if __name__ == '__main__':
    main()
    print('Finish')
