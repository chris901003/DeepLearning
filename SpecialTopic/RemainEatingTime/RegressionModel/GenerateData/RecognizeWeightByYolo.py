import argparse
import os
import torch
import cv2
import pandas as pd
from SpecialTopic.YoloxObjectDetection.api import init_model, detect_image


def args_parser():
    parser = argparse.ArgumentParser()
    # 訓練權重檔案位置
    parser.add_argument('--pretrained-path', type=str, default=r'C:\Checkpoint\YoloxWeightNumberDetection\first_'
                                                               r'version_1_8.pth')
    # 偵測影片路徑
    parser.add_argument('--video-path', type=str, default=r'C:\Dataset\vedio\weight.mp4')
    # 保存資料路徑
    parser.add_argument('--save-path', type=str, default='remain.xlsx')
    args = parser.parse_args()
    return args


def reduce_weight(weight_info, scope=(0.3, 0.7)):
    """ 將一秒內的幀數進行均值，這裡可以選擇經過排序後要取哪個區間內的值進行平均計算
    """
    weight_len = len(weight_info)
    weight_info = sorted(weight_info)
    left_idx, right_idx = int(weight_len * scope[0]), int(weight_len * scope[1])
    left_idx = max(0, max(left_idx, right_idx - 1))
    right_idx = min(weight_len, max(right_idx, left_idx + 1))
    weight_info = weight_info[left_idx:right_idx]
    avg_weight = sum(weight_info) / len(weight_info)
    return avg_weight


def write_to_excel(weights, remains, save_path):
    """ 將重量資料以及剩餘量資料存放到指定的excel路徑當中
    """
    weight_dict = dict()
    remain_dict = dict()
    for idx, weight in enumerate(weights):
        weight_dict[str(idx)] = weight
    for idx, remain in enumerate(remains):
        remain_dict[str(idx)] = remain
    data = pd.DataFrame({'weights': weight_dict, 'remains': remain_dict})
    data.to_excel(save_path)


def detect_weight(model, video_path, save_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    model = model.to(device)
    model.eval()
    weight_record = list()
    while True:
        ret, image = cap.read()
        if ret:
            image_height, image_width = image.shape[:2]
            results = detect_image(model, device, image, input_shape=[640, 640], num_classes=10, confidence=0.8)
            labels, scores, boxes = results
            detect_info = list()
            for label, box in zip(labels, boxes):
                data = box.copy()
                data.append(label)
                detect_info.append(data)
            # 用xmin作為排序，從左到右
            detect_info = sorted(detect_info, key=lambda s: s[1])
            current_weight = 0
            for detect in detect_info:
                ymin, xmin, ymax, xmax = detect[:4]
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                if ymin <= 0 or xmin <= 0 or ymax >= image_height or xmax >= image_width:
                    continue
                number = detect[-1]
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                current_weight = current_weight * 10 + number
            weight_record.append(current_weight)
            cv2.putText(image, f"Detect weight : {current_weight}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 3)
            cv2.imshow('img', image)
        else:
            break
        if cv2.waitKey(1) == ord('q'):
            break
    avg_weight_per_sec = list()
    for idx in range(0, len(weight_record), video_fps):
        weight_info = weight_record[idx: idx + video_fps]
        weight = reduce_weight(weight_info)
        avg_weight_per_sec.append(weight)
    max_weight, min_weight = avg_weight_per_sec[0], avg_weight_per_sec[-1]
    total_weight = max_weight - min_weight
    if total_weight == 0:
        print('重量變動為0，請確認是否有問題')
        print('會先將食物重量設定成很小的值使得程式可以繼續執行')
        total_weight = 1e-9
    remain = list()
    for weight in avg_weight_per_sec:
        current = weight - min_weight
        current_remain = round(current / total_weight * 100, 2)
        remain.append(current_remain)
    write_to_excel(avg_weight_per_sec, remain, save_path)
    print('Finish detect weight')


def main():
    args = args_parser()
    pretrained_path = args.pretrained_path
    video_path = args.video_path
    save_path = args.save_path
    assert pretrained_path is not None and os.path.exists(pretrained_path), '需提供訓練權重路徑'
    assert video_path is not None and os.path.exists(video_path), '影片路徑不存在'
    model = init_model(pretrained=pretrained_path, num_classes=10)
    detect_weight(model, video_path, save_path)


if __name__ == '__main__':
    main()
