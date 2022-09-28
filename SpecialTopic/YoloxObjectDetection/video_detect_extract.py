import cv2
from PIL import Image
import shutil
import torch
import os
import argparse
import copy
import numpy as np
from SpecialTopic.ST.build import build_detector
from api import detect_image
from utils import get_classes


def parse_args():
    parser = argparse.ArgumentParser('Extract object from video')
    # 預訓練權重位置
    parser.add_argument('--models-path', type=str, default='/Users/huanghongyan/Downloads/best_weight.pth')
    # 模型大小，注意這裡要與預訓練權重匹配
    parser.add_argument('--phi', type=str, default='l')
    # 類別資訊
    parser.add_argument('--classes-path', type=str, default='/Users/huanghongyan/Downloads/data_annotation/classes.txt')
    # 要預測的圖像路徑
    parser.add_argument('--video-path', type=str, default='/Users/huanghongyan/Downloads/test.mp4')
    # 最終輸入到網路的圖像大小，不是給的圖像大小
    parser.add_argument('--input-shape', type=int, default=[640, 640], nargs='+')
    # 開啟後會對整段影片進行預測
    parser.add_argument('--hole-video', action='store_true')
    # 是否關閉擷取圖像
    parser.add_argument('--extract-picture', action='store_false')
    # 是否需要產生整張圖像的預測匡同時帶有labelImg標註資訊，這裡的採樣間隔數會是frame_interval
    parser.add_argument('--extract-labelImg', action='store_true')
    # 間隔多少幀預測一次
    parser.add_argument('--frame-interval', type=int, default=150)
    # 擷取出的圖像檔案路徑
    parser.add_argument('--save_path', type=str, default='./save')
    # 置信度閾值
    parser.add_argument('--confidence', type=float, default=0.5)
    # nms處理閾值
    parser.add_argument('--nms-iou', type=float, default=0.3)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    hole_video = args.hole_video
    extract_picture = args.extract_picture
    extract_labelImg = args.extract_labelImg
    video_write = None
    if hole_video:
        print('將會對整段影片進行預測，最終影片結果會輸出到指定資料夾當中')
    if extract_picture:
        print('將會根據指定間隔幀數進行預測，會將預測圖像擷取出來放到指定資料夾當中')
    assert hole_video or extract_picture, '至少需要指定一種作為輸出，不然就是做白工'
    assert os.path.exists(args.models_path), '指定的預訓練權重資料不存在'
    assert os.path.exists(args.video_path), '指定影片檔案不存在'
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.mkdir(args.save_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class_names, num_classes = get_classes(args.classes_path)
    for i in range(num_classes):
        folder_path = os.path.join(args.save_path, str(i))
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
    model_cfg = {
        'type': 'YoloBody',
        'phi': args.phi,
        'backbone_cfg': {
            'type': 'YOLOPAFPN'
        },
        'head_cfg': {
            'type': 'YOLOXHead',
            'num_classes': num_classes
        }
    }
    model = build_detector(model_cfg)
    print(f'Load weights {args.models_path}')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.models_path, map_location=device)
    if 'model_weight' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_weight']
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    model = model.to(device)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    assert len(no_load_key) == 0, '給定的預訓練權重與模型不匹配'
    cap = cv2.VideoCapture(args.video_path)
    idx = 0
    total_picture = 0
    while True:
        ret, img = cap.read()
        image = copy.deepcopy(img)
        if img is None:
            break
        height, width = img.shape[:2]
        results = None
        if hole_video:
            if video_write is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_name = os.path.join(args.save_path, 'test.mp4')
                video_write = cv2.VideoWriter(video_name, fourcc, 30, (width, height))
            results = detect_image(model, device, img, args.input_shape, num_classes,
                                   confidence=args.confidence, nms_iou=args.nms_iou)
            labels, scores, bboxes = results
            image = copy.deepcopy(img)
            for label, bbox, score in zip(labels, bboxes, scores):
                ymin, xmin, ymax, xmax = bbox
                ymin = int(max(0, ymin))
                xmin = int(max(0, xmin))
                ymax = int(min(height, ymax))
                xmax = int(min(width, xmax))
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                info = class_names[label] + '|' + str(round(score * 100, 2))
                cv2.putText(image, info, (xmin + 30, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (89, 214, 210), 2, cv2.LINE_AA)
            video_write.write(image)

        if (extract_picture or extract_labelImg) and (idx % args.frame_interval == 0):
            if results is None:
                results = detect_image(model, device, img, args.input_shape, num_classes,
                                       confidence=args.confidence, nms_iou=args.nms_iou)
            labels, scores, bboxes = results
            for label, bbox, score in zip(labels, bboxes, scores):
                ymin, xmin, ymax, xmax = bbox
                ymin = int(max(0, ymin))
                xmin = int(max(0, xmin))
                ymax = int(min(height, ymax))
                xmax = int(min(width, xmax))
                if extract_picture:
                    target = img[ymin:ymax + 1, xmin:xmax + 1, :]
                    target = target.astype(np.uint8)
                    image_name = str(total_picture) + '.jpg'
                    target_folder = os.path.join(args.save_path, str(label), image_name)
                    target = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
                    target.save(target_folder)
                    total_picture += 1
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 10)
                result_str = str(class_names[int(label)]) + '|' + str(round(score * 100, 2))
                cv2.putText(image, result_str, (xmin + 50, ymin + 70),
                            cv2.FONT_HERSHEY_TRIPLEX, 2, (246, 152, 40), 5, cv2.LINE_AA)
            cv2.imshow('current', image)
        idx += 1
        print(idx)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    print('Finish')
