import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from utils import get_classes
from api import detect_image
from SpecialTopic.ST.build import build_detector


def parse_args():
    parser = argparse.ArgumentParser('Yolox single picture test')
    # 預訓練權重位置
    parser.add_argument('--models-path', type=str, default='/Users/huanghongyan/Downloads/best_weight.pth')
    # 模型大小，注意這裡要與預訓練權重匹配
    parser.add_argument('--phi', type=str, default='l')
    # 類別資訊
    parser.add_argument('--classes-path', type=str, default='/Users/huanghongyan/Downloads/food_data_flag/classes.txt')
    # 要預測的圖像路徑
    parser.add_argument('--image-path', type=str, default='/Users/huanghongyan/Downloads/food_data_flag/imgs/0.jpeg')
    # 最終輸入到網路的圖像大小，不是給的圖像大小
    parser.add_argument('--input-shape', type=int, default=[640, 640], nargs='+')
    # 置信度閾值
    parser.add_argument('--confidence', type=float, default=0.5)
    # nms處理閾值
    parser.add_argument('--nms-iou', type=float, default=0.3)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert os.path.exists(args.models_path), '指定的預訓練權重資料不存在'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class_names, num_classes = get_classes(args.classes_path)
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
    load_key, no_load_key, temp_dict = [], [], {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    assert len(no_load_key) == 0, '給定的預訓練權重與模型不匹配'
    results = detect_image(model, device, args.image_path, args.input_shape, num_classes, confidence=args.confidence,
                           nms_iou=args.nms_iou)
    labels, scores, bboxes = results
    image = cv2.imread(args.image_path)
    height, width = image.shape[:2]
    for label, score, bbox in zip(labels, scores, bboxes):
        ymin, xmin, ymax, xmax = bbox
        ymin = int(max(0, ymin))
        xmin = int(max(0, xmin))
        ymax = int(min(height, ymax))
        xmax = int(min(width, xmax))
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 10)
        result_str = str(class_names[int(label)]) + '|' + str(round(score, 2))
        cv2.putText(image, result_str, (xmin + 50, ymin + 70),
                    cv2.FONT_HERSHEY_TRIPLEX, 2, (246, 152, 40), 5, cv2.LINE_AA)
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.show()


if __name__ == '__main__':
    main()
