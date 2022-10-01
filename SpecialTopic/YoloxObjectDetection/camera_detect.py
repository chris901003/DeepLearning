import argparse
import cv2
import torch
import time
from utils import get_classes
from api import init_model, detect_image


def parse_args():
    parser = argparse.ArgumentParser('利用攝影機進行即時檢測')
    # 使用的攝影機id，如果電腦只有一個攝影機就直接使用默認的0就可以
    parser.add_argument('--camera-id', type=int, default=0)
    # 模型訓練權重地址
    parser.add_argument('--pretrained', type=str, default='none')
    # 模型大小
    parser.add_argument('--phi', type=str, default='l')
    # 類別txt文件位置
    parser.add_argument('--classes-path', type=str, default='./classes.txt')
    # 置信度閾值
    parser.add_argument('--confidence', type=float, default=0.7)
    # nms時閾值
    parser.add_argument('--nms-iou', type=float, default=0.3)
    # 是否要過濾掉畫面邊緣預測
    parser.add_argument('--filter', action='store_false')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class_names, num_classes = get_classes(args.classes_path)
    model = init_model(pretrained=args.pretrained, num_classes=num_classes, device=device)
    cap = cv2.VideoCapture(args.camera_id)
    pTime = 0
    while True:
        ret, img = cap.read()
        if ret:
            img_height, img_width = img.shape[:2]
            results = detect_image(model, device, img, [640, 640], num_classes,
                                   confidence=args.confidence, nms_iou=args.nms_iou)
            labels, scores, bboxes = results
            for label, score, bbox in zip(labels, scores, bboxes):
                bbox = [int(box) for box in bbox]
                ymin, xmin, ymax, xmax = bbox
                if args.filter:
                    if ymin < 0 or xmin < 0 or ymax >= img_height or xmax >= img_width:
                        continue
                else:
                    ymin, xmin, ymax, xmax = max(0, ymin), max(0, xmin), min(img_height, ymax), min(img_width, xmax)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                info = class_names[label] + '|' + str(round(score * 100, 2))
                cv2.putText(img, info, (xmin + 30, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (89, 214, 210), 2, cv2.LINE_AA)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
