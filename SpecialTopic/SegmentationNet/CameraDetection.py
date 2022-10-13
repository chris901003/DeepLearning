import cv2
import argparse
import torch
import time
from SpecialTopic.ST.utils import get_classes
from SpecialTopic.SegmentationNet.api import init_module, detect_single_picture


def parse_args():
    parser = argparse.ArgumentParser('Camera detection for segmentation')
    # 設定要使用的攝影機ID
    parser.add_argument('--camera-id', type=int, default=0)
    # 模型類型
    parser.add_argument('--model-type', type=str, default='Segformer')
    # 模型的大小
    parser.add_argument('--phi', type=str, default='m')
    # 類別文件
    parser.add_argument('--classes-path', type=str, default='./classes.txt')
    # 訓練權重路徑
    parser.add_argument('--pretrained', type=str, default='/Users/huanghongyan/Downloads/segformer_mit-b2_512x512_16'
                                                          '0k_ade20k_20220620_114047-64e4feca.pth')
    # 要使用哪個調色盤
    parser.add_argument('--with-color-platte', type=str, default='ADE20KDataset')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classes_name, num_classes = get_classes(args.classes_path)
    model = init_module(model_type=args.model_type, phi=args.phi, pretrained=args.pretrained, num_classes=num_classes,
                        device=device, with_color_platte=args.with_color_platte)
    cap = cv2.VideoCapture(0)
    pTime = 0

    while True:
        ret, img = cap.read()
        if ret:
            with torch.no_grad():
                results = detect_single_picture(model, device, img, with_class=True)
            draw_image_mix = results[0]
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(draw_image_mix, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('img', draw_image_mix)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    print('Camera detection for segmentation')
    main()
