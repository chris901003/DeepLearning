import argparse
import cv2
from SpecialTopic.ST.utils import get_classes
from api import init_model, detect_single_picture


def parse_args():
    parser = argparse.ArgumentParser('單張圖像檢測')
    # 使用的模型類型
    parser.add_argument('--model-type', type=str, default='ResNet')
    # 模型尺寸
    parser.add_argument('--phi', type=str, default='m')
    # 類別文件
    parser.add_argument('--classes-path', type=str, default='./classes.txt')
    # 圖像路徑
    parser.add_argument('--image-path', type=str, default='./test.jpg')
    # 預訓練權重地址
    parser.add_argument('--pretrained-path', type=str, default='./pretrained.pth')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    classes_name, num_classes = get_classes(args.classes_path)
    model = init_model(num_classes=num_classes, pretrained=args.pretrained_path)
    image = cv2.imread(args.image_path)
    output = detect_single_picture(model, image)[0]
    pred = output.argmax(dim=0).item()
    print(classes_name[pred])


if __name__ == '__main__':
    main()
