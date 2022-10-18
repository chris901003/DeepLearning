import cv2
import os
import argparse
import torch
from PIL import Image
from SpecialTopic.ST.utils import get_classes
from SpecialTopic.SegmentationNet.api import init_module, detect_single_picture


def parse_args():
    parser = argparse.ArgumentParser()
    # 這裡可以直接放單張圖像的路徑或是資料夾路徑
    parser.add_argument('--image-path', type=str, default='/Users/huanghongyan/Downloads/temp')
    parser.add_argument('--model-type', type=str, default='Segformer')
    parser.add_argument('--phi', type=str, default='m')
    parser.add_argument('--classes-path', type=str, default='/Users/huanghongyan/Documents/DeepLearning/SpecialTopic/W'
                                                            'orkingFlow/prepare/remain_segformer_detection_classes.txt')
    parser.add_argument('--pretrained', type=str, default='/Users/huanghongyan/Downloads/1017_eval.pth')
    parser.add_argument('--with-color-platte', type=str, default='FoodAndNotFood')
    # 如果有想要看哪個類別佔多少個pixel就可以指定，-1表示不進行檢查
    parser.add_argument('--main-pixel', type=int, default=0)
    # 直接進行展示
    parser.add_argument('--show', action='store_false')
    # 如果設定成none就不會保存
    parser.add_argument('--save', type=str, default='none')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    show = args.show
    save = args.save
    main_pixel = args.main_pixel
    assert show or save != 'none', '不可以不顯示同時不保存，這樣沒有任何輸出'
    if save != 'none':
        if os.path.exists(save):
            os.mkdir(save)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classes_name, num_classes = get_classes(args.classes_path)
    model = init_module(model_type=args.model_type, phi=args.phi, pretrained=args.pretrained, num_classes=num_classes,
                        device=device, with_color_platte=args.with_color_platte)
    image_path = args.image_path
    if os.path.isdir(image_path):
        support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
        images_name = [image_name for image_name in os.listdir(image_path)
                       if os.path.splitext(image_name)[1] in support_image_format]
        root = image_path
        images_path = [os.path.join(root, image_name) for image_name in images_name]
    else:
        images_path = [image_path]
    for index, path in enumerate(images_path):
        image = cv2.imread(path)
        with torch.no_grad():
            results = detect_single_picture(model, device, image, with_class=True, threshold=0.8)
        draw_image_mix, draw_image, seg_pred = results
        if main_pixel != -1:
            image_height, image_width = image.shape[:2]
            total_main_pixel = (seg_pred == main_pixel).sum()
            percentage = total_main_pixel / (image_height * image_width)
            info = f'Total pixel: {image_height * image_width}, Main pixel: {total_main_pixel}'
            cv2.putText(draw_image_mix, info, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            info = f'Percentage: {percentage}'
            cv2.putText(draw_image_mix, info, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        img = Image.fromarray(cv2.cvtColor(draw_image_mix, cv2.COLOR_BGR2RGB))
        if show:
            img.show()
        if save != 'none':
            save_path = os.path.join(save, f'{index}.jpg')
            cv2.imwrite(save_path, draw_image_mix)


if __name__ == '__main__':
    main()
