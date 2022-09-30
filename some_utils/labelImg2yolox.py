import os
import cv2
from tqdm import tqdm
import argparse


def parse_args():
    parse = argparse.ArgumentParser('將labelImg檔案資料轉成yolox訓練時需要的標註檔')
    # 圖像資料夾位置
    parse.add_argument('--image-folder', type=str, default='./imgs')
    # 標註文件資料夾位置
    parse.add_argument('--anno-folder', type=str, default='./annotations')
    # 標註檔保存位置
    parse.add_argument('--save-path', type=str, default='./2012_train.txt')
    args = parse.parse_args()
    return args


def main():
    args = parse_args()
    image_folder = args.image_folder
    anno_folder = args.anno_folder
    save_path = args.save_path
    support_image = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    imgs_name = [img_name for img_name in os.listdir(image_folder) if os.path.splitext(img_name)[1] in support_image]
    annos_name = [anno_name for anno_name in os.listdir(anno_folder) if os.path.splitext(anno_name)[1] == '.txt']
    imgs_name = sorted(imgs_name)
    annos_name = sorted(annos_name)
    for img_name, anno_name in tqdm(zip(imgs_name, annos_name)):
        if os.path.splitext(img_name)[0] != os.path.splitext(anno_name)[0]:
            assert ValueError
        img_path = os.path.join(image_folder, img_name)
        anno_path = os.path.join(anno_folder, anno_name)
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        with open(anno_path, 'r') as f:
            annos = f.readlines()
        targets = list()
        for anno in annos:
            label, center_x, center_y, w, h = anno.strip().split(' ')
            center_x = (float(center_x) * img_width)
            center_y = (float(center_y) * img_height)
            w = (float(w) * img_width)
            h = (float(h) * img_height)
            xmin = int(center_x - w / 2)
            ymin = int(center_y - h / 2)
            xmax = int(center_x + w / 2)
            ymax = int(center_y + h / 2)
            res = str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + label
            targets.append(res)
        annotation = img_path + ' ' + ' '.join(targets)
        with open(save_path, 'a') as f:
            f.write(annotation)
            f.write('\n')


if __name__ == '__main__':
    main()
