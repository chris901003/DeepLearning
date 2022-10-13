import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm


def args_parse():
    parser = argparse.ArgumentParser()
    # 圖像資料夾位置
    parser.add_argument('--image-folder-path', type=str, default='/Users/huanghongyan/Downloads/fish')
    # 標註文件資料夾位置
    parser.add_argument('--annotation-folder-path', type=str, default='/Users/huanghongyan/Downloads/fish')
    # 圖像保存位置
    parser.add_argument('--save-folder-path', type=str, default='/Users/huanghongyan/Downloads/fish')
    # 將0設定成背景，如果關閉背景就會默認設定成255
    parser.add_argument('--zero-background', action='store_false')
    args = parser.parse_args()
    return args


def to_int(points):
    results = [[int(x), int(y)] for x, y in points]
    return results


def main():
    args = args_parse()
    image_folder_path = args.image_folder_path
    annotation_folder_path = args.annotation_folder_path
    save_folder_path = args.save_folder_path
    zero_background = args.zero_background
    assert os.path.exists(image_folder_path) and os.path.exists(annotation_folder_path), 'File is not exist'
    if not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)
    support_image_format = ['.jpg', '.jpeg', '.JPG', '.JPEG']
    images_name = [image_name for image_name in os.listdir(image_folder_path)
                   if os.path.splitext(image_name)[1] in support_image_format]
    pbar = tqdm(total=len(images_name))
    for idx, image_name in enumerate(images_name):
        image_path = os.path.join(image_folder_path, image_name)
        annotation_name = os.path.splitext(image_name)[0] + '.json'
        annotation_path = os.path.join(annotation_folder_path, annotation_name)
        assert os.path.exists(annotation_path), f'標註文件{annotation_name}不存在'
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]
        full_val = 0 if zero_background else 255
        seg_image = np.full((image_height, image_width, 3), full_val, dtype=np.uint8)
        shapes_info = annotations['shapes']
        foods_points = None
        not_foods_points = None
        for shape_info in shapes_info:
            label = shape_info['label']
            if label == 'Food':
                foods_points = np.array(to_int(shape_info['points']))
            elif label == 'NotFood':
                not_foods_points = np.array(to_int(shape_info['points']))
        if not_foods_points is not None:
            cv2.fillPoly(seg_image, [not_foods_points], (2, 2, 2))
        if foods_points is not None:
            cv2.fillPoly(seg_image, [foods_points], (1, 1, 1))
        save_name = os.path.splitext(image_name)[0] + '.png'
        save_path = os.path.join(save_folder_path, save_name)
        cv2.imwrite(save_path, seg_image)
        pbar.update(1)


if __name__ == '__main__':
    print('Create annotation png picture')
    main()
