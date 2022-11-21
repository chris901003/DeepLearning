import os
import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm


def args_parse():
    parser = argparse.ArgumentParser()
    # 圖像資料夾位置
    parser.add_argument('--image-folder-path', type=str, default=r'C:\Dataset\SegmentationFoodRemain\seg_\images\training')
    # 標註文件資料夾位置
    parser.add_argument('--annotation-folder-path', type=str, default=r'C:\Dataset\SegmentationFoodRemain\seg_\jsons\training')
    # 圖像保存位置
    parser.add_argument('--save-folder-path', type=str, default=r'C:\Dataset\SegmentationFoodRemain\seg_\annotations\training')
    # 總共有哪些類別，這裡的順序會與到時圖像的時候相同，所以如果要避免被覆蓋就需要調整順序
    parser.add_argument('--labels', type=str, default=['NotFood', 'Food'],
                        nargs='+')
    # 是否需要嚴格規定每張圖像當中都需要有labels的標註，當strict_check_label為False時就不會嚴格檢查
    parser.add_argument('--strict-check-label', action='store_true')
    # 要標註的idx，這裡的長度需要與labels相同
    parser.add_argument('--labels-idx', type=int, default=[2, 1], nargs='+')
    # 背景值
    parser.add_argument('--background-val', type=int, default=3)
    args = parser.parse_args()
    return args


def check_picture():
    from PIL import Image
    image_path = r'C:\Dataset\SegmentationFoodRemain\seg_\annotations\training\199.png'
    idx_to_rgb = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    png_image = cv2.imread(image_path)[..., 0]
    image_height, image_width = png_image.shape[:2]
    # image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image = [[idx_to_rgb[png_image[i][j]][::-1] for j in range(image_width)] for i in range(image_height)]
    image = np.array(image, dtype=np.uint8)
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img.show()


def to_int(points):
    results = [[int(x), int(y)] for x, y in points]
    return results


def main():
    args = args_parse()
    image_folder_path = args.image_folder_path
    annotation_folder_path = args.annotation_folder_path
    save_folder_path = args.save_folder_path
    labels = args.labels
    labels_idx = args.labels_idx
    background_val = args.background_val
    strict_check_label = args.strict_check_label
    assert len(labels) == len(labels_idx), '標註名稱與標註值數量需要相同'
    labels2labels2_idx = {label: label_idx for label, label_idx in zip(labels, labels_idx)}
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
        seg_image = np.full((image_height, image_width, 3), background_val, dtype=np.uint8)
        points_dict = dict()
        shapes_info = annotations['shapes']
        for shape_info in shapes_info:
            label = shape_info['label']
            point = np.array(to_int(shape_info['points']))
            if label not in points_dict.keys():
                points_dict[label] = list()
            points_dict[label].append(point)
        for label in labels:
            points = points_dict.get(label, None)
            if strict_check_label:
                assert points is not None, f'{image_name}沒有標到{label}請不要這麼走心'
            else:
                if points is None:
                    continue
            color = labels2labels2_idx[label]
            for point in points:
                cv2.fillPoly(seg_image, [point], (color, color, color))
        save_name = os.path.splitext(image_name)[0] + '.png'
        save_path = os.path.join(save_folder_path, save_name)
        cv2.imwrite(save_path, seg_image)
        pbar.update(1)


if __name__ == '__main__':
    print('Create annotation png picture')
    check_picture()
