import json
import os
import argparse
from datetime import datetime
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser('Transform labelImg yolo format to coco format')
    parser.add_argument('--source', default='./data', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    source_path = args.source
    assert os.path.exists(source_path) and os.path.isdir(source_path), f'資料源 {source_path} 不存在或是不是資料夾'
    info_data = {
        'description': os.path.basename(source_path) + ' Dataset',
        'url': 'https://drive.google.com/file/d/1JNj0fI72NLzPQUIo8Duuc8t2y9f_JEGF/view?usp=sharing',
        'version': '1.0',
        'year': 2022,
        'contributor': 'Edieth',
        'date_created': datetime.today().strftime('%Y-%m-%d')
    }
    licenses_data = [
        {
            'url': 'https://drive.google.com/file/d/1JNj0fI72NLzPQUIo8Duuc8t2y9f_JEGF/view?usp=sharing',
            'id': 1,
            'name': 'Online picture'
        }
    ]
    images_data, anno_name_to_hw = get_images_data(source_path)
    annotations_data = get_annotations_data(source_path, anno_name_to_hw)
    categories_data = get_categories_data(source_path)
    annotation_file = {
        'info': info_data,
        'licenses': licenses_data,
        'images': images_data,
        'annotations': annotations_data,
        'categories': categories_data
    }
    save_path = os.path.join(source_path, 'self_annotation.json')
    with open(save_path, "w") as f:
        json.dump(annotation_file, f, indent=4)
    print('Finish')


def get_images_data(source_path):
    img_path = os.path.join(source_path, 'imgs')
    assert os.path.exists(img_path), f'圖像要保存在 {img_path} 當中，如果有存放錯誤請改正'
    images = list()
    support_img_format = ['.jpeg', '.JPEG', '.jpg', '.JPG']
    images_name = os.listdir(img_path)
    images_name = sorted(images_name)
    label_name_to_hw = dict()
    for image_name in tqdm(images_name, desc='Create images ... '):
        if os.path.splitext(image_name)[1] not in support_img_format:
            continue
        image_path = os.path.join(img_path, image_name)
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        picture_id = int(os.path.splitext(image_name)[0])
        annotation_name = os.path.splitext(image_name)[0] + '.txt'
        label_name_to_hw[annotation_name] = (height, width, picture_id)
        data = {
            'license': 1,
            'file_name': image_name,
            'coco_url': 'https://drive.google.com/file/d/1JNj0fI72NLzPQUIo8Duuc8t2y9f_JEGF/view?usp=sharing',
            'height': height,
            'width': width,
            'data_captured': datetime.today().strftime('%Y-%m-%d'),
            'flickr_url': 'https://drive.google.com/file/d/1JNj0fI72NLzPQUIo8Duuc8t2y9f_JEGF/view?usp=sharing',
            'id': int(os.path.splitext(image_name)[0])
        }
        images.append(data)
    return images, label_name_to_hw


def get_annotations_data(source_data, anno_name_to_hw):
    annotations_path = os.path.join(source_data, 'annotations')
    assert os.path.exists(annotations_path), f'無法找到 {annotations_path} 來獲取yolo標註訊息，如果保存位置有誤請更正'
    annotations = list()
    annotations_name = os.listdir(annotations_path)
    annotations_name = sorted(annotations_name)
    idx = 0
    for annotation_name in tqdm(annotations_name, desc='Create annotations ... '):
        if annotation_name in ['classes.txt']:
            continue
        img_height, img_width, picture_id = anno_name_to_hw.get(annotation_name, None)
        assert img_height and img_width is not None, '無法獲取標註訊息的高寬'
        annotation_path = os.path.join(annotations_path, annotation_name)
        assert os.path.isfile(annotation_path)
        with open(annotation_path) as f:
            annos = f.readlines()
            for anno in annos:
                cls, center_x, center_y, width, height = anno.split(' ')
                cls = int(cls) + 1
                center_x = float(center_x) * img_width
                center_y = float(center_y) * img_height
                width = float(width) * img_width
                height = float(height) * img_height
                xmin = max(0, center_x - width / 2)
                ymin = max(0, center_y - height / 2)
                xmax = min(img_width, center_x + width / 2)
                ymax = min(img_height, center_y + height / 2)
                data = {
                    'segmentation': [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]],
                    'area': width * height,
                    'iscrowd': 0,
                    'image_id': picture_id,
                    'bbox': [xmin, ymin, width, height],
                    'category_id': cls,
                    'id': idx
                }
                idx += 1
                annotations.append(data)
    return annotations


def get_categories_data(source_data):
    categories_file = os.path.join(source_data, 'classes.txt')
    assert os.path.isfile(categories_file), f'請將classes.txt直接放在source_data底下 {categories_file}'
    categories = list()
    with open(categories_file) as f:
        classes_list = f.readlines()
        for idx, classes_name in enumerate(classes_list):
            data = {
                'supercategory': classes_name.strip(),
                'id': idx + 1,
                'name': classes_name.strip()
            }
            categories.append(data)
    return categories


if __name__ == '__main__':
    main()
