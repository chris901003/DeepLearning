import argparse
import json
import os
from tqdm import tqdm
import cv2


def parse_args():
    parser = argparse.ArgumentParser('將coco數據集指定類別提取出來')
    # coco的標註檔案，請確認有與圖像資料夾匹配
    parser.add_argument('--coco-json-file', type=str, default='./instances_val2017.json')
    # coco的圖像檔案，請確認有與標註檔案匹配
    parser.add_argument('--coco-image-folder', type=str, default='./val2017')
    # 保存路徑，強烈建議在運行前將指定保存資料夾清空
    parser.add_argument('--save-path', type=str, default='./save')
    # 需要提取出來的類別
    parser.add_argument('--extract-classes', type=int, default=[1], nargs='+')
    # 是否需要對擷取位置擴大
    parser.add_argument('--extend-picture', action='store_true')
    # 往外擴張比例，這裡會是原始標註框的高寬乘上多少倍
    parser.add_argument('--extend-percent', type=float, default=0.3)
    # 保存jpg圖像質量
    parser.add_argument('--quality', type=int, default=95)
    # 是否需要同時生成labelImg的標註文件，這樣就可以在貼合圖像時圖時標註
    parser.add_argument('--save-annotation', action='store_true')
    # 如果有需要生成annotation文件就可以指定保存地址
    parser.add_argument('--save-annotation-path', type=str, default='./save_annotation')
    # 提取出來的圖像對應類別，這裡的長度要與extract-classes數量相同
    parser.add_argument('--classes-label', type=int, default=[9], nargs='+')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    json_file = args.coco_json_file
    image_folder = args.coco_image_folder
    save_path = args.save_path
    extract_classes = args.extract_classes
    extend_picture = args.extend_picture
    extend_percent = args.extend_percent / 2
    save_annotation = args.save_annotation
    save_annotation_path = args.save_annotation_path
    classes_label = args.classes_label
    if save_annotation:
        assert len(extract_classes) == len(classes_label), '需提供對應的類別編號'
    extract_class_to_anno_class = dict()
    for extract_cls, cls_label in zip(extract_classes, classes_label):
        extract_class_to_anno_class[extract_cls] = cls_label
    assert os.path.exists(json_file) and os.path.exists(image_folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_annotation_path):
        os.mkdir(save_annotation_path)
    for idx in extract_classes:
        folder_name = os.path.join(save_path, str(idx))
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
    assert len(extract_classes) > 0, '需要至少指定一種類別'
    with open(json_file, 'r') as f:
        coco_json_info = json.load(f)
    images_info = coco_json_info['images']
    annotations_info = coco_json_info['annotations']
    index2image = dict()
    for image_info in tqdm(images_info):
        image_name = image_info['file_name']
        image_id = int(image_info['id'])
        index2image[image_id] = image_name
    print(f'Load {len(index2image)} picture name')
    total_cut_image = 0
    for annotation_info in tqdm(annotations_info):
        image_id = int(annotation_info['image_id'])
        bbox = annotation_info['bbox']
        category_id = int(annotation_info['category_id'])
        if category_id not in extract_classes:
            continue
        xmin, ymin, width, height = bbox
        x_center = width / 2
        y_center = height / 2
        xmax = xmin + width
        ymax = ymin + height
        image_name = index2image[image_id]
        image_path = os.path.join(image_folder, image_name)
        assert os.path.exists(image_path), f'圖像 {image_name} 找不到'
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]
        if extend_picture:
            ori_x, ori_y = xmin, ymin
            width_extend = width * extend_percent
            height_extend = height * extend_percent
            xmin = max(0, xmin - width_extend)
            ymin = max(0, ymin - height_extend)
            xmax = min(img_width, xmax + width_extend)
            ymax = min(img_height, ymax + height_extend)
            offset_x, offset_y = ori_x - xmin, ori_y - ymin
            x_center, y_center = x_center + offset_x, y_center + offset_y
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cut_image = image[ymin:ymax + 1, xmin:xmax + 1, :]
        save_image_path = os.path.join(save_path, str(category_id), str(total_cut_image) + '.jpg')
        cv2.imwrite(save_image_path, cut_image, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        if save_annotation:
            cut_image_height, cut_image_width = cut_image.shape[:2]
            x_center, y_center = x_center / cut_image_width, y_center / cut_image_height
            anno_width, anno_height = width / cut_image_width, height / cut_image_height
            label = extract_class_to_anno_class[category_id]
            res = str(label) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + \
                  str(anno_width) + ' ' + str(anno_height)
            annotation_path = os.path.join(save_annotation_path, str(total_cut_image) + '.txt')
            with open(annotation_path, 'w') as f:
                f.write(res)
        total_cut_image += 1
    print('Finish')


if __name__ == '__main__':
    main()
