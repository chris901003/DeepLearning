import os
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from shutil import copyfile


def parse_args():
    parser = argparse.ArgumentParser('Generate Picture with gt boxes')
    # 圖像資料夾地址
    parser.add_argument('--img', default='/Users/huanghongyan/Downloads/data_annotation/imgs', type=str)
    # 標註文件地址
    parser.add_argument('--annotation', default='/Users/huanghongyan/Downloads/data_annotation/annotations', type=str)
    # 背景圖地止，如果使用none不指定就會隨機生成顏色作為背景
    parser.add_argument('--background', default='none', type=str)
    # 如果有需要合成圖像且該圖像不帶有標註，創造負樣本使用
    parser.add_argument('--img-without-annotation', default='none', type=str)
    # 存放位置，auto表示會在img資料夾下建立generate_img資料夾，並且會將圖像存放在該資料架下
    parser.add_argument('--save', default='auto', type=str)
    # 總共需要生產多少張圖像
    parser.add_argument('--num_picture', default=10, type=int)
    # 圖像大小，如果傳入的是tuple就會是(高, 寬)，如果只傳入一個值就會是正方形的圖像
    parser.add_argument('--picture-size', default=[640, 640], nargs='+', type=int)
    # 一張圖上會重疊多少張圖像，如果是tuple就是最多與最少，如果是一個值就會是指定的數量(固定的貼圖數)
    parser.add_argument('--combine', default=[3, 7], nargs='+', type=int)
    # 對於一張圖像中有多個目標圖片，放到畫布上的大小範圍，如果只輸入一個數就是直接固定大小
    parser.add_argument('--size-multi-object', default=[300, 450], nargs='+', type=int)
    # 對於一張圖像中只有單個目標圖片，放到畫布上的大小範圍，如果只輸入一個數就是直接固定大小
    parser.add_argument('--size-single-object', default=[200, 300], nargs='+', type=int)
    # 是否需要保持圖像高寬比
    parser.add_argument('--keep-ratio', action='store_true')
    # 堆疊照片時左上角點的最近距離(如果嘗試過多次也會直接放棄)
    parser.add_argument('--picture-distance', default=[150, 150], nargs='+', type=int)
    args = parser.parse_args()
    return args


def get_img_data(imgs_path, annotations_path):
    support_img_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    result = list()
    for img_name in os.listdir(imgs_path):
        if os.path.splitext(img_name)[1] not in support_img_format:
            continue
        annotation_name = os.path.splitext(img_name)[0] + '.txt'
        img_path = os.path.join(imgs_path, img_name)
        annotation_path = os.path.join(annotations_path, annotation_name)
        if not os.path.isfile(annotation_path):
            continue
        bboxes = list()
        labels = list()
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        with open(annotation_path) as f:
            annotations = f.readlines()
        for annotation in annotations:
            label, center_x, center_y, width, height = annotation.strip().split(' ')
            label = int(label)
            center_x = float(center_x) * img_width
            center_y = float(center_y) * img_height
            width = float(width) * img_width
            height = float(height) * img_height
            xmin = max(0, center_x - width / 2)
            ymin = max(0, center_y - height / 2)
            xmax = min(center_x + width / 2, img_width)
            ymax = min(center_y + height / 2, img_height)
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        del img
        data = dict(img_path=img_path, bboxes=bboxes, labels=labels)
        result.append(data)
    return result


def get_negative_img(img_folder):
    result = list()
    support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    for image_name in os.listdir(img_folder):
        if os.path.splitext(image_name)[1] not in support_image_format:
            continue
        image_path = os.path.join(img_folder, image_name)
        bboxes, labels = list(), list()
        bboxes.append([-1, -1, -1, -1])
        labels.append(-1)
        data = dict(img_path=image_path, bboxes=bboxes, labels=labels)
        result.append(data)
    return result


def resize(img, height, width, keep_ratio):
    img_height, img_width = img.shape[:2]
    if keep_ratio:
        img_scale = min(height / img_height, width / img_width)
        img = cv2.resize(img, (int(img_width * img_scale), int(img_height * img_scale)),
                         interpolation=cv2.INTER_NEAREST)
        new_height, new_width = img.shape[:2]
        height_scale, width_scale = new_height / img_height, new_width / img_width
        return dict(image=img, height_scale=height_scale, width_scale=width_scale)
    else:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        new_height, new_width = img.shape[:2]
        height_scale, width_scale = new_height / img_height, new_width / img_width
        return dict(image=img, height_scale=height_scale, width_scale=width_scale)


def check_pos(xmin, ymin, record_pos, picture_distance):
    for pos in record_pos:
        dis_x = abs(xmin - pos[0])
        dis_y = abs(ymin - pos[1])
        if dis_x < picture_distance[0] and dis_y < picture_distance[1]:
            return False
    return True


def filter_bboxes_in_picture(bboxes):
    result = [idx for idx in range(len(bboxes)) if (bboxes[idx][2] - bboxes[idx][0]) >= 1
              and (bboxes[idx][3] - bboxes[idx][1]) >= 1]
    return result


def get_valid_bboxes(boxes1, boxes2, max_area):
    threshold = 0.7
    cover_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = rb - lt
    np.clip(wh, 0, max_area, out=wh)
    area = wh[..., 0] * wh[..., 1]
    overlap = area > (cover_area[:, None] * threshold)
    overlap = overlap.sum(axis=1)
    valid = overlap == 0
    return valid


def filter_bboxes_iou(total_bboxes, total_labels, max_area, xmin, ymin, xmax, ymax):
    target_bboxes = np.array([xmin, ymin, xmax, ymax])
    target_bboxes = target_bboxes[None]
    for idx, bboxes in enumerate(total_bboxes[:-1]):
        valid_bboxes = get_valid_bboxes(bboxes, target_bboxes, max_area)
        total_bboxes[idx] = total_bboxes[idx][valid_bboxes]
        total_labels[idx] = total_labels[idx][valid_bboxes]


def filter_bboxes_cover(total_bboxes, total_labels, xmin, ymin, xmax, ymax):
    for index, bboxes in enumerate(total_bboxes[:-1]):
        valid_list = list()
        for bbox in bboxes:
            valid = True
            if bbox[0] <= xmin and xmax <= bbox[2]:
                valid = False
            elif bbox[1] <= ymin and ymax <= bbox[3]:
                valid = False

            if ymin <= bbox[1] <= bbox[3] <= ymax or ymin <= bbox[1] <= ymax <= bbox[3] \
                    or bbox[1] <= ymin <= bbox[3] <= ymax:
                if xmin <= bbox[0] <= bbox[2] <= xmax:
                    pass
                elif bbox[0] < xmin < bbox[2] < xmax:
                    bbox[2] = xmin
                elif xmin < bbox[0] < xmax < bbox[2]:
                    bbox[0] = xmax

            if xmin <= bbox[0] <= bbox[2] <= xmax or xmin <= bbox[0] <= xmax <= bbox[2] \
                    or bbox[0] <= xmin <= bbox[2] <= xmax:
                if ymin <= bbox[1] <= bbox[3] <= ymax:
                    pass
                elif bbox[1] < ymin < bbox[3] < ymax:
                    bbox[3] = ymin
                elif ymin < bbox[1] < ymax < bbox[3]:
                    bbox[1] = ymax
            valid_list.append(valid)
        total_bboxes[index] = total_bboxes[index][valid_list]
        total_labels[index] = total_labels[index][valid_list]


def main():
    args = parse_args()
    assert os.path.isdir(args.img), '給定圖像路徑不是資料夾路徑'
    assert os.path.isdir(args.annotation), '給定標註文件路徑不是資料夾路徑'
    use_background = True if args.background != 'none' else False
    save_path = args.save if args.save != 'auto' else os.path.join(args.img, 'generate_img')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_save_path = os.path.join(save_path, 'imgs')
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    annotation_save_path = os.path.join(save_path, 'annotations')
    if not os.path.exists(annotation_save_path):
        os.mkdir(annotation_save_path)
    classes_txt = os.path.join(args.annotation, 'classes.txt')
    copyfile(classes_txt, os.path.join(annotation_save_path, 'classes.txt'))
    data_info = get_img_data(args.img, args.annotation)
    if args.img_without_annotation != 'none':
        negative_info = get_negative_img(args.img_without_annotation)
        data_info.extend(negative_info)
    assert len(data_info) > 0, '沒有圖像資料可以使用'

    background = None
    support_img_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    if use_background:
        background = [os.path.join(args.background, background_name) for background_name in os.listdir(args.background)
                      if os.path.splitext(background_name)[1] in support_img_format]

    combine = args.combine
    if len(combine) == 1:
        combine = [combine, combine + 1]
    elif len(combine) != 2:
        raise ValueError('Combine資訊錯誤')
    picture_size = args.picture_size
    if len(picture_size) == 1:
        picture_size = [picture_size, picture_size]
    elif len(picture_size) != 2:
        raise ValueError('Picture-size設定錯誤')

    picture_size_multi_object = args.size_multi_object
    if len(picture_size_multi_object) == 1:
        picture_size_multi_object = [picture_size_multi_object, picture_size_multi_object + 1]
    elif len(picture_size_multi_object) != 2:
        raise ValueError('size-multi-object輸入錯誤')
    picture_size_single_object = args.size_single_object
    if len(picture_size_single_object) == 1:
        picture_size_single_object = [picture_size_single_object, picture_size_single_object + 1]
    elif len(picture_size_single_object) != 2:
        raise ValueError('size-single-object輸入錯誤')

    picture_distance = args.picture_distance
    if len(picture_distance) == 1:
        picture_distance = [picture_distance, picture_distance]
    elif len(picture_distance) != 2:
        raise ValueError('picture-distance出入錯誤')

    for idx in tqdm(range(args.num_picture)):
        record_pos = list()
        total_bboxes = list()
        total_labels = list()
        number_of_pictures = np.random.randint(low=combine[0], high=combine[1])
        pictures_idx = np.random.randint(low=0, high=len(data_info), size=number_of_pictures)
        padding = np.random.randint(low=0, high=255, size=3)
        image = np.full(picture_size + [3], padding, dtype=np.uint8)

        if use_background and len(background) > 0:
            background_idx = np.random.randint(low=0, high=len(background))
            background_path = background[background_idx]
            background_img = cv2.imread(background_path)
            background_img = cv2.resize(background_img, picture_size, interpolation=cv2.INTER_AREA)
            image = background_img

        for picture_idx in pictures_idx:
            picture_info = data_info[picture_idx]
            picture_size_range = picture_size_single_object if len(picture_info['labels']) == 1 \
                else picture_size_multi_object
            picture_path = picture_info['img_path']
            bboxes = picture_info['bboxes']
            labels = picture_info['labels']
            bboxes = np.array(bboxes)
            labels = np.array(labels)
            img_width, img_height = np.random.randint(low=picture_size_range[0], high=picture_size_range[1], size=2)
            picture = cv2.imread(picture_path)
            results = resize(picture, img_height, img_width, args.keep_ratio)
            picture = results['image']
            height_scale = results['height_scale']
            width_scale = results['width_scale']
            img_height, img_width = picture.shape[:2]
            bboxes[:, 0::2] = bboxes[:, 0::2] * width_scale
            bboxes[:, 1::2] = bboxes[:, 1::2] * height_scale
            xmin, ymin = 0, 0
            for _ in range(15):
                xmin = np.random.randint(low=0, high=picture_size[1] - 200)
                ymin = np.random.randint(low=0, high=picture_size[0] - 200)
                if check_pos(xmin, ymin, record_pos, picture_distance):
                    break
            record_pos.append([xmin, ymin])
            xmax = min(xmin + img_width, picture_size[1])
            ymax = min(ymin + img_height, picture_size[0])
            new_img_height, new_img_width = ymax - ymin, xmax - xmin
            image[ymin: ymax, xmin: xmax] = picture[:new_img_height, :new_img_width]
            bboxes[:, 0::2] = bboxes[:, 0::2] + xmin
            bboxes[:, 1::2] = bboxes[:, 1::2] + ymin
            np.clip(bboxes[:, 0::2], 0, picture_size[1], out=bboxes[:, 0::2])
            np.clip(bboxes[:, 1::2], 0, picture_size[0], out=bboxes[:, 1::2])
            valid_bboxes_index = filter_bboxes_in_picture(bboxes)
            bboxes = bboxes[valid_bboxes_index]
            labels = labels[valid_bboxes_index]
            total_bboxes.append(bboxes)
            total_labels.append(labels)
            filter_bboxes_iou(total_bboxes, total_labels, picture_size[0] * picture_size[1], xmin, ymin, xmax, ymax)
            filter_bboxes_cover(total_bboxes, total_labels, xmin, ymin, xmax, ymax)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image.save(os.path.join(img_save_path, f'{idx}.jpg'))
        annotation_path = os.path.join(annotation_save_path, f'{idx}.txt')
        with open(annotation_path, 'w') as f:
            for labels, bboxes in zip(total_labels, total_bboxes):
                bboxes[..., 0::2] = bboxes[..., 0::2] / picture_size[1]
                bboxes[..., 1::2] = bboxes[..., 1::2] / picture_size[0]
                width = bboxes[..., 2] - bboxes[..., 0]
                height = bboxes[..., 3] - bboxes[..., 1]
                center_x = bboxes[..., 0] + width / 2
                center_y = bboxes[..., 1] + height / 2
                labels = labels.tolist()
                for label, x, y, w, h in zip(labels, center_x, center_y, width, height):
                    anno = [str(label), str(x), str(y), str(w), str(h)]
                    anno = ' '.join(anno)
                    f.write(anno + '\n')


if __name__ == '__main__':
    main()
    print('Finish')
