import os
import cv2
from tqdm import tqdm


def main():
    image_folder = '/Users/huanghongyan/Downloads/food_data_flag/imgs'
    anno_folder = '/Users/huanghongyan/Downloads/food_data_flag/annotations'
    save_path = '/Users/huanghongyan/Downloads/food_data_flag/2012_train.txt'
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
