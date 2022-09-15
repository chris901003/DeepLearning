import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def check_dis(cur, record_pos, min_dis):
    min_dis = min_dis[0] ** 2 + min_dis[1] ** 2
    for (xmin, ymin) in record_pos:
        dis = (xmin - cur[0]) ** 2 + (ymin - cur[1]) ** 2
        if dis < min_dis:
            return False
    return True


def main():
    dataset_path = '/Users/huanghongyan/Downloads/UECFOOD100'
    background_path = '/Users/huanghongyan/Downloads/background'
    save_path = '/Users/huanghongyan/Downloads/data'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    num_picture_create = 300
    max_picture_mix = 7
    picture_size = (640, 640, 3)
    using_picture_classes = [1, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 23, 26, 27, 31, 34, 37, 42, 43, 51, 72, 81, 83, 84,
                             87, 89, 90, 94, 96]
    support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    num_classes = len(using_picture_classes)
    picture_path = list()
    for using_picture_class in using_picture_classes:
        picture_file_path = os.path.join(dataset_path, str(using_picture_class))
        if not os.path.isdir(picture_file_path):
            continue
        pictures_name = os.listdir(picture_file_path)
        picture_path.append([os.path.join(picture_file_path, picture_name) for picture_name in pictures_name
                             if os.path.splitext(picture_name)[1] in support_image_format])
    num_picture_cls = [len(cls) for cls in picture_path]
    background_picture_path = [os.path.join(background_path, background_name)
                               for background_name in os.listdir(background_path)
                               if os.path.splitext(background_name)[1] in support_image_format]
    num_background_picture = len(background_picture_path)
    for idx in tqdm(range(num_picture_create)):
        record_pos = list()
        pictures = np.random.randint(low=3, high=max_picture_mix + 1)
        padding = np.random.randint(low=0, high=255, size=3)
        image = np.full(picture_size, padding, dtype=np.uint8)
        if num_background_picture > 0:
            background_idx = np.random.randint(low=0, high=num_background_picture)
            background_image = cv2.imread(background_picture_path[background_idx])
            background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
            background_image = cv2.resize(background_image, (picture_size[0], picture_size[1]))
            image = background_image
        for _ in range(pictures):
            picture_cls = np.random.randint(low=0, high=num_classes)
            picture_idx = np.random.randint(low=0, high=num_picture_cls[picture_cls])
            image_path = picture_path[picture_cls][picture_idx]
            tmp_img = cv2.imread(image_path)
            tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2BGR)
            height, width = np.random.randint(low=200, high=300, size=2)
            tmp_img = cv2.resize(tmp_img, (height, width), interpolation=cv2.INTER_AREA)
            xmin, ymin = 0, 0
            for _ in range(15):
                xmin, ymin = np.random.randint(low=0, high=picture_size[0] - 150, size=2)
                if check_dis((xmin, ymin), record_pos, (150, 150)):
                    break
            record_pos.append((xmin, ymin))
            xmax = min(xmin + width, picture_size[0])
            ymax = min(ymin + height, picture_size[1])
            img_width = xmax - xmin
            img_height = ymax - ymin
            image[xmin:xmax, ymin:ymax] = tmp_img[:img_width, :img_height]
        img = Image.fromarray(image)
        img.save(os.path.join(save_path, f'{idx}.jpg'))


if __name__ == '__main__':
    main()
