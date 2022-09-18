import cv2
import os
from pathlib import Path
from tqdm import tqdm


def main():
    file_path = '/Users/huanghongyan/Downloads/food_data_flag/img'
    target_path = '/Users/huanghongyan/Downloads/imgs'
    assert os.path.exists(file_path)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    support_img = ['.png', '.jpg', '.JPG', '.jpeg', '.JPEG']
    if os.path.isfile(file_path):
        assert os.path.splitext(file_path)[1] in support_img
        file_name = Path(file_path).stem
        save_path = os.path.join(target_path, file_name + '.jpg')
        img = cv2.imread(file_path, 1)
        cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    else:
        for img_path in tqdm(os.listdir(file_path)):
            if os.path.splitext(img_path)[1] in support_img:
                path = os.path.join(file_path, img_path)
                file_name = Path(path).stem
                save_path = os.path.join(target_path, file_name + '.jpg')
                img = cv2.imread(path, 1)
                cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 50])


if __name__ == '__main__':
    main()
