import os
import argparse
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser('將整個資料夾底下的PNG圖像轉成JPG圖像')
    # 圖像資料夾
    parser.add_argument('--image-folder', type=str, default='./imgs')
    # 保存資料夾
    parser.add_argument('--save-folder', type=str, default='./save')
    args = parser.parse_args()
    return args


def convert_image_format(image_folder, save_folder):
    support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']
    for file_name in os.listdir(image_folder):
        file_path = os.path.join(image_folder, file_name)
        if os.path.isdir(file_path):
            save_folder_path = os.path.join(save_folder, file_name)
            if not os.path.exists(save_folder_path):
                os.mkdir(save_folder_path)
            convert_image_format(file_path, save_folder_path)
        elif os.path.splitext(file_name)[1] in support_image_format:
            save_path = os.path.join(save_folder, os.path.splitext(file_name)[0] + '.jpg')
            image = Image.open(file_path)
            if not image.mode == 'RGB':
                image = image.convert('RGB')
            image.save(save_path)


def main():
    args = parse_args()
    image_folder = args.image_folder
    save_folder = args.save_folder
    assert os.path.exists(image_folder), '提供的影片資料夾不存在'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    convert_image_format(image_folder, save_folder)


if __name__ == '__main__':
    main()
