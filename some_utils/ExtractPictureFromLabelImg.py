import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser('將標註到的圖像獨立取出來，分類到指定類別當中')
    # 圖像資料夾位置
    parser.add_argument('--images-folder', type=str, default='none')
    # 標註資料夾位置
    parser.add_argument('--annotations-folder', type=str, default='none')
    # 保存資料夾位置，如果設定成auto就會放到當前目錄下的save當中
    parser.add_argument('--save-folder', type=str, default='auto')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    images_folder = args.images_folder
    annotations_folder = args.annotations_folder
    assert images_folder is not None and annotations_folder is not None, '需要提供照片以及標註文件的檔案路徑'
    support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    images_name = [image_name for image_name in images_folder
                   if os.path.splitext(image_name)[1] in support_image_format]
    images_name = sorted(images_name)


if __name__ == '__main__':
    main()
