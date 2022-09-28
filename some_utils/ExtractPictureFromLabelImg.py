import argparse
import os
import shutil
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser('將標註到的圖像獨立取出來，分類到指定類別當中')
    # 圖像資料夾位置
    parser.add_argument('--images-folder', type=str, default='/Users/huanghongyan/Downloads/data_annotation/imgs')
    # 標註資料夾位置
    parser.add_argument('--annotations-folder', type=str, default='/Users/huanghongyan/Downloads/'
                                                                  'data_annotation/annotations')
    # 如果設定成auto就默認classes文件會在annotations當中
    parser.add_argument('--classes-file', type=str, default='auto')
    # 保存資料夾位置，如果設定成auto就會放到當前目錄下的save當中，該檔案資料夾會在開始訓練時被清除
    parser.add_argument('--save-folder', type=str, default='auto')
    args = parser.parse_args()
    return args


def parse_classes_file(classes_file, with_classes_name=False):
    assert os.path.exists(classes_file), '需要在annotations資料夾當中提供classes或是自己指定'
    classes_name = list()
    with open(classes_file, 'r') as f:
        infos = f.readlines()
    for info in infos:
        classes_name.append(info.strip())
    if with_classes_name:
        return len(classes_name), classes_name
    else:
        return len(classes_name)


def main():
    args = parse_args()
    images_folder = args.images_folder
    annotations_folder = args.annotations_folder
    save_folder = args.save_folder
    if save_folder == 'auto':
        save_folder = './save'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)
    classes_file = args.classes_file
    if classes_file == 'auto':
        classes_file = os.path.join(annotations_folder, 'classes.txt')
    num_classes = parse_classes_file(classes_file)
    for idx in range(num_classes):
        folder_name = os.path.join(save_folder, str(idx))
        os.mkdir(folder_name)
    assert images_folder is not None and annotations_folder is not None, '需要提供照片以及標註文件的檔案路徑'
    support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    images_name = [image_name for image_name in images_folder
                   if os.path.splitext(image_name)[1] in support_image_format]
    images_name = sorted(images_name)
    for idx, image_name in enumerate(tqdm(images_name)):
        annotation_name = os.path.join(annotations_folder, os.path.splitext(image_name)[0] + '.txt')
        if not os.path.exists(annotation_name):
            print(f'{image_name}找不到對應的標註文件')
            continue


if __name__ == '__main__':
    main()
