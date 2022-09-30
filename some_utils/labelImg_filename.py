import os
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser('更改檔名')
    # 圖像資料夾位置
    parser.add_argument('--images-folder', type=str, default='./imgs')
    # 標註文件資料夾位置
    parser.add_argument('--annotations-folder', type=str, default='./annotations')
    # 更新名稱後存放位置
    parser.add_argument('--save-path', type=str, default='./save')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    images_folder = args.images_folder
    annotations_folder = args.annotations_folder
    save_path = args.save_path
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    os.mkdir(os.path.join(save_path, 'annotations'))
    os.mkdir(os.path.join(save_path, 'imgs'))
    offset = 0
    support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    images_name = [image_name for image_name in os.listdir(images_folder)
                   if os.path.splitext(image_name)[1] in support_image_format]
    annotations_name = [os.path.splitext(image_name)[0] + '.txt' for image_name in images_name]
    images_name = sorted(images_name)
    annotations_name = sorted(annotations_name)
    tot = 0
    for idx, (image_name, annotation_name) in enumerate(zip(images_name, annotations_name)):
        image_path = os.path.join(images_folder, image_name)
        annotation_path = os.path.join(annotations_folder, annotation_name)
        assert os.path.exists(image_path) and os.path.exists(annotation_path)
        assert os.path.splitext(annotation_path)[1] == '.txt'
        last_name = os.path.splitext(image_name)[1]
        target_image_name = os.path.join(save_path, 'imgs', f'{idx * 2 + offset}{last_name}')
        target_annotation_name = os.path.join(save_path, 'annotations', f'{idx * 2 + offset}.txt')
        shutil.copyfile(image_path, target_image_name)
        shutil.copyfile(annotation_path, target_annotation_name)
        tot += 1
    print(f'Total change {tot} files')


if __name__ == '__main__':
    main()
