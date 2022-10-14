import argparse
import os
import json
import shutil


def args_parse():
    parser = argparse.ArgumentParser()
    # 圖像資料資料夾
    parser.add_argument('--images-folder', type=str, default='./images')
    # 標註文件資料夾，如果圖像與標註文件在相同資料夾就填一樣就可以
    parser.add_argument('--annotations-folder', type=str, default='./annotations')
    # 要保存新圖像檔名的資料夾
    parser.add_argument('--save-images-folder', type=str, default='./save_images')
    # 要保存新標註文件的資料夾
    parser.add_argument('--save-annotations-folder', type=str, default='./save_annotations')
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    images_folder = args.images_folder
    annotations_folder = args.annotations_folder
    save_images_folder = args.save_images_folder
    save_annotations_folder = args.save_annotations_folder
    if not os.path.exists(save_images_folder):
        os.mkdir(save_images_folder)
    if not os.path.exists(save_annotations_folder):
        os.mkdir(save_annotations_folder)
    support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    images_name = [image_name for image_name in os.listdir(images_folder)
                   if os.path.splitext(image_name)[1] in support_image_format]
    for index, image_name in enumerate(images_name):
        name = os.path.splitext(image_name)[0]
        annotation_name = name + '.json'
        image_path = os.path.join(images_folder, image_name)
        annotation_path = os.path.join(annotations_folder, annotation_name)
        assert os.path.exists(annotation_path), f'標註文件{annotation_name}不存在'
        with open(annotation_path, 'r') as f:
            annotation_info = json.load(f)
            annotation_info['imagePath'] = str(index) + '.jpg'
        new_image_path = os.path.join(save_images_folder, str(index) + '.jpg')
        new_annotation_path = os.path.join(save_annotations_folder, str(index) + '.json')
        with open(new_annotation_path, 'w') as f:
            json.dump(annotation_info, f)
        shutil.copyfile(image_path, new_image_path)
    print(f'Rename {len(images_name)}')


if __name__ == '__main__':
    print('Rename labelme')
    main()
