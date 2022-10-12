import os
import cv2
import argparse


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder-path', type=str, default='./images')
    parser.add_argument('--annotation-folder-path', type=str, default='./annotations')
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    image_folder_path = args.image_folder_path
    annotation_folder_path = args.annotation_folder_path
    assert os.path.exists(image_folder_path) and os.path.exists(annotation_folder_path), 'File is not exist'
    support_image_format = ['.jpg', '.jpeg', '.JPG', '.JPEG']
    images_name = [image_name for image_name in os.listdir(image_folder_path)
                   if os.path.splitext(image_name)[1] in support_image_format]
    for idx, image_name in enumerate(images_name):
        image_path = os.path.join(image_folder_path, image_name)
        annotation_name = os.path.splitext(image_name)[0] + '.txt'


if __name__ == '__main__':
    print('Create annotation png picture')
    main()
