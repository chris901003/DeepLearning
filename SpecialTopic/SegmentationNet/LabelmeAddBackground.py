import argparse
import os
import json
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    # 原始圖像位置
    parser.add_argument('--images-folder', type=str, default='./images')
    # 標註文件位置
    parser.add_argument('--annotations-folder', type=str, default='./annotations')
    # 背景圖片檔案
    parser.add_argument('--background-folder', type=str, default='./backgrounds')
    # 保存新圖像位置
    parser.add_argument('--save-images-folder', type=str, default='./save_images')
    # 保存新標註文件位置
    parser.add_argument('--save-annotations-folder', type=str, default='./save_annotations')
    # 最終輸出圖像的高寬範圍，高寬值會在指定區間隨機取值
    parser.add_argument('--output-size-range', type=int, default=[500, 1000], nargs='+')
    # 對圖像進行resize時是否需要保持原始圖像高寬比
    parser.add_argument('--keep-ratio', action='store_false')
    # 貼上背景後原圖佔畫面百分比，這裡表示從[0.5-0.8]隨機選一個比例
    parser.add_argument('--new-ratio', type=float, default=[0.5, 0.8], nargs='+')
    args = parser.parse_args()
    return args


def resize(image, output_height, output_width, keep_ratio=True, with_scale=False):
    origin_height, origin_width = image.shape[:2]
    if keep_ratio:
        scale = min(output_height / origin_height, output_width / origin_width)
        new_height, new_width = origin_height * scale, origin_width * scale
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    else:
        image = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_NEAREST)
    if with_scale:
        height, width = image.shape[:2]
        height_scale = height / origin_height
        width_scale = width / origin_width
        return image, (height_scale, width_scale)
    else:
        return image


def main():
    args = parse_args()
    images_folder = args.images_folder
    annotations_folder = args.annotations_folder
    background_folder = args.background_folder
    save_images_folder = args.save_images_folder
    save_annotations_folder = args.save_annotations_folder
    output_size_range = args.output_size_range
    keep_ratio = args.keep_ratio
    new_ratio = args.new_ratio
    assert os.path.exists(images_folder) and os.path.exists(annotations_folder)
    support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    background_images_name = [background_name for background_name in os.listdir(background_folder)
                              if os.path.splitext(background_name)[1] in support_image_format]
    background_images_path = [os.path.join(background_folder, image_name) for image_name in background_images_name]
    num_background = len(background_images_name)
    images_name = [image_name for image_name in os.listdir(images_folder)
                   if os.path.splitext(image_name) in support_image_format]
    for image_name in images_name:
        name = os.path.splitext(image_name)[0]
        image_path = os.path.join(images_folder, image_name)
        annotation_path = os.path.join(annotations_folder, name + '.json')
        assert os.path.exists(annotation_path), f'{name} 的標注文件不在 {annotation_path}'
        with open(annotation_path, 'r') as f:
            annotation_info = json.load(f)
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]
        background_index = np.random.randint(low=0, high=num_background, size=1)
        background_image = cv2.imread(background_images_path[background_index])
        output_height = np.random.randint(low=min(image_height, output_size_range[0]),
                                          high=max(image_height, output_size_range[1]), size=1)
        output_width = np.random.randint(low=min(image_width, output_size_range[0]),
                                         high=max(image_width, output_size_range[1]), size=1)
        background_image = resize(image=background_image, output_height=output_height, output_width=output_width,
                                  keep_ratio=keep_ratio)
        image_scale = np.random.uniform(new_ratio[0], new_ratio[1], size=1)
        new_image_height, new_image_width = int(image_height * image_scale), int(image_width * image_width)
        image, image_scale = resize(image, new_image_height, new_image_width, with_scale=True)
        background_image_height, background_image_width = background_image.shape[:2]
        image_height, image_width = image.shape[:2]
        x_offset = np.random.randint(low=0, high=background_image_width - image_width, size=1)
        y_offset = np.random.randint(low=0, high=background_image_height - image_height, size=1)
        result_image = background_image
        result_image[y_offset:y_offset + image_height, x_offset:x_offset + image_width, :] = image
        for shape in annotation_info['shapes']:
            for index, (x, y) in enumerate(shape['points']):
                x = x * image_scale[1] + x_offset
                y = y * image_scale[0] + y_offset
                shape['points'][index] = [x, y]
        save_image_path = os.path.join(save_images_folder, image_name)
        save_annotation_path = os.path.join(save_annotations_folder, name + '.json')
        cv2.imwrite(save_image_path, result_image)
        with open(save_annotation_path, 'w') as f:
            json.dump(save_annotation_path, f)
    print(f'Generate {len(images_name)}')


if __name__ == '__main__':
    print('Add background picture')
    main()
