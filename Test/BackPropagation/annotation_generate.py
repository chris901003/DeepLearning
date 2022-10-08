import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-folder', type=str, default='/Users/huanghongyan/Downloads/mnist_png/training')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    image_folder = args.image_folder
    images_folder_name = os.listdir(image_folder)
    support_image_format = ['.png', '.PNG', '.jpg', '.JPG']
    results = list()
    for image_folder_name in images_folder_name:
        if image_folder_name[0] == '.':
            continue
        image_folder_path = os.path.join(image_folder, image_folder_name)
        images_name = os.listdir(image_folder_path)
        for image_name in images_name:
            if os.path.splitext(image_name)[1] not in support_image_format:
                continue
            image_path = os.path.join(image_folder_path, image_name)
            label = str(image_folder_name)
            info = image_path + ' ' + label
            results.append(info)
    with open('train_annotation.txt', 'w') as f:
        for result in results:
            f.write(result)
            f.write('\n')
    print(f'Total {len(results)} photo')


if __name__ == '__main__':
    main()
