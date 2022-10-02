import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser('生成訓練ClassifyNet所需的標註文件')
    # 圖像資料夾，該資料夾下會有不同剩餘量的資料夾，剩餘量就會是資料夾名稱
    parser.add_argument('--images-folder', type=str, default='./imgs')
    # 保存文件名稱
    parser.add_argument('--file-name', type=str, default='train_annotation.txt')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    images_folder = args.images_folder
    assert os.path.isdir(images_folder), '給定的圖像路徑需要是資料夾'
    label_name = [int(image_folder) for image_folder in os.listdir(images_folder) if image_folder.isdigit()]
    label_name = sorted(label_name)
    print(f'總共有{len(label_name)}種類別')
    support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    results = list()
    for idx, label_folder in enumerate(label_name):
        label_folder = str(label_folder)
        folder_path = os.path.join(images_folder, label_folder)
        for image_name in os.listdir(folder_path):
            if os.path.splitext(image_name)[1] not in support_image_format:
                continue
            image_path = os.path.join(folder_path, image_name)
            res = image_path + ' ' + str(idx)
            results.append(res)
    save_path = os.path.join(images_folder, args.file_name)
    with open(save_path, 'w') as f:
        for info in results:
            f.write(info)
            f.write('\n')
    print('Finish')


if __name__ == '__main__':
    main()
