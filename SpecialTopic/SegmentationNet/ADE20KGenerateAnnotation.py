import argparse
import os


def args_parse():
    parser = argparse.ArgumentParser('將ADE20K數據寫成訓練用的標註文件')
    parser.add_argument('--data-root', type=str, default='/Users/huanghongyan/Documents/DeepLea'
                                                         'rning/mmsegmentation/data/ade/ADEChallengeData2016')
    parser.add_argument('--image-file-name', type=str, default='images')
    parser.add_argument('--train-image-file-name', type=str, default='training')
    parser.add_argument('--val-image-file-name', type=str, default='validation')
    parser.add_argument('--annotation-file-name', type=str, default='annotations')
    parser.add_argument('--train-annotation-file-name', type=str, default='training')
    parser.add_argument('--val-annotation-file-name', type=str, default='validation')
    # 是否需要同時生成驗證標註，如果沒有分出驗證集就需要開啟
    parser.add_argument('--with-val', action='store_true')
    args = parser.parse_args()
    return args


def generate_annotation_file(annotation_file_name, image_folder_path, anno_folder_path):
    images_name = os.listdir(image_folder_path)
    results = list()
    for image_name in images_name:
        name = os.path.splitext(image_name)[0]
        image_path = os.path.join(image_folder_path, image_name)
        anno_path = os.path.join(anno_folder_path, name + '.png')
        info = image_path + ' ' + anno_path
        results.append(info)
    with open(annotation_file_name, 'w') as f:
        for result in results:
            f.write(result)
            f.write('\n')
    print(f'Write {len(results)} images')


def main():
    args = args_parse()
    data_root = args.data_root
    image_file_path = os.path.join(data_root, args.image_file_name)
    train_image_path = os.path.join(image_file_path, args.train_image_file_name)
    val_image_path = os.path.join(image_file_path, args.val_image_file_name)
    anno_file_path = os.path.join(data_root, args.annotation_file_name)
    train_anno_path = os.path.join(anno_file_path, args.train_annotation_file_name)
    val_anno_path = os.path.join(anno_file_path, args.val_annotation_file_name)
    generate_annotation_file('train_annotation.txt', train_image_path, train_anno_path)
    if args.with_val:
        generate_annotation_file('eval_annotation.txt', val_image_path, val_anno_path)


if __name__ == '__main__':
    main()
    print('Finish')
