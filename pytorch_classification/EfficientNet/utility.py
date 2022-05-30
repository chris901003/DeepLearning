import os
import json
from PIL import Image
import random


def preprocess_data(root: str):
    assert os.path.exists(root), f'Dataset root:{root} does not exist.'
    assert os.path.exists(os.path.join(root, 'train')), f'Train file dose not exit.'
    class_label = []
    for cla in os.listdir(os.path.join(root, 'train')):
        class_label.append(cla)
    assert os.path.exists(os.path.join(root, 'words.txt')), f'Labels txt not found.'
    label_file = open(os.path.join(root, 'words.txt'), 'r')
    tiny_image_labels_to_number = {}
    tiny_image_number_to_labels = {}
    count_labels = 0
    for label_txt in label_file.readlines():
        line = label_txt.split()
        if line[0] in class_label:
            tiny_image_labels_to_number[line[0]] = count_labels
            label_name = ''
            idx = 1
            while idx < len(line) and line[idx][-1] != ',':
                label_name += " " + line[idx]
                idx += 1
            if idx < len(line):
                label_name += " " + line[idx]
            label_name = label_name[1:]
            if label_name[-1] == ',':
                label_name = label_name[:-1]
            tiny_image_number_to_labels[count_labels] = label_name
            count_labels += 1
    with open('label_table.json', 'w') as json_file:
        json_file.write(json.dumps(tiny_image_number_to_labels, indent=4))
    train_images_path = []
    train_images_label = []
    test_images_path = []
    test_images_label = []
    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]
    assert os.path.exists(os.path.join(root, 'train')), 'Train dir is not found.'
    for class_file in os.listdir(os.path.join(root, 'train')):
        if os.path.isdir(os.path.join(root, 'train', class_file)):
            current_path = os.path.join(root, 'train', class_file, 'images')
            images = []
            for img in os.listdir(current_path):
                if os.path.splitext(img)[-1] in supported:
                    images.append(os.path.join(current_path, img))
            number = tiny_image_labels_to_number[class_file]
            for image_path in images:
                train_images_path.append(image_path)
                train_images_label.append(number)
    assert os.path.exists(os.path.join(root, 'val', 'val_annotations.txt')), 'Test labels is not found.'
    test_label_file = open(os.path.join(root, 'val', 'val_annotations.txt'), 'r')
    test_label = {}
    for label_txt in test_label_file.readlines():
        line = label_txt.split()
        test_label[line[0]] = tiny_image_labels_to_number[line[1]]
    assert os.path.exists(os.path.join(root, 'val', 'images')), 'Test dir is not found.'
    for img in os.listdir(os.path.join(root, 'val', 'images')):
        if os.path.splitext(img)[-1] in supported:
            test_images_path.append(os.path.join(root, 'val', 'images', img))
            test_images_label.append(test_label[img])
    # for i in range(0, 10):
    #     target = int(random.uniform(0, 10000))
    #     img = Image.open(test_images_path[target])
    #     img.show()
    #     print(test_images_path[target], test_images_label[target], tiny_image_number_to_labels[test_images_label[target]])
    return train_images_path, train_images_label, test_images_path, test_images_label


preprocess_data('/Users/huanghongyan/Documents/LeNet/Pytorch/dataset/tiny-imagenet-200')
