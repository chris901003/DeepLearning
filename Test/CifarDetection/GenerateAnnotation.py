import os


def main():
    data_path = './Training_data'
    save_name = './train_annotation.txt'
    classes_idx = [class_idx for class_idx in os.listdir(data_path)
                      if os.path.isdir(os.path.join(data_path, class_idx))]
    classes_folder = [os.path.join(data_path, class_idx) for class_idx in classes_idx]
    print('Classes folder')
    for class_folder in classes_folder:
        print(class_folder)
    support_image_format = ['.png']
    annotations = list()
    for class_idx, class_folder in zip(classes_idx, classes_folder):
        images_name = [image_name for image_name in os.listdir(class_folder)
                       if os.path.splitext(image_name)[1] in support_image_format]
        images_path = [os.path.join(class_folder, image_name) for image_name in images_name]
        annotation = [image_path + ' ' + str(class_idx) + ' ' + os.path.splitext(image_name)[0]
                      for image_path, image_name in zip(images_path, images_name)]
        annotations = annotations + annotation
    with open(save_name, 'w') as f:
        for annotation in annotations:
            f.write(annotation)
            f.write('\n')
    print(f'Total data: {len(annotations)}')


if __name__ == '__main__':
    main()
