import json
import argparse
import os


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-path', type=str, default=r'C:\Other\FaceDataset\Face\train.json')
    parser.add_argument('--picture-folder-path', type=str, default=r'C:\Other\FaceDataset\Face\train2017\train2017\train2017')
    parser.add_argument('--save-path', type=str, default=r'C:\Other\FaceDataset\Face\all')
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    json_path = args.json_path
    picture_folder_path = args.picture_folder_path
    save_path = args.save_path
    with open(json_path, 'r') as f:
        json_info = json.load(f)
    json_annotations = json_info.get('annotations')
    # bbox [xmin, ymin, w, h](絕對座標), image_id, category_id
    json_images = json_info.get('images')
    # file_name, id, width, height
    image_id_to_image_name = dict()
    image_width_height = dict()
    for image_info in json_images:
        image_name = image_info.get('file_name', None)
        assert image_name is not None, 'image_name'
        image_id = image_info.get('id', None)
        assert image_id is not None, 'image_id'
        image_width = image_info.get('width', None)
        assert image_width is not None, 'image_width'
        image_height = image_info.get('height', None)
        assert image_height is not None, 'image_height'
        image_id_to_image_name[image_id] = image_name
        image_width_height[image_id] = (image_width, image_height)
    image_annotations = dict()
    for annotation in json_annotations:
        bbox = annotation.get('bbox', None)
        assert bbox is not None, 'bbox'
        image_id = annotation.get('image_id', None)
        assert image_id is not None, 'image_id'
        category_id = annotation.get('category_id', None)
        assert category_id is not None, 'category_id'
        image_name = image_id_to_image_name[image_id]
        image_width, image_height = image_width_height[image_id]

        xmin, ymin, w, h = bbox
        xmax, ymax = xmin + w, ymin + h
        # import cv2
        # from PIL import Image
        # image = cv2.imread(os.path.join(picture_folder_path, image_name))
        # cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # image.show()

        center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
        center_x, center_y = center_x / image_width, center_y / image_height
        w, h = w / image_width, h / image_height
        if image_name in image_annotations.keys():
            image_annotations[image_name].append((category_id, center_x, center_y, w, h))
        else:
            image_annotations[image_name] = [(category_id, center_x, center_y, w, h)]

    mark_image = dict()
    for image_name, annotations in image_annotations.items():
        txt_path = os.path.join(save_path, os.path.splitext(image_name)[0] + '.txt')
        mark_image[image_name] = 1
        with open(txt_path, 'w') as f:
            for annotation in annotations:
                info = ''
                for anno in annotation:
                    info += str(anno) + ' '
                info = info[:-1]
                f.write(info)
                f.write('\n')

    for image_id, image_name in image_id_to_image_name.items():
        if image_name not in mark_image.keys():
            print(f'Image name: {image_name}沒有對應的標註框')
            txt_path = os.path.join(save_path, os.path.splitext(image_name)[0] + '.txt')
            with open(txt_path, 'w') as f:
                pass


if __name__ == '__main__':
    main()
