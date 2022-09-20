import cv2
from PIL import Image
import os


def main():
    # 圖像位置
    images_path_root = '/Users/huanghongyan/Downloads/data_annotation/img/generate_img/imgs'
    # 標註檔案位置
    annotations_path_root = '/Users/huanghongyan/Downloads/data_annotation/img/generate_img/annotations'
    # 結果保存位置
    save_path = '/Users/huanghongyan/Downloads/data_annotation/img/generate_img/check'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
    images_name = [image_name for image_name in os.listdir(images_path_root)
                   if os.path.splitext(image_name)[1] in support_image_format]
    annotations_name = [annotation_name for annotation_name in os.listdir(annotations_path_root)
                        if os.path.splitext(annotation_name)[1] == '.txt']
    images_name = sorted(images_name)
    annotations_name = sorted(annotations_name)
    for index, (image_name, annotation_name) in enumerate(zip(images_name, annotations_name)):
        image_path = os.path.join(images_path_root, image_name)
        annotation_path = os.path.join(annotations_path_root, annotation_name)
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]
        with open(annotation_path, 'r') as f:
            annotations = f.readlines()
        for annotation in annotations:
            label, center_x, center_y, width, height = annotation.strip().split(' ')
            center_x = float(center_x) * img_width
            center_y = float(center_y) * img_height
            width = float(width) * img_width
            height = float(height) * img_height
            xmin = int(max(0, center_x - width / 2))
            ymin = int(max(0, center_y - height / 2))
            xmax = int(min(center_x + width / 2, img_width))
            ymax = int(min(center_y + height / 2, img_height))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 90, 44), 5, cv2.LINE_AA)
        cv2.putText(image, str(len(annotations)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 90, 44), 5, cv2.LINE_AA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image.save(os.path.join(save_path, f'{index}.jpg'))


if __name__ == '__main__':
    main()