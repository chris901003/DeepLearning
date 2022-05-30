from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class VOC2012DataSet(Dataset):
    def __init__(self, voc_root, transforms, train_set=True):
        self.root = os.path.join(voc_root, 'VOCdevkit', 'VOC2012')
        self.img_root = os.path.join(self.root, 'JPEGImages')
        self.annotations_root= os.path.join(self.root, 'Annotations')

        if train_set:
            txt_list = os.path.join(self.root, 'ImageSets', 'Main', 'train.txt')
        else:
            txt_list = os.path.join(self.root, 'ImageSets', 'Main', 'val.txt')
        with open(txt_list) as read:
            self.xml_list = [os.path.join(self.annotations_root, line.strip() + '.xml')
                             for line in read.readlines()]

        try:
            json_file = open('./pascal_voc_classes.json', 'r')
            self.class_dict = json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        img_path = os.path.join(self.img_root, data['filename'])
        image= Image.open(img_path)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG!')
        # iscrowd 這個是看有沒有重疊object算是VOC特有的
        boxes, labels, iscrowd = [], [], []
        for obj in data['object']:
            x_min = float(obj['bndbox']['xmin'])
            x_max = float(obj['bndbox']['xmax'])
            y_min = float(obj['bndbox']['ymin'])
            y_max = float(obj['bndbox']['ymax'])
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(self.class_dict[obj['name']])
            iscrowd.append(int(obj['difficult']))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target = {}
        target['boxes'] = boxes
        target['labels']= labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrows'] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def get_height_and_width(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        data_height = int(data['size']['height'])
        data_width = int(data['size']['width'])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            # 一張圖片中可能有多個object要偵測，所以我們用list保存，裡面的每個是這個object的詳細內容
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

# import transforms
# from draw_box_utils import draw_objs
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random
# import numpy as np
#
# # read class_indict
# category_index = {}
# try:
#     json_file = open('./pascal_voc_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {str(v): str(k) for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)
#
# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
#
# # load train data set
# train_data_set = VOC2012DataSet(os.getcwd(), data_transform["train"], True)
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     img = ts.ToPILImage()(img)
#     plot_img = draw_objs(img,
#                          target["boxes"].numpy(),
#                          target["labels"].numpy(),
#                          np.ones(target["labels"].shape[0]),
#                          category_index=category_index,
#                          box_thresh=0.5,
#                          line_thickness=3,
#                          font='arial.ttf',
#                          font_size=20)
#     plt.imshow(plot_img)
#     plt.show()
