import os
import numpy as np
import cv2
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    def __init__(self, annotation_path, data_prefix, norm='Default', std='Default'):
        assert os.path.isfile(annotation_path), 'Given annotation is not file format'
        self.annotation_path = annotation_path
        self.data_prefix = data_prefix
        self.data_info = self.parse_annotation()
        if norm == 'Default':
            self.norm = np.array([0.1307])
        else:
            self.norm = np.array([norm])
        if std == 'Default':
            self.std = np.array([0.3081])
        else:
            self.std = np.array([std])

    def __getitem__(self, index):
        results = self.data_info[index]
        image_path = results.get('image_path')
        image = cv2.imread(image_path)
        if image.ndim == 3:
            image = image[:, :, 0]
        # image = image / 255.0
        # image = image - self.norm[None, None, -1]
        # image = image / self.std[None, None, -1]
        label = results.get('label')
        results = {'image': image, 'label': label}
        return results

    def __len__(self):
        return len(self.data_info)

    def parse_annotation(self):
        results = list()
        with open(self.annotation_path) as f:
            annotations = f.readlines()
        for annotation in annotations:
            image_name, label = annotation.split(' ')
            if self.data_prefix != '':
                image_path = os.path.join(self.data_prefix, image_name + '.png')
            else:
                image_path = image_name
            data = dict(image_path=image_path, label=int(label))
            results.append(data)
        return results

    @staticmethod
    def collate_fn(batch):
        images, labels = list(), list()
        for info in batch:
            images.append(info['image'])
            labels.append(info['label'])
        images = np.array(images).squeeze(0)
        labels = np.array(labels)
        return images, labels
