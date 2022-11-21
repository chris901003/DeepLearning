import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
from torchvision import transforms


class Cifar100Dataset(Dataset):
    def __init__(self, annotation_path, mean='Default', std='Default'):
        if mean == 'Default':
            mean = [0.485, 0.456, 0.406]
        if std == 'Default':
            std = [0.229, 0.224, 0.225]
        self.transform_data = transforms.Compose([
            # transforms.Resize(224),
            # transforms.RandomRotation(90),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.annotation_path = annotation_path
        self.data_info = self.parse_annotation()

    def __getitem__(self, idx):
        result = self.data_info[idx]
        image_path = result.get('image_path')
        image = Image.open(image_path)
        if np.random.rand() < 0.5:
            image = image.rotate(90)
        image = self.transform_data(image)
        label = result.get('label')
        image_name = result.get('image_name')
        result = {'image': image, 'label': label, 'image_name': image_name}
        return result

    def __len__(self):
        return len(self.data_info)

    def parse_annotation(self):
        results = list()
        with open(self.annotation_path, 'r') as f:
            annotations = f.readlines()
        for annotation in annotations:
            image_path, label, image_name = annotation.strip().split(' ')
            data = dict(image_path=image_path, label=int(label), image_name=image_name)
            results.append(data)
        return results

    @staticmethod
    def collate_fn(batch):
        images, labels, images_name = list(), list(), list()
        for info in batch:
            images.append(info['image'])
            labels.append(info['label'])
            images_name.append(info['image_name'])
        images = torch.stack(images, dim=0)
        labels = torch.Tensor(labels).long()
        return images, labels, images_name
