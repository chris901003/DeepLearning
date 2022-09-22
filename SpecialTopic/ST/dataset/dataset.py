import numpy as np
import torch
from random import sample, shuffle
from torch.utils.data.dataset import Dataset
from .utils import preprocess_input, Compose


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, pipeline_cfg, mosaic=True, train=True):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.mosaic = mosaic
        self.epoch_now = -1
        self.train = train
        self.pipelines = Compose(pipeline_cfg)

    def __getitem__(self, index):
        index = index % self.length
        data = dict(annotation_lines=[self.annotation_lines[index]])
        if self.mosaic:
            lines = sample(self.annotation_lines, 3)
            for line in lines:
                data['annotation_lines'].append(line)
            shuffle(data['annotation_lines'])
        data = self.pipelines(data)
        image = data['image']
        if isinstance(image, list):
            assert len(image) == 1, '圖像資料錯誤'
            image = image[0]
        box = data['bboxes']
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        if isinstance(box, list):
            box = np.concatenate(box, axis=0)
        else:
            box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        if self.train:
            return image, box
        data['image'] = image
        data['bboxes'] = box
        return data

    def __len__(self):
        return self.length

    @staticmethod
    def custom_collate_fn(batch):
        images = list()
        bboxes = list()
        for img, box in batch:
            images.append(img)
            bboxes.append(box)
        images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
        bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
        return images, bboxes

    @staticmethod
    def custom_collate_fn_val(batch):
        image = [batch[0]['image']]
        bboxes = [batch[0]['bboxes']]
        bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
        image = torch.from_numpy(np.array(image)).type(torch.FloatTensor)
        return image, bboxes, batch[0]['ori_size'], batch[0]['keep_ratio'], batch[0]['images_path'][0]
