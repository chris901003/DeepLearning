import numpy as np
import torch
from random import sample, shuffle
import copy
import os
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
        box = [bx if bx.ndim == 2 else np.ndarray((0, 5)) for bx in box]
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


class VideoDataset(Dataset):
    def __init__(self, ann_file, pipeline, data_prefix=None, test_mode=False, start_index=0, num_class=None, modality='RGB'):
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.test_mode = test_mode
        self.num_class = num_class
        self.start_index = start_index
        self.modality = modality
        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()

    def load_annotations(self):
        assert self.ann_file.endswith('.txt'), '需要是.txt的標註文件'
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                filename, label = line_split
                label = int(label)
                if self.data_prefix is not None:
                    filename = os.path.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label))
        return video_infos

    def prepare_train_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        return self.pipeline(results)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_frames(idx)
        return self.prepare_train_frames(idx)

    def __len__(self):
        return len(self.video_infos)

    @staticmethod
    def custom_collate_fn(batch):
        imgs, labels = list(), list()
        for info in batch:
            imgs.append(info['imgs'])
            labels.append(info['label'])
        imgs = torch.stack(imgs)
        labels = torch.stack(labels)
        return imgs, labels


class RemainingDataset(Dataset):
    def __init__(self, annotation_file, data_prefix, pipeline_cfg):
        self.annotation_file = annotation_file
        self.data_prefix = data_prefix
        self.pipeline_cfg = pipeline_cfg
        self.data_info = self.load_annotation()
        self.pipeline = Compose(pipeline_cfg)

    def load_annotation(self):
        results = list()
        support_image_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
        with open(self.annotation_file, 'r') as f:
            annotations = f.readlines()
        for annotation in annotations:
            image_path, label = annotation.split(' ')
            if self.data_prefix != '':
                image_path = os.path.join(self.data_prefix, image_path)
            if os.path.splitext(image_path)[1] not in support_image_format:
                raise ValueError(f'{image_path} 圖像資料檔案格式不支援')
            data = dict(image_path=image_path, label=int(label))
            results.append(data)
        return results

    def __getitem__(self, index):
        data = self.data_info[index]
        data = self.pipeline(data)
        return data

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def train_collate_fn(batch):
        images, labels = list(), list()
        for info in batch:
            image = info['image']
            image = image.transpose((2, 0, 1))
            images.append(torch.from_numpy(image))
            labels.append(torch.LongTensor([info['label']]))
        images = torch.stack(images)
        labels = torch.stack(labels)
        return images, labels
