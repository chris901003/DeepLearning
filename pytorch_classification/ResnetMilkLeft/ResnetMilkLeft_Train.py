import os
from torchvision import models
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
from tqdm import tqdm


def get_cls_from_cfg(support, cfg):
    cls_type = cfg.pop('type', None)
    assert cls_type is not None, '在設定檔當中沒有指定type'
    assert cls_type in support, f'指定的{cls_type}尚未支援'
    cls = support[cls_type]
    return cls


class Compose:
    def __init__(self, cfg):
        support_operation = {
            'LoadImageFromFile': LoadImageFromFile,
            'RandomFlip': RandomFlip,
            'Resize': Resize,
            'Normalize': Normalize,
            'ToTensor': ToTensor,
            'Collect': Collect
        }
        self.pipeline = list()
        for pipeline_cfg in cfg:
            operation_cls = get_cls_from_cfg(support_operation, pipeline_cfg)
            operation = operation_cls(**pipeline_cfg)
            self.pipeline.append(operation)

    def __call__(self, data):
        for operation in self.pipeline:
            data = operation(data)
        return data


class LoadImageFromFile:
    def __init__(self, key, img_style):
        self.key = key
        self.img_style = img_style

    def __call__(self, data):
        img_path = data.get(self.key, None)
        assert img_path is not None, f'在data當中沒有 {self.key} 資料'
        assert os.path.isfile(img_path), f'圖像 {img_path} 不存在'
        img = cv2.imread(img_path)
        if self.img_style == 'Gray':
            img = img[..., None]
        if self.img_style == 'RGB':
            assert img.shape[2] == 3, '輸入圖像非RGB圖像'
        data[self.key] = img
        return data


class RandomFlip:
    def __init__(self, flip_ratio, flip_direction, keys):
        self.flip_ratio = flip_ratio
        self.flip_direction = flip_direction
        self.keys = keys

    def __call__(self, data):
        random_flip = np.random.uniform()
        flip = True if random_flip > self.flip_ratio else False
        data['flip'] = flip
        if flip:
            for info_name in self.keys:
                img = data.get(info_name, None)
                assert img is not None, f'在data當中沒有找到 {info_name} 資料'
                if 'horizon' in self.flip_direction:
                    img = cv2.flip(img, 1)
                if 'vertical' in self.flip_direction:
                    img = cv2.flip(img, 0)
                data[info_name] = img
        return data


class Resize:
    def __init__(self, keep_ratio, keys, scale):
        self.keep_ratio = keep_ratio
        self.keys = keys
        if isinstance(scale, int):
            scale = (scale, scale)
        self.scale = scale

    def __call__(self, data):
        for info_name in self.keys:
            img = data.get(info_name, None)
            assert img is not None, f'在data當中沒有找到 {info_name} 資料'
            if not self.keep_ratio:
                img = cv2.resize(img, self.scale, interpolation=cv2.INTER_LINEAR)
            else:
                raise NotImplementedError('尚未實作保持高寬比的resize')
            data[info_name] = img
        return data


class Normalize:
    def __init__(self, mean, std, keys, to_rgb=True, transpose=True):
        self.norm = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
        self.keys = keys
        self.to_rgb = to_rgb
        self.transpose = transpose

    def __call__(self, data):
        for info_name in self.keys:
            img = data.get(info_name, None)
            assert img is not None, f'在data當中沒有找到 {info_name} 資料'
            img = (img - self.norm) / self.std
            if self.to_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transpose:
                img = img.transpose(2, 0, 1)
            data[info_name] = img
        return data


class ToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for info_name in self.keys:
            info = data.get(info_name, None)
            assert info is not None, f'在data當中沒有找到 {info_name} 資料'
            if type(info) is np.ndarray:
                info = torch.from_numpy(info)
            elif isinstance(info, int):
                info = torch.LongTensor([info])
            else:
                raise NotImplementedError('該變數類別未實作')
            data[info_name] = info
        return data


class Collect:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        results = dict()
        for info_name in self.keys:
            info = data.get(info_name, None)
            assert info is not None, f'在data當中沒有找到 {info_name} 資料'
            results[info_name] = info
        return results


class MilkDataset(Dataset):
    def __init__(self, data_path, dataset_cfg):
        assert os.path.exists(data_path), f'給定的 {data_path} 不存在'
        assert os.path.isdir(data_path), '給定的資料集需要是資料夾格式'
        support_img_format = ['.jpg', '.JPG', '.jpeg', '.JPEG']
        self.data_info = list()
        for cls_number in os.listdir(data_path):
            current_path = os.path.join(data_path, cls_number)
            if not os.path.isdir(current_path):
                continue
            for img_name in os.listdir(current_path):
                img_path = os.path.join(current_path, img_name)
                if os.path.splitext(img_path)[1] in support_img_format:
                    data = {
                        'img': img_path,
                        'label': int(cls_number)
                    }
                    self.data_info.append(data)
        self.pipeline = Compose(dataset_cfg)

    def __getitem__(self, idx):
        data = self.data_info[idx]
        data = self.pipeline(data)
        return data

    def __len__(self):
        return len(self.data_info)


def CreateDataloader(cfg):
    dataloader = DataLoader(**cfg)
    return dataloader


def custom_collate_fn(batch):
    imgs = list()
    labels = list()
    for info in batch:
        imgs.append(info['img'])
        labels.append(info['label'])
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    labels = labels.squeeze(dim=-1)
    return imgs, labels


def run(model, train_epoch, train_dataloader, device, optimizer, loss_function, val_epoch=None, val_dataloader=None):
    if val_epoch is not None:
        assert val_dataloader is not None, '啟用驗證模式就需要給定驗證的Dataloader'
    best_loss = 10000
    for epoch in range(1, train_epoch + 1):
        loss = train_one_epoch(model, train_dataloader, device, epoch, optimizer, loss_function, best_loss)
        best_loss = min(loss, best_loss)
        if val_epoch is not None:
            if epoch % val_epoch == 0:
                val_one_epoch(model, val_dataloader, device, epoch)


def train_one_epoch(model, dataloader, device, epoch, optimizer, loss_function, best_loss):
    model.train()
    total_loss = 0
    correct = 0
    picture = 0
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch}: ', postfix=f'Correct {correct}', mininterval=1) as pbar:
        for imgs, labels in dataloader:
            picture += imgs.shape[0]
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += torch.eq(pred, labels).sum().item()
            pbar.set_postfix_str(f'Accuracy => {round(correct / picture * 100, 2)}, Picture => {picture}')
            pbar.update(1)
    if total_loss < best_loss:
        torch.save(model.state_dict(), 'best_model.pkt')
    return total_loss


def val_one_epoch(model, dataloader, device, epoch):
    pass


def main():
    train_dataset_cfg = [
        {'type': 'LoadImageFromFile', 'img_style': 'RGB', 'key': 'img'},
        {'type': 'RandomFlip', 'flip_ratio': 0.5, 'flip_direction': ['horizon'], 'keys': ['img']},
        {'type': 'Resize', 'keep_ratio': False, 'keys': ['img'], 'scale': (256, 256)},
        {'type': 'Normalize', 'mean': [127.5] * 3, 'std': [127.5] * 3, 'keys': ['img']},
        {'type': 'ToTensor', 'keys': ['img', 'label']},
        {'type': 'Collect', 'keys': ['img', 'label']}
    ]
    train_data_path = './data/Train'
    train_dataset = MilkDataset(train_data_path, train_dataset_cfg)
    dataloader_cfg = {
        'dataset': train_dataset,
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 1,
        'pin_memory': True,
        'drop_last': False,
        'collate_fn': custom_collate_fn
    }
    train_dataloader = CreateDataloader(dataloader_cfg)
    num_classes = 4
    model = models.resnet34(pretrained=True)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(in_features=fc_inputs, out_features=num_classes)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_function = nn.CrossEntropyLoss()
    train_epoch = 100
    run(model, train_epoch, train_dataloader, device, optimizer, loss_function)


if __name__ == '__main__':
    main()
