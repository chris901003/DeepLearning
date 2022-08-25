import torch
import mediapipe as mp
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
import math


class Compose:
    def __init__(self, cfg):
        support_transform = {
            'ReadImage': ReadImage,
            'ToTensor': ToTensor,
            'Collect': Collect
        }
        self.transform = list()
        for operation in cfg:
            transform_type = operation.pop('type', None)
            assert transform_type is not None, '圖像處理流當中一定需要type進行指定'
            if transform_type not in support_transform:
                raise NotImplemented
            cls = support_transform[transform_type]
            create_operation = cls(**operation)
            self.transform.append(create_operation)

    def __call__(self, data):
        for operation in self.transform:
            data = operation(data)
        return data


class ReadImage:
    def __init__(self, *args):
        pass

    def __call__(self, data):
        img_path = data.get('img_path', None)
        assert img_path is not None, 'Data當中缺少img_path資訊'
        if not os.path.exists(img_path):
            raise FileNotFoundError(f'Picture path : {img_path} not found.')
        img = cv2.imread(img_path)
        data['img'] = img
        return data


class ToTensor:
    def __init__(self, targets, to_rgb, img_transpose=True):
        self.targets = targets
        self.to_rgb = to_rgb
        self.img_transpose = img_transpose

    def __call__(self, data):
        if self.to_rgb:
            img = data['img']
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.img_transpose:
                img = img.transpose(2, 0, 1)
            data['img'] = img
        for target in self.targets:
            info = data.get(target, None)
            assert info is not None, f'Data當中缺少{target}資訊'
            info = torch.from_numpy(info)
            data[target] = info
        return data


class Collect:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        collate_data = dict()
        for target in self.keys:
            info = data.get(target, None)
            assert info is not None, f'Data當中缺少{target}資訊'
            collate_data[target] = info
        return collate_data


class PoseDataset(Dataset):
    def __init__(self, file_root, cfg):
        if not os.path.exists(file_root):
            raise FileNotFoundError(f'File {file_root} not found.')
        support_img_format = ['.jpg', '.png', '.jpeg']
        self.imgs = list()
        for file_name in os.listdir(file_root):
            if not os.path.isdir(os.path.join(file_root, file_name)):
                continue
            file_dir = os.path.join(file_root, file_name)
            for img_name in os.listdir(file_dir):
                if os.path.splitext(img_name)[1] in support_img_format:
                    data = {
                        'img_name': img_name,
                        'img_path': os.path.join(file_dir, img_name),
                        'class': np.fromstring(file_name, dtype=int, sep=' ')
                    }
                    self.imgs.append(data)
        self.compose = Compose(cfg)

    def __getitem__(self, index):
        data = self.imgs[index]
        data = self.compose(data)
        return data

    def __len__(self):
        return len(self.imgs)


def custom_collate_fn(batch):
    imgs = list()
    cls = list()
    for data in batch:
        for key, value in data.items():
            if key == 'img':
                imgs.append(value)
            elif key == 'class':
                cls.append(value - 1)
    # imgs = torch.stack(imgs)
    # cls = torch.stack(cls)
    return imgs, cls


def CreateDataLoader(dataset, collate_fn):
    dataloader_cfg = {
        'dataset': dataset,
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 1,
        'pin_memory': True,
        'drop_last': False,
        'collate_fn': collate_fn
    }
    dataloader = DataLoader(**dataloader_cfg)
    return dataloader


class PoseDetection(nn.Module):
    def __init__(self, num_joints, num_classes):
        super(PoseDetection, self).__init__()
        self.fc1 = nn.Linear(num_joints * num_joints, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 128)
        self.ln3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.ln2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.ln3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.relu(out)
        return out


def CreateModel(num_joints, num_classes):
    return PoseDetection(num_joints, num_classes)


def GetDistance(landmark, scale, idx1, idx2):
    dis_x = max(landmark[idx1].x, landmark[idx2].x) - min(landmark[idx1].x, landmark[idx2].x)
    dis_y = max(landmark[idx1].y, landmark[idx2].y) - min(landmark[idx1].y, landmark[idx2].y)
    dis = math.sqrt(dis_x * dis_x + dis_y * dis_y)
    if scale != -1:
        dis = dis * scale
    return dis


def run(prev_model, pose_model, device, train_dataloader, train_epoch, loss_function, optimizer
        , eval_epoch=None, eval_dataloader=None):
    if eval_epoch is not None and eval_dataloader is None:
        assert False, '如果需要驗證就需要有驗證的資料'
    for epoch in range(train_epoch):
        train_one_epoch(prev_model, pose_model, train_dataloader, device, loss_function, optimizer)
        if eval_epoch is not None:
            if (epoch + 1) % eval_epoch == 0:
                eval_one_epoch(prev_model, pose_model, eval_dataloader, device)
    print('Finish training.')


def train_one_epoch(prev_model, pose_model, dataloader, device, loss_function, optimizer):
    pose_model.train()
    total_loss = 0
    correct = 0
    cnt = 0
    for batch_index, (imgs, cls) in enumerate(dataloader):
        keypoint_results = list()
        cls_results = np.array([])
        for idx, img in enumerate(imgs):
            keypoint_result = prev_model.process(img)
            if keypoint_result.multi_hand_landmarks:
                cnt += 1
                cls_results = np.append(cls_results, cls[idx])
                for handLms in keypoint_result.multi_hand_landmarks:
                    num_points = len(handLms.landmark)
                    dis_5_17 = GetDistance(handLms.landmark, -1, 5, 17)
                    dis_scale = 1 / dis_5_17
                    all_distance = [[GetDistance(handLms.landmark, dis_scale, i, j)
                                     for j in range(num_points)] for i in range(num_points)]
                    all_distance = torch.Tensor(all_distance)
                    keypoint_results.append(all_distance)
        keypoint_results = torch.stack(keypoint_results)
        optimizer.zero_grad()
        results = pose_model(keypoint_results)
        cls = torch.from_numpy(cls_results)
        cls = cls.type(torch.LongTensor)
        loss = loss_function(results, cls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predict = torch.softmax(results, dim=1)
        predict = torch.argmax(predict, dim=1)
        correct += torch.eq(predict, cls).sum().item()
    print("==========================")
    print(f'Accuracy {round(correct / cnt * 100, 2)}')
    print("==========================")


def main():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cfg = [{'type': 'ReadImage'},
           {'type': 'ToTensor', 'targets': [], 'to_rgb': True, 'img_transpose': False},
           {'type': 'Collect', 'keys': ['img', 'class']}]
    pose_dataset = PoseDataset('/Users/huanghongyan/Documents/DeepLearning/Test/PoseData', cfg)
    pose_dataloader = CreateDataLoader(pose_dataset, custom_collate_fn)
    next(iter(pose_dataloader))

    num_joints = 21
    num_classes = 5
    pose_model = CreateModel(num_joints, num_classes)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    pose_model = pose_model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(pose_model.parameters(), lr=0.01, momentum=0.9)

    train_epoch = 30
    run(hands, pose_model, device, pose_dataloader, train_epoch, loss_function, optimizer)


if __name__ == '__main__':
    main()
    print("==========================")
    print('Finish!')
    print("==========================")
