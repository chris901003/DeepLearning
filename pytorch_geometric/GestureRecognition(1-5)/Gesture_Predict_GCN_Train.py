import torch
import mediapipe as mp
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
from torch_geometric.nn import GCNConv


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
    return imgs, cls


def CreateDataLoader(dataset, collate_fn):
    dataloader_cfg = {
        'dataset': dataset,
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': True,
        'drop_last': False,
        'collate_fn': collate_fn
    }
    dataloader = DataLoader(**dataloader_cfg)
    return dataloader


def CreateModel(cfg):
    support_model = {
        'GestureRecognition': GestureRecognition
    }
    model_type = cfg.pop('type', None)
    assert model_type is not None, "模型cfg當中需要type指定使用模型"
    if model_type not in support_model:
        assert False, "該模型沒有實作對象"
    model_cls = support_model[model_type]
    model = model_cls(**cfg)
    return model


class GestureRecognition(nn.Module):
    def __init__(self, num_joints, num_classes, keypoint_line=None, bilateral=None):
        super(GestureRecognition, self).__init__()
        self.conv1 = GCNConv(num_joints * 2, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        if keypoint_line is not None:
            if bilateral:
                keypoint_line_bilateral = [[p[1], p[0]]for p in keypoint_line]
                keypoint_line.extend(keypoint_line_bilateral)
            self.keypoint_line = keypoint_line
            if not torch.is_tensor(self.keypoint_line):
                self.keypoint_line = torch.LongTensor(self.keypoint_line)
            if self.keypoint_line.shape[0] != 2:
                self.keypoint_line = self.keypoint_line.t()

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = self.keypoint_line
        assert edge_index is not None, "使用GCNConv需要提供連線資訊"
        out = self.conv1(x, edge_index)
        out = self.relu(out)
        out = self.conv2(out, edge_index)
        out = self.relu(out)
        out = self.fc(out)
        return out


def get_keypoint_feature(keypoints_position, idx1, idx2):
    dif_x = keypoints_position[idx2].x - keypoints_position[idx1].x
    dif_y = keypoints_position[idx2].y - keypoints_position[idx1].y
    return torch.tensor([dif_x, dif_y])


def run(prev_model, pose_model, device, train_epoch, train_dataloader, loss_function, optimizer
        , eval_epoch=None, eval_dataloader=None):
    if eval_epoch is not None and eval_dataloader is None:
        assert False, '如果需要驗證就需要有驗證的資料'
    best_accuracy = 0
    for epoch in range(train_epoch):
        acc = train_one_epoch(prev_model, pose_model, train_dataloader, device, loss_function, optimizer, best_accuracy)
        best_accuracy = max(acc, best_accuracy)
        if eval_epoch is not None:
            if (epoch + 1) % eval_epoch == 0:
                eval_one_epoch(prev_model, pose_model, eval_dataloader, device)
    print('Finish training.')


def train_one_epoch(prev_model, pose_model, train_dataloader, device, loss_function, optimizer, best_accuracy):
    pose_model.train()
    total_loss = 0
    correct = 0
    legal_picture = 0
    for batch_index, (imgs, labels) in enumerate(train_dataloader):
        keypoint_features = list()
        labels_result = np.array([])
        for idx, img in enumerate(imgs):
            keypoint_result = prev_model.process(img)
            if keypoint_result.multi_hand_landmarks:
                legal_picture += 1
                labels_result = np.append(labels_result, labels[idx])
                for handLms in keypoint_result.multi_hand_landmarks:
                    num_points = len(handLms.landmark)
                    keypoint_feature = [torch.stack([get_keypoint_feature(handLms.landmark, i, j)
                                                     for i in range(num_points)]) for j in range(num_points)]
                    keypoint_feature = torch.stack(keypoint_feature)
                    keypoint_features.append(keypoint_feature)
        keypoint_features = torch.stack(keypoint_features)
        batch_size, num_joints, _, _ = keypoint_features.shape
        keypoint_features = keypoint_features.reshape(batch_size, num_joints, -1)
        labels = torch.from_numpy(labels_result)
        labels = labels.type(torch.LongTensor)
        optimizer.zero_grad()
        predict = pose_model(keypoint_features)
        results = predict.sum(dim=1)
        loss = loss_function(results, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predict = torch.softmax(results, dim=1)
        predict = torch.argmax(predict, dim=1)
        correct += torch.eq(predict, labels).sum().item()
    accuracy = 0
    print("==========================")
    if legal_picture != 0:
        accuracy = round(correct / legal_picture * 100, 2)
        print(f'Accuracy {accuracy}')
        if accuracy > best_accuracy:
            torch.save(pose_model.state_dict(), 'best_model.pkt')
    else:
        print('No legal picture')
    print("==========================")
    return accuracy


def eval_one_epoch(prev_model, pose_model, eval_dataloader, device):
    pose_model.eval()
    correct = 0
    legal_picture = 0
    with torch.no_grad():
        for batch_index, (imgs, labels) in enumerate(eval_dataloader):
            keypoint_features = list()
            labels_result = np.array([])
            for idx, img in enumerate(imgs):
                keypoint_result = prev_model.process(img)
                if keypoint_result.multi_hand_landmarks:
                    legal_picture += 1
                    labels_result = np.append(labels_result, labels[idx])
                    for handLms in keypoint_result.multi_hand_landmarks:
                        num_points = len(handLms.landmark)
                        keypoint_feature = [torch.stack([get_keypoint_feature(handLms.landmark, i, j)
                                                         for i in range(num_points)]) for j in range(num_points)]
                        keypoint_feature = torch.stack(keypoint_feature)
                        keypoint_features.append(keypoint_feature)
            keypoint_features = torch.stack(keypoint_features)
            batch_size, num_joints, _, _ = keypoint_features.shape
            keypoint_features = keypoint_features.reshape(batch_size, num_joints, -1)
            labels = torch.from_numpy(labels_result)
            labels = labels.type(torch.LongTensor)
            predict = pose_model(keypoint_features)
            results = predict.sum(dim=1)
            predict = torch.softmax(results, dim=1)
            predict = torch.argmax(predict, dim=1)
            correct += torch.eq(predict, labels).sum().item()
    print("==========================")
    if legal_picture != 0:
        accuracy = round(correct / legal_picture * 100, 2)
        print(f'Accuracy of validate {accuracy}')
    else:
        print('No legal picture')
    print("==========================")


def main():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cfg = [{'type': 'ReadImage'},
           {'type': 'ToTensor', 'targets': [], 'to_rgb': True, 'img_transpose': False},
           {'type': 'Collect', 'keys': ['img', 'class']}]
    data_path = '/Users/huanghongyan/Documents/DeepLearning/pytorch_geometric/GestureRecognition(1-5)/PoseData'
    pose_dataset = PoseDataset(data_path, cfg)
    pose_dataloader = CreateDataLoader(pose_dataset, custom_collate_fn)
    model_cfg = {
        'type': 'GestureRecognition',
        'num_joints': 21,
        'num_classes': 5,
        'keypoint_line': [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
                          [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20],
                          [0, 17]],
        'bilateral': True
    }
    # v1, v2 = next(iter(pose_dataloader))
    gesture_model = CreateModel(model_cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gesture_model = gesture_model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(gesture_model.parameters(), lr=0.001, momentum=0.9)

    train_epoch = 300
    run(hands, gesture_model, device, train_epoch, pose_dataloader, loss_function, optimizer)


if __name__ == '__main__':
    main()
    print("==========================")
    print('Finish!')
    print("==========================")
