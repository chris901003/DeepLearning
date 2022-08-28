import os
import random
from torch.utils.data.dataset import Dataset
import av
import mediapipe as mp
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch_geometric.nn import GCNConv


class GestureDataset(Dataset):
    def __init__(self, data_path, cfg):
        assert os.path.exists(data_path), '資料集不存在'
        assert os.path.isdir(data_path), '資料集需要是資料夾形式'
        support_media_type = ['.mp4']
        self.data_info = list()
        for file_name in os.listdir(data_path):
            file_path = os.path.join(data_path, file_name)
            if not os.path.isdir(file_path):
                continue
            for video_name in os.listdir(file_path):
                if os.path.splitext(video_name)[1] in support_media_type:
                    data = {
                        'video_name': video_name,
                        'video_path': os.path.join(file_path, video_name),
                        'label': int(file_name)
                    }
                    self.data_info.append(data)
        self.pipeline = Compose(cfg)

    def __getitem__(self, idx):
        data = self.data_info[idx]
        data = self.pipeline(data)
        return data

    def __len__(self):
        return len(self.data_info)


class Compose:
    def __init__(self, cfg):
        if not isinstance(cfg, list):
            cfg = [cfg]
        support_compose = {
            'ReadVideo': ReadVideo,
            'ExtractFrame': ExtractFrame,
            'DecodeVideo': DecodeVideo,
            'ToTensor': ToTensor,
            'Collect': Collect
        }
        self.pipelines = list()
        for pipeline_cfg in cfg:
            assert isinstance(pipeline_cfg, dict), '處理流中的模塊需要由dict組成'
            pipeline_type = pipeline_cfg.pop('type', None)
            assert pipeline_type is not None, '需要透過type指定模塊類型'
            assert pipeline_type in support_compose, f'{pipeline_type}模塊尚未支持'
            pipeline_cls = support_compose[pipeline_type]
            pipeline = pipeline_cls(**pipeline_cfg)
            self.pipelines.append(pipeline)

    def __call__(self, data):
        for pipeline in self.pipelines:
            data = pipeline(data)
        return data


class ReadVideo:
    def __init__(self):
        pass

    def __call__(self, data):
        video_path = data.get('video_path', None)
        assert video_path is not None, '需要有video_path參數'
        container = av.open(video_path)
        data['video_container'] = container
        data['total_frame'] = container.streams.video[0].frames
        return data


class ExtractFrame:
    def __init__(self, num_clips, clip_len, interval):
        self.num_clips = num_clips
        self.clip_len = clip_len
        self.interval = interval

    def __call__(self, data):
        total_frame = data.get('total_frame', None)
        assert total_frame is not None, '缺少total_frame參數'
        # 獲取至少要從哪幀開始取才不會越界，如果真的不夠就從0
        last_key_frame = max(0, total_frame - 1 - self.clip_len * self.interval)
        frame_index = list()
        for _ in range(self.num_clips):
            random_start_frame = random.randint(0, last_key_frame)
            # 獲取要的index幀
            index = [random_start_frame + i * self.interval for i in range(self.clip_len)]
            # 將超出的部分用最後一幀取代
            index = [idx if idx < total_frame else total_frame - 1 for idx in index]
            frame_index.append(index)
        data['frame_index'] = frame_index
        return data


class DecodeVideo:
    def __init__(self):
        pass

    def __call__(self, data):
        video_container = data.get('video_container', None)
        frame_index = data.get('frame_index', None)
        assert video_container is not None, 'data資料當中缺少video_container資訊'
        assert frame_index is not None, 'data資料當中缺少frame_index資訊'
        assert isinstance(frame_index, list), 'frame_index型態錯誤'
        if not list_of_list(frame_index):
            frame_index = [frame_index]
        last_frame = max([max(F) for F in frame_index])
        imgs_buffer = list()
        for frame in video_container.decode(video=0):
            if len(imgs_buffer) > last_frame + 1:
                break
            imgs_buffer.append(frame.to_rgb().to_ndarray())
        imgs = [[imgs_buffer[frame] for frame in F] for F in frame_index]
        data['imgs'] = imgs
        return data


class ExtractKeypoint:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        mpHands = mp.solutions.hands
        self.hands = mpHands.Hands(max_num_hands=max_num_hands,
                                   min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)

    def __call__(self, imgs):
        assert imgs is not None, 'data當中沒有imgs資訊，無法進行關節點預測'
        assert isinstance(imgs, list), 'imgs型態錯誤'
        if not list_of_list(imgs):
            imgs = [imgs]
        clip_len = len(imgs[0])
        keypoint_results = [[self.hands.process(img) for img in I] for I in imgs]
        keypoint_features = list()
        for clip in keypoint_results:
            keypoint_feature_clip = list()
            for keypoint_result in clip:
                if not keypoint_result.multi_hand_landmarks:
                    continue
                for handLms in keypoint_result.multi_hand_landmarks:
                    num_points = len(handLms.landmark)
                    keypoint_feature = [torch.stack([get_keypoint_feature(handLms.landmark, i, j)
                                                     for i in range(num_points)]) for j in range(num_points)]
                    keypoint_feature = torch.stack(keypoint_feature)
                    keypoint_feature_clip.append(keypoint_feature)
            if len(keypoint_feature_clip) == 0:
                return None
            # assert len(keypoint_feature_clip) > 0, '當前影片為獲取任何關節點資訊'
            while len(keypoint_feature_clip) < clip_len:
                keypoint_feature_clip.append(keypoint_feature_clip[-1])
            keypoint_feature_clip = torch.stack(keypoint_feature_clip)
            keypoint_features.append(keypoint_feature_clip)
        keypoint_features = torch.stack(keypoint_features)
        num_clips, clip_len, num_joint, _, _ = keypoint_features.shape
        keypoint_features = keypoint_features.reshape(num_clips, clip_len, num_joint, -1)
        return keypoint_features


class ToTensor:
    def __init__(self, targets, to_rgb=True, img_transpose=True):
        self.targets = targets
        self.to_rgb = to_rgb
        self.img_transpose = img_transpose

    def __call__(self, data):
        for target in self.targets:
            info = data.get(target, None)
            assert info is not None, f'data當中沒有{info}資訊'
            if isinstance(info, list):
                info = torch.Tensor(info)
            elif isinstance(info, int):
                info = torch.Tensor([info])
            else:
                assert False, '尚未實作'
            data[target] = info
        return data


class Collect:
    def __init__(self, targets):
        self.targets = targets

    def __call__(self, data):
        results = dict()
        for target in self.targets:
            info = data.get(target, None)
            assert info is not None, f'data當中沒有{target}資訊'
            results[target] = info
        return results


def custom_collate_fn(batch):
    imgs = list()
    label = list()
    for info in batch:
        imgs.append(info['imgs'])
        label.append(info['label'])
    label = torch.Tensor(label)
    return imgs, label


def CreateDataloader(cfg):
    dataloader = DataLoader(**cfg)
    return dataloader


def list_of_list(list_of_list_var):
    if not isinstance(list_of_list_var, list):
        return False
    elif not isinstance(list_of_list_var[0], list):
        return False
    return True


def get_keypoint_feature(keypoints_position, idx1, idx2):
    dif_x = keypoints_position[idx2].x - keypoints_position[idx1].x
    dif_y = keypoints_position[idx2].y - keypoints_position[idx1].y
    return torch.tensor([dif_x, dif_y])


def CreateModel(cfg):
    support_model = {
        'TemporalGestureRecognition': TemporalGestureRecognition
    }
    model_type = cfg.pop('type', None)
    assert model_type is not None, '需要指定使用的模型類別'
    if model_type not in support_model:
        assert False, '該模型尚未支持'
    model_cls = support_model[model_type]
    model = model_cls(**cfg)
    return model


class TemporalGestureRecognition(nn.Module):
    def __init__(self, num_joints, num_classes, clip_len, keypoint_line=None, bilateral_without_temporal=True,
                 bilateral_with_temporal=False, preprocess=None, num_stage=None, stage_blocks=None, block=None):
        super(TemporalGestureRecognition, self).__init__()
        support_preprocess = {
            'ExtractKeypoint': ExtractKeypoint
        }
        self.num_joints = num_joints
        self.num_classes = num_classes
        if keypoint_line is not None:
            if bilateral_without_temporal:
                bilateral_keypoint_line = [[p[1], p[0]] for p in keypoint_line]
                keypoint_line.extend(bilateral_keypoint_line)
            self.keypoint_line_without_temporal = torch.LongTensor(keypoint_line).t()
            keypoint_line_with_temporal = [[point + num_joints * i, point + num_joints * (i + 1)]
                                           for point in range(num_joints)
                                           for i in range(clip_len - 1)]
            if bilateral_with_temporal:
                bilateral_keypoint_line = [[p[1], p[0]] for p in keypoint_line_with_temporal]
                keypoint_line_with_temporal.extend(bilateral_keypoint_line)
            keypoint_line_with_temporal.extend(keypoint_line)
            self.keypoint_line_with_temporal = torch.LongTensor(keypoint_line_with_temporal).t()
        if preprocess:
            preprocess_type = preprocess.pop('type', None)
            assert preprocess_type is not None, 'Preprocess需給定type決定使用類'
            if preprocess_type not in support_preprocess:
                assert False, f'Preprocess {preprocess_type} 尚未實作'
            preprocess_cls = support_preprocess[preprocess_type]
            self.preprocess = preprocess_cls(**preprocess)
        assert num_stage and stage_blocks and block, '模型結構不能為None'
        assert len(stage_blocks) == num_stage, 'stage數量與每個stage堆疊的block數量需要相同'
        in_channel = num_joints * 2
        self.conv1 = GCNConv(in_channel, 64)
        # 注意記得需要reshape將channel維度放到第二個維度
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2_T = GCNConv(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.stages = nn.ModuleList()
        in_channel = 128
        for idx in range(num_stage):
            blocks = list()
            for idx_block in range(stage_blocks[idx]):
                if idx_block == 0:
                    downsample = nn.ModuleList([
                        GCNConv(in_channel, in_channel * 2),
                        nn.BatchNorm1d(in_channel * 2)]
                    )
                    current_block = block(
                        in_channel=in_channel,
                        out_channel=in_channel * 2,
                        downsample=downsample,
                        keypoint_line_without_temporal=self.keypoint_line_without_temporal,
                        keypoint_line_with_temporal=self.keypoint_line_with_temporal
                    )
                    in_channel = in_channel * 2
                else:
                    current_block = block(
                        in_channel=in_channel,
                        out_channel=in_channel,
                        keypoint_line_without_temporal=self.keypoint_line_without_temporal,
                        keypoint_line_with_temporal=self.keypoint_line_with_temporal
                    )
                blocks.append(current_block)
            self.stages.append(nn.Sequential(*blocks))
        self.classifier = nn.Linear(in_channel, num_classes)

    def forward(self, x, keypoint_line_without_temporal=None, keypoint_line_with_temporal=None):
        if keypoint_line_without_temporal is None:
            keypoint_line_without_temporal = self.keypoint_line_without_temporal
        if keypoint_line_with_temporal is None:
            keypoint_line_with_temporal = self.keypoint_line_with_temporal
        assert keypoint_line_with_temporal is not None and keypoint_line_without_temporal is not None, \
            '須提供連線方式'
        batch_size, num_joints, channel = x.shape
        out = self.conv1(x, keypoint_line_without_temporal)
        # out = out.reshape(batch_size, -1, num_joints)
        # out = self.bn1(out)
        # out = out.reshape(batch_size, num_joints, -1)
        out = self.relu(out)
        out = self.conv2_T(out, keypoint_line_with_temporal)
        # out = out.reshape(batch_size, -1, num_joints)
        # out = self.bn2(out)
        # out = out.reshape(batch_size, num_joints, -1)
        out = self.relu(out)
        for stage in self.stages:
            out = stage(out)
        out = self.classifier(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=None,
                 keypoint_line_without_temporal=None, keypoint_line_with_temporal=None):
        super(BasicBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.downsample = downsample
        self.keypoint_line_without_temporal = keypoint_line_without_temporal
        self.keypoint_line_with_temporal = keypoint_line_with_temporal
        self.conv1 = GCNConv(in_channel, out_channel)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2_T = GCNConv(out_channel, out_channel)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.conv3 = GCNConv(out_channel, out_channel)
        self.bn3 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x, keypoint_line_without_temporal=None, keypoint_line_with_temporal=None):
        if keypoint_line_without_temporal is None:
            keypoint_line_without_temporal = self.keypoint_line_without_temporal
        if keypoint_line_with_temporal is None:
            keypoint_line_with_temporal = self.keypoint_line_with_temporal
        assert keypoint_line_with_temporal is not None and keypoint_line_without_temporal is not None, \
            '須提供連線方式'
        batch_size, num_joints, channel = x.shape
        residual = x
        if self.downsample:
            residual = self.downsample[0](residual, keypoint_line_with_temporal)
            residual = residual.reshape(batch_size, -1, num_joints)
            residual = self.downsample[1](residual)
            residual = residual.reshape(batch_size, num_joints, -1)
        out = self.conv1(x, keypoint_line_without_temporal)
        # out = out.reshape(batch_size, -1, num_joints)
        # out = self.bn1(out)
        # out = out.reshape(batch_size, num_joints, -1)
        out = self.relu(out)
        out = self.conv2_T(out, keypoint_line_with_temporal)
        # out = out.reshape(batch_size, -1, num_joints)
        # out = self.bn2(out)
        # out = out.reshape(batch_size, num_joints, -1)
        out = self.relu(out)
        out = self.conv3(out, keypoint_line_without_temporal)
        # out = out.reshape(batch_size, -1, num_joints)
        # out = self.bn3(out)
        # out = out.reshape(batch_size, num_joints, -1)
        out = out + residual
        out = self.relu(out)
        return out


def run(model, device, train_epoch, train_dataloader, loss_function, optimizer,
        val_epoch=None, val_dataloader=None):
    if val_epoch is not None:
        assert val_dataloader is not None, '啟用驗證就需要提供驗證的DataLoader'
    best_loss = 10000
    for epoch in range(train_epoch):
        loss = train_one_epoch(model, train_dataloader, device, loss_function, optimizer, best_loss, epoch)
        best_loss = min(loss, best_loss)
        if val_epoch is not None:
            if (epoch + 1) % val_epoch == 0:
                eval_one_epoch(model, val_dataloader, device)
    print('Finish Training')


def train_one_epoch(model, dataloader, device, loss_function, optimizer, best_loss, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total_video = 0
    for idx, (imgs, labels) in enumerate(dataloader):
        labels = labels.type(torch.LongTensor)
        keypoint_features = list()
        legal_label = list()
        for index, img in enumerate(imgs):
            keypoint_feature = model.preprocess(img)
            if keypoint_feature is None:
                continue
            total_video += 1
            legal_label.append(labels[index].item())
            keypoint_features.append(keypoint_feature)
        keypoint_features = torch.concat(keypoint_features, dim=0)
        batch_size, clip_len, num_joints, vector_len = keypoint_features.shape
        keypoint_features = keypoint_features.reshape(batch_size, -1, vector_len)
        optimizer.zero_grad()
        results = model(keypoint_features)
        results = results.sum(dim=1)
        labels = torch.LongTensor(legal_label)
        loss = loss_function(results, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predict = torch.softmax(results, dim=1)
        predict = torch.argmax(predict, dim=1)
        correct += torch.eq(predict, labels).sum().item()
    accuracy = round(correct / total_video * 100, 2)
    print(f'Epoch {epoch} => Accuracy {accuracy}')
    if best_loss > total_loss:
        torch.save(model.state_dict(), 'best_model.pkt')
    return total_loss


def eval_one_epoch(model, dataloader, device):
    pass


def main():
    dataset_cfg = [
        {'type': 'ReadVideo'},
        {'type': 'ExtractFrame', 'num_clips': 1, 'clip_len': 15, 'interval': 2},
        {'type': 'DecodeVideo'},
        {'type': 'ToTensor', 'targets': ['label'], 'to_rgb': False, 'img_transpose': False},
        {'type': 'Collect', 'targets': ['imgs', 'label', 'video_path']}
    ]
    data_path = '/Users/huanghongyan/Documents/DeepLearning/pytorch_geometric/TemporalGestureRecognition/PoseVideo'
    gesture_dataset = GestureDataset(data_path, dataset_cfg)
    dataloader_cfg = {
        'dataset': gesture_dataset,
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': True,
        'drop_last': False,
        'collate_fn': custom_collate_fn
    }
    gesture_dataloader = CreateDataloader(dataloader_cfg)
    model_cfg = {
        'type': 'TemporalGestureRecognition',
        'num_joints': 21,
        'num_classes': 4,
        'clip_len': 15,
        'keypoint_line': [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
                          [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20],
                          [0, 17]],
        'bilateral_without_temporal': True,
        'bilateral_with_temporal': False,
        'preprocess': {
            'type': 'ExtractKeypoint',
            'max_num_hands': 1,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        },
        'num_stage': 2,
        'stage_blocks': [3, 3],
        'block': BasicBlock
    }
    temporal_gesture_model = CreateModel(model_cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    temporal_gesture_model = temporal_gesture_model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(temporal_gesture_model.parameters(), lr=0.001, momentum=0.9)

    train_epoch = 100
    run(temporal_gesture_model, device, train_epoch, gesture_dataloader, loss_function, optimizer)
    print('Finish')


if __name__ == '__main__':
    main()
