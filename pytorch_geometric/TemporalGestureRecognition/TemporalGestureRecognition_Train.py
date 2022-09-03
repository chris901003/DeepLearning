import os
from torch.utils.data.dataset import Dataset
import pickle
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm


def zero(_):
    return 0


def get_cls_from_cfg(support, cfg):
    cls_type = cfg.pop('type', None)
    assert cls_type is not None, '在設定檔當中沒有指定type'
    assert cls_type in support, f'指定的{cls_type}尚未支援'
    cls = support[cls_type]
    return cls


class GestureDataset(Dataset):
    def __init__(self, data_path, dataset_pipeline):
        assert os.path.exists(data_path), '檔案資料不存在'
        assert os.path.splitext(data_path)[1] == '.pkl', '需要pkl檔案資料，如果需要轉換可以使用模板'
        file = open(data_path, 'rb')
        content = pickle.load(file)
        self.imgWidth = content['imgWidth']
        self.imgHeight = content['imgHeight']
        self.data_info = content['keypoints']
        self.pipeline = Compose(dataset_pipeline)

    def __getitem__(self, idx):
        data = self.data_info[idx]
        data = self.pipeline(data)
        return data

    def __len__(self):
        return len(self.data_info)


class Compose:
    def __init__(self, dataset_pipeline):
        support_operation = {
            'PoseDecode': PoseDecode,
            'FormateGCNInput': FormateGCNInput,
            'KeypointNormalize': KeypointNormalize,
            'Collect': Collect,
            'ToTensor': ToTensor
        }
        self.pipelines = list()
        for pipeline_cfg in dataset_pipeline:
            pipeline_cls = get_cls_from_cfg(support_operation, pipeline_cfg)
            pipeline = pipeline_cls(**pipeline_cfg)
            self.pipelines.append(pipeline)

    def __call__(self, data):
        for pipeline in self.pipelines:
            data = pipeline(data)
        return data


class PoseDecode:
    def __init__(self, merge):
        self.merge = merge

    def __call__(self, data):
        if len(self.merge) == 0:
            return data
        # keypoint shape = [幀數, 關節點數量, 每個關節點的資訊數量, 人數]
        keypoint = data.get(self.merge[0], None)
        # keypoint shape = [人數, 幀數, 關節點數量, 每個關節點的資訊數量]
        keypoint = keypoint.permute(3, 0, 1, 2)
        assert keypoint is not None, f'指定的{self.merge[0]}不在data當中'
        for info_name in self.merge[1:]:
            info = data.get(info_name, None)
            assert info is not None, f'指定的{info_name}不在data當中'
            info = info.permute(3, 0, 1, 2)
            assert keypoint.shape[:3] == info.shape[:3], f'指定的{info_name}與當前資料的shape無法進行拼接'
            keypoint = torch.cat([keypoint, info], dim=-1)
        # keypoint shape = [幀數, 關節點數量, 每個關節點的資訊數量, 人數]
        keypoint = keypoint.permute(1, 2, 3, 0).contiguous()
        data['keypoints'] = keypoint
        return data


class FormateGCNInput:
    def __init__(self, input_format, max_hands):
        support_format = ['NCTVM']
        assert input_format in support_format, '該整理的型態尚未支持'
        self.input_format = input_format
        self.max_hands = max_hands

    def __call__(self, data):
        # keypoint shape = [幀數, 關節點數量, 每個關節點的資訊數量, 人數]
        keypoints = data.get('keypoints', None)
        assert keypoints is not None, '無法獲取data當中keypoints資訊'
        frames, num_keypoints, information, hands = keypoints.shape
        if hands < self.max_hands:
            pad_dim = self.max_hands - hands
            pad = np.zeros(keypoints.shape[:-1] + (pad_dim, ), dtype=float)
            pad = torch.from_numpy(pad)
            keypoints = torch.cat([keypoints, pad], dim=-1)
        else:
            keypoints = keypoints[..., :self.max_hands]
        keypoints = keypoints.permute(2, 0, 1, 3).contiguous()
        data['keypoints'] = keypoints
        return data


class KeypointNormalize:
    def __init__(self, min_value, max_value, mean):
        self.min_value = np.array(min_value, dtype=np.float32).reshape(-1, 1, 1, 1)
        self.max_value = np.array(max_value, dtype=np.float32).reshape(-1, 1, 1, 1)
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1, 1, 1)

    def __call__(self, data):
        keypoints = data.get('keypoints', None)
        assert keypoints is not None, '在Data當中沒有獲取keypoints資訊'
        keypoints = (keypoints - self.mean) / (self.max_value - self.min_value)
        data['keypoints'] = keypoints
        data['keypoints_norm_cfg'] = {
            'mean': self.mean,
            'min_value': self.min_value,
            'max_value': self.max_value
        }
        return data


class Collect:
    def __init__(self, target):
        self.target = target

    def __call__(self, data):
        results = dict()
        for info_name in self.target:
            info = data.get(info_name, None)
            assert info is not None, f'Data當中沒有指定{info_name}資訊'
            results[info_name] = info
        return results


class ToTensor:
    def __init__(self, target):
        self.target = target

    def __call__(self, data):
        for info_name in self.target:
            info = data.get(info_name, None)
            assert info is not None, f'Data當中沒有指定{info_name}資訊'
            info = torch.Tensor(info)
            data[info_name] = info
        return data


def CreateDataloader(cfg):
    dataloader = DataLoader(**cfg)
    return dataloader


def custom_collate_fn(batch):
    keypoints_collate = list()
    label_collate = list()
    for info in batch:
        keypoints_collate.append(info['keypoints'])
        label_collate.append(int(info['label']))
    keypoints_collate = torch.stack(keypoints_collate)
    label_collate = torch.Tensor(label_collate)
    return keypoints_collate, label_collate


def get_hop_distance(num_node, edge, max_hop=1):
    adj_mat = np.zeros((num_node, num_node))
    for i, j in edge:
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(adj_mat, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(adj_matrix):
    Dl = np.sum(adj_matrix, 0)
    num_nodes = adj_matrix.shape[0]
    Dn = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    norm_matrix = np.dot(adj_matrix, Dn)
    return norm_matrix


class KeypointGraph:
    def __init__(self, layout='mediapipe_hands', strategy='spatial', max_hop=1, dilation=1):
        self.num_node = None
        self.self_link = None
        self.neighbor = None
        self.edge = []
        self.center = None
        self.A = None
        self.max_hop = max_hop
        self.dilation = dilation
        # 目前比較偷懶先實現這兩個就可以了
        assert layout in ['mediapipe_hands']
        assert strategy in ['spatial']
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'mediapipe_hands':
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
                              [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20],
                              [0, 17]]
            self.self_link = self_link
            self.neighbor = neighbor_link
            self.edge = self.self_link + self.neighbor
            self.center = 9
        else:
            raise ValueError(f'{layout}尚未實作')

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)
        if strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                            a_root[j, i] = normalize_adjacency[j, i]
                        elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                            a_close[j, i] = normalize_adjacency[j, i]
                        else:
                            a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError(f'沒有{strategy}作法')


def CreateModel(model_cfg):
    support_model = {
        'SkeletonGCN': SkeletonGCN
    }
    model_cls = get_cls_from_cfg(support_model, model_cfg)
    model = model_cls(**model_cfg)
    return model


def build_backbone(backbone_cfg):
    support_backbone = {
        'STGCN': STGCN
    }
    backbone_cls = get_cls_from_cfg(support_backbone, backbone_cfg)
    backbone = backbone_cls(**backbone_cfg)
    return backbone


def build_cls_head(cls_head_cfg):
    support_cls_head = {
        'STGCNHead': STGCNHead
    }
    cls_head_cls = get_cls_from_cfg(support_cls_head, cls_head_cfg)
    cls_head = cls_head_cls(**cls_head_cfg)
    return cls_head


def build_loss(loss_cfg):
    support_loss = {
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss
    }
    loss_cls = get_cls_from_cfg(support_loss, loss_cfg)
    loss = loss_cls(**loss_cfg)
    return loss


class SkeletonGCN(nn.Module):
    def __init__(self, backbone, cls_head):
        super(SkeletonGCN, self).__init__()
        self.backbone = build_backbone(backbone)
        self.cls_head = build_cls_head(cls_head) if cls_head is not None else None

    def forward(self, x, labels):
        assert self.cls_head is not None, '需要解碼頭'
        x = self.backbone(x)
        output = self.cls_head(x)
        gt_labels = labels.squeeze(-1)
        loss = self.cls_head.loss(output, gt_labels)
        return loss


class STGCN(nn.Module):
    def __init__(self, in_channels, graph_cfg, edge_importance_weighting=True, data_bn=True, pretrained=None):
        super(STGCN, self).__init__()
        self.graph = KeypointGraph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1)) if data_bn else nn.Identity
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, kernel_size, 1, residual=False),
            STGCNBlock(64, 64, kernel_size, stride=1),
            STGCNBlock(64, 64, kernel_size, stride=1),
            STGCNBlock(64, 64, kernel_size, stride=1),
            STGCNBlock(64, 128, kernel_size, stride=2),
            STGCNBlock(128, 128, kernel_size, stride=1),
            STGCNBlock(128, 128, kernel_size, stride=1),
            STGCNBlock(128, 256, kernel_size, stride=2),
            STGCNBlock(256, 256, kernel_size, stride=1),
            STGCNBlock(256, 256, kernel_size, stride=1),
        ))
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1 for _ in self.st_gcn_networks]
        self.pretrained = pretrained

    def forward(self, x):
        # x = tensor shape [batch_size, channel(x, y, z), frames, num_node, people]
        x = x.float()
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        # x shape [batch_size, people, channel, frames, num_node]
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        return x


class BaseHead(nn.Module):
    def __init__(self, num_classes, in_channels, loss_cls, top_k=(1,)):
        super(BaseHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = build_loss(loss_cls)
        assert isinstance(top_k, (int, tuple))
        if isinstance(top_k, int):
            top_k = (top_k,)
        for _topk in top_k:
            assert _topk > 0
        self.top_k = top_k

    def loss(self, cls_score, labels):
        # cls_score shape = [batch_size, num_classes]
        loss_dict = dict()
        labels = labels.type(torch.LongTensor)
        loss = self.loss_cls(cls_score, labels)
        loss_dict['loss'] = loss
        pred = torch.softmax(cls_score, dim=1)
        res = torch.argmax(pred, dim=1)
        loss_dict['pred'] = pred
        loss_dict['cls'] = res
        correct = torch.eq(res, labels).sum().item()
        loss_dict['acc'] = correct
        return loss_dict


class STGCNHead(BaseHead):
    def __init__(self, num_classes, in_channels, loss_cls, spatial_type='avg', num_hands=1, init_std=0.01):
        super(STGCNHead, self).__init__(num_classes, in_channels, loss_cls)
        self.spatial_type = spatial_type
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_hands = num_hands
        self.init_std = init_std
        self.pool = None
        if self.spatial_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        elif self.spatial_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise NotImplementedError
        self.fc = nn.Conv2d(self.in_channels, self.num_classes, kernel_size=1)

    def forward(self, x):
        # x shape = [batch_size * hands, channel, frames, num_node]
        assert self.pool is not None
        x = self.pool(x)
        # x shape = [batch_size, channel, 1, 1]
        x = x.view(x.shape[0] // self.num_hands, self.num_hands, -1, 1, 1).mean(dim=1)
        x = self.fc(x)
        # x shape = [batch_size, num_classes]
        x = x.view(x.shape[0], -1)
        return x


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=False):
        super(STGCNBlock, self).__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding),
            nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True)
        )
        if not residual:
            self.residual = zero
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = residual
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, adj_mat):
        # x shape = [batch_size * hands, channel, frames, num_node]
        res = self.residual(x)
        x, adj_mat = self.gcn(x, adj_mat)
        x = self.tcn(x) + res
        return self.relu(x), adj_mat


class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                            t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super(ConvTemporalGraphical, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, adj_mat):
        # x shape = [batch_size * hands, channel, frames, num_node]
        assert adj_mat.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, adj_mat))
        return x.contiguous(), adj_mat


def run(model, device, train_epoch, train_dataloader, optimizer, val_epoch=None, val_dataloader=None):
    if val_epoch is not None:
        assert val_dataloader is not None, '啟用驗證模式就需要給定驗證的Dataloader'
    best_loss = 10000
    for epoch in range(train_epoch):
        loss = train_one_epoch(model, device, train_dataloader, optimizer, best_loss, epoch)
        best_loss = min(best_loss, loss)
        if val_epoch is not None:
            if (epoch + 1) % val_epoch == 0:
                eval_one_epoch(model, device, val_dataloader)
    print('Finish training')


def train_one_epoch(model, device, dataloader, optimizer, best_loss, epoch):
    model.train()
    total_loss = 0
    correct = 0
    video = 0
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}: ', postfix=f'Correct {correct}', mininterval=1) as pbar:
        for imgs, labels in dataloader:
            video += imgs.size(0)
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss_dict = model(imgs, labels)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += loss_dict['acc']
            pbar.set_postfix_str(f'Accuracy => {round(correct / video * 100, 2)}, Loss => {round(total_loss, 2)}')
            pbar.update(1)
    if best_loss > total_loss:
        torch.save(model.state_dict(), 'best_model.pkt')
    return total_loss


def eval_one_epoch(model, device, dataloader):
    pass


def main():
    dataset_pipeline = [
        {'type': 'PoseDecode', 'merge': ['keypoints', 'z_axis']},  # 主要是將座標資訊以及置信度合併
        {'type': 'FormateGCNInput', 'input_format': 'NCTVM', 'max_hands': 1},
        {'type': 'KeypointNormalize',
         'min_value': (0., 0., 0.), 'max_value': (1920., 1080., 1.), 'mean': (940., 540., 0.5)},
        {'type': 'Collect', 'target': ['keypoints', 'label']},
        {'type': 'ToTensor', 'target': ['keypoints']}
    ]
    data_path = '/Users/huanghongyan/Documents/DeepLearning/pytorch_geometric/TemporalGestureRecognition' \
                '/PoseVideo/extract.pkl'
    gesture_dataset = GestureDataset(data_path, dataset_pipeline)
    dataloader_cfg = {
        'dataset': gesture_dataset,
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 8,
        'pin_memory': True,
        'drop_last': False,
        'collate_fn': custom_collate_fn
    }
    gesture_dataloader = CreateDataloader(dataloader_cfg)
    model_cfg = {
        'type': 'SkeletonGCN',
        'backbone': {
            'type': 'STGCN',
            'in_channels': 3,
            'edge_importance_weighting': True,
            'graph_cfg': {
                'layout': 'mediapipe_hands',
                'strategy': 'spatial'
            }
        },
        'cls_head': {
            'type': 'STGCNHead',
            'num_classes': 4,
            'in_channels': 256,
            'loss_cls': {
                'type': 'CrossEntropyLoss'
            }
        }
    }
    temporal_gesture_model = CreateModel(model_cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    temporal_gesture_model = temporal_gesture_model.to(device)
    optimizer = torch.optim.SGD(temporal_gesture_model.parameters(), lr=0.001, momentum=0.9)

    train_epoch = 30
    run(temporal_gesture_model, device, train_epoch, gesture_dataloader, optimizer)
    print('Finish run')


if __name__ == '__main__':
    main()
    print('Finish')
