import os
import pickle
import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import math


class Compose:
    def __init__(self, cfg):
        support_operation = {
            'Format': Format,
            'ToTensor': ToTensor,
            'Collect': Collect
        }
        self.pipeline = list()
        for operation_cfg in cfg:
            operation_name = operation_cfg.pop('type', None)
            operation_cls = support_operation[operation_name]
            operation = operation_cls(**operation_cfg)
            self.pipeline.append(operation)

    def __call__(self, data):
        for operation in self.pipeline:
            data = operation(data)
        return data


class Format:
    def __init__(self, len, time, left):
        self.len = len
        self.time = time
        self.left = left

    def __call__(self, data):
        time = data.get(self.time, None)
        left = data.get(self.left, None)
        assert time is not None, '未獲取時間資料'
        assert left is not None, '未獲取剩餘量資料'
        label = time
        left = [101] + left + [102]
        label = [self.len] + label + [self.len + 1]
        left = left + [103] * (self.len + 2)
        label = label + [self.len + 2] * (self.len + 3)
        left = left[:self.len + 2]
        label = label[:self.len + 3]
        data['left'] = left
        data['label'] = label
        return data


class ToTensor:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for info_name in self.keys:
            info = data.get(info_name, None)
            assert info is not None, f'指定的{info_name}不在data當中'
            if isinstance(info, list):
                info = torch.LongTensor(info)
            else:
                raise NotImplementedError()
            data[info_name] = info
        return data


class Collect:
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        results = dict()
        for info_name in self.keys:
            info = data.get(info_name, None)
            assert info is not None, f'指定的{info_name}不在data當中'
            results[info_name] = info
        return results


class EatTimeDataset(Dataset):
    def __init__(self, data_path, dataset_cfg):
        assert os.path.exists(data_path), '指定的資料檔案不存在'
        assert os.path.splitext(data_path)[1] in ['.pkl'], '檔案不是pkl格式'
        file = open(data_path, 'rb')
        self.info = pickle.load(file)
        self.pipeline = Compose(dataset_cfg)

    def __getitem__(self, idx):
        data = self.info[idx]
        data = self.pipeline(data)
        return data

    def __len__(self):
        return len(self.info)


def EatTimeDataloader(dataloader_cfg):
    loader = DataLoader(**dataloader_cfg)
    return loader


def custom_collate_fn(batch):
    lefts = list()
    labels = list()
    for info in batch:
        lefts.append(info['left'])
        labels.append(info['label'])
    lefts = torch.stack(lefts)
    labels = torch.stack(labels)
    return lefts, labels


class PositionEmbedding(nn.Module):
    def __init__(self, len, time_embedding_channel, num_cls):
        super(PositionEmbedding, self).__init__()

        def get_pe(pos, i, d_model):
            fe_nmu = 1e4 ** (i / d_model)
            pe = pos / fe_nmu

            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)

        pe = torch.empty(len, time_embedding_channel)
        for i in range(len):
            for j in range(time_embedding_channel):
                pe[i, j] = get_pe(i, j, time_embedding_channel)
        pe = pe.unsqueeze(dim=0)
        self.register_buffer('pe', pe)
        self.embed = torch.nn.Embedding(num_cls, time_embedding_channel)
        self.embed.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # x shape = [batch_size, len=50]

        # x shape = [batch_size, len=50, channel=32]
        embed = self.embed(x)
        embed = embed + self.pe
        return embed


class Encoder(nn.Module):
    def __init__(self, time_embedding_channel, head):
        super(Encoder, self).__init__()
        self.layer1 = EncoderLayer(time_embedding_channel, head)
        self.layer2 = EncoderLayer(time_embedding_channel, head)
        self.layer3 = EncoderLayer(time_embedding_channel, head)

    def forward(self, x, mask):
        out = self.layer1(x, mask)
        out = self.layer2(out, mask)
        out = self.layer3(out, mask)
        return out


class EncoderLayer(nn.Module):
    def __init__(self, time_embedding_channel, head):
        super(EncoderLayer, self).__init__()
        self.multi_head = MultiHead(time_embedding_channel, head)
        self.fpn = FPN(time_embedding_channel)

    def forward(self, x, mask):
        # x shape = [batch_size, len, channel]
        out = self.multi_head(x, x, x, mask)
        out = self.fpn(out)
        return out


class Decoder(nn.Module):
    def __init__(self, time_embedding_channel, head):
        super(Decoder, self).__init__()
        self.layer1 = DecoderLayer(time_embedding_channel, head)
        self.layer2 = DecoderLayer(time_embedding_channel, head)
        self.layer3 = DecoderLayer(time_embedding_channel, head)

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer3(x, y, mask_pad_x, mask_tril_y)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, time_embedding_channel, head):
        super(DecoderLayer, self).__init__()
        self.self_multi_head = MultiHead(time_embedding_channel, head)
        self.cross_multi_head = MultiHead(time_embedding_channel, head)
        self.fpn = FPN(time_embedding_channel)

    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.self_multi_head(y, y, y, mask_tril_y)
        y = self.cross_multi_head(y, x, x, mask_pad_x)
        y = self.fpn(y)
        return y


class MultiHead(nn.Module):
    def __init__(self, time_embedding_channel, head):
        super(MultiHead, self).__init__()
        self.head = head
        self.fc_q = nn.Linear(time_embedding_channel, time_embedding_channel)
        self.fc_k = nn.Linear(time_embedding_channel, time_embedding_channel)
        self.fc_v = nn.Linear(time_embedding_channel, time_embedding_channel)
        self.out_fc = nn.Linear(time_embedding_channel, time_embedding_channel)
        self.norm = nn.LayerNorm(normalized_shape=time_embedding_channel, elementwise_affine=True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, q, k, v, mask):
        # q, k, v shape = [batch_size, len=50, channel=32]
        clone_q = q.clone()
        q = self.norm(q)
        k = self.norm(k)
        v = self.norm(v)
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)
        # [b, len, channel] -> [b, len, head, channel_per_head] -> [b, head, len, channel_per_head]
        batch_size, length, channel = q.shape
        q = q.reshape(batch_size, length, self.head, channel // self.head).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, length, self.head, channel // self.head).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, length, self.head, channel // self.head).permute(0, 2, 1, 3)
        norm = channel // self.head
        score = self.attention(q, k, v, mask, length, channel, norm)
        score = self.dropout(self.out_fc(score))
        score = score + clone_q
        return score

    @staticmethod
    def attention(q, k, v, mask, length, channel, norm):
        score = torch.matmul(q, k.permute(0, 1, 3, 2))
        score /= norm**0.5
        score = score.masked_fill_(mask, -float('inf'))
        score = torch.softmax(score, dim=-1)
        score = torch.matmul(score, v)
        score = score.permute(0, 2, 1, 3).reshape(-1, length, channel)
        return score


class FPN(nn.Module):
    def __init__(self, time_embedding_channel):
        super(FPN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=time_embedding_channel, out_features=time_embedding_channel * 2),
            nn.ReLU(),
            nn.Linear(in_features=time_embedding_channel * 2, out_features=time_embedding_channel),
            nn.Dropout(p=0.1)
        )
        self.norm = nn.LayerNorm(normalized_shape=time_embedding_channel, elementwise_affine=True)

    def forward(self, x):
        clone_x = x.clone()
        x = self.norm(x)
        out = self.fc(x)
        out = out + clone_x
        return out


class Transformer(nn.Module):
    def __init__(self, pad_left, pad_time, time_embedding_channel, len, num_cls_left, num_cls_time, head):
        super(Transformer, self).__init__()
        self.pad_left = pad_left
        self.pad_time = pad_time
        self.time_embedding_channel = time_embedding_channel
        self.len = len
        self.num_cls_left = num_cls_left
        self.num_cls_time = num_cls_time
        assert time_embedding_channel % head == 0, '設定的頭數與channel數需要是倍數關係'

        self.embed_x = PositionEmbedding(len, time_embedding_channel, num_cls_left)
        self.embed_y = PositionEmbedding(len, time_embedding_channel, num_cls_time)
        self.encoder = Encoder(time_embedding_channel, head)
        self.decoder = Decoder(time_embedding_channel, head)
        self.cls_fc = nn.Linear(time_embedding_channel, num_cls_time)

    def forward(self, x, y):
        # x, y shape = [batch_size, len=50]
        mask_pad_x = self.mask_pad(x, self.pad_left, self.len)
        mask_tril_y = self.mask_tril(y, self.pad_time, self.len)
        x, y = self.embed_x(x), self.embed_y(y)
        x = self.encoder(x, mask_pad_x)
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)
        out = self.cls_fc(y)
        return out

    @staticmethod
    def mask_pad(data, pad, len):
        # data shape = [batch_size, len=50]
        mask = data == pad
        mask = mask.reshape(-1, 1, 1, len)
        mask = mask.expand(-1, 1, len, len)
        return mask

    @staticmethod
    def mask_tril(data, pad, len):
        # data shape = [batch_size, len=50]
        tril = 1 - torch.tril(torch.ones(1, len, len, dtype=torch.long))
        mask = data == pad
        # mask shape = [batch_size, 1, len=50]
        mask = mask.unsqueeze(1).long()
        # mask shape = [batch_size, 50, len=50]
        mask = mask + tril
        mask = mask > 0
        # mask shape = [batch_size, 1, 50, len=50]
        mask = (mask == 1).unsqueeze(dim=1)
        return mask


def run(model, train_epoch, train_dataloader, device, optimizer, loss_function, val_epoch=None, val_dataloader=None):
    if val_epoch is not None:
        assert val_dataloader is not None, '啟用驗證模式就需要給定驗證的Dataloader'
    best_loss = 10000
    for epoch in range(train_epoch):
        loss = train_one_epoch(model, device, train_dataloader, optimizer, loss_function, best_loss, epoch)
        best_loss = min(best_loss, loss)
        if val_epoch is not None:
            if (epoch + 1) % val_epoch == 0:
                eval_one_epoch(model, device, val_dataloader)
    print('Finish training')


def train_one_epoch(model, device, dataloader, optimizer, loss_function, best_loss, epoch):
    model.train()
    total_loss = 0
    accuracy = 0
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}: ', postfix=f'Correct {accuracy}', mininterval=1) as pbar:
        for left, label in dataloader:
            left = left.to(device)
            label = label.to(device)
            output = model(left, label[:, :-1])
            batch_size, length, num_cls = output.shape
            pred = output.reshape(-1, num_cls)
            label = label[:, 1:].reshape(-1)
            select = label != 63
            pred = pred[select]
            label = label[select]
            loss = loss_function(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = pred.argmax(1)
            correct = torch.eq(pred, label).sum().item()
            accuracy = correct / len(pred)
            pbar.set_postfix_str(f'Accuracy => {round(accuracy * 100, 2)}, Loss => {round(total_loss, 2)}')
            pbar.update(1)
    if best_loss > total_loss:
        torch.save(model.state_dict(), 'best_model.pkt')
    return total_loss


def eval_one_epoch(model, device, dataloader):
    pass


def main():
    dataset_cfg = [
        {'type': 'Format', 'len': 61, 'time': 'time', 'left': 'left'},
        {'type': 'ToTensor', 'keys': ['left', 'label']},
        {'type': 'Collect', 'keys': ['left', 'label']}
    ]
    data_path = './data/food_time.pkl'
    train_dataset = EatTimeDataset(data_path, dataset_cfg)
    dataloader_cfg = {
        'dataset': train_dataset,
        'batch_size': 8,
        'drop_last': False,
        'shuffle': True,
        'num_workers': 1,
        'pin_memory': True,
        'collate_fn': custom_collate_fn
    }
    train_dataloader = EatTimeDataloader(dataloader_cfg)
    model_cfg = {
        'pad_left': 103,
        'pad_time': 63,
        'time_embedding_channel': 32,
        'len': 61 + 2,
        'num_cls_left': 104,
        'num_cls_time': 64,
        'head': 4
    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Transformer(**model_cfg)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_function = nn.CrossEntropyLoss()
    train_epoch = 7
    run(model, train_epoch, train_dataloader, device, optimizer, loss_function)


if __name__ == '__main__':
    main()
