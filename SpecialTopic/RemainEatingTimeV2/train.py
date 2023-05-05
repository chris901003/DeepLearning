import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pickle
from tqdm import tqdm


class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, with_elapsed_time, with_avg_diff):
        super(RegressionModel, self).__init__()
        self.with_elapsed_time = with_elapsed_time
        self.with_avg_diff = with_avg_diff
        self.remain_embed = nn.Embedding(101, input_size)
        if with_elapsed_time:
            self.elapsed_time_embed = nn.Linear(1, input_size)
        if with_avg_diff:
            # -100 ~ 100
            self.avg_diff_embed = nn.Embedding(201, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.seq = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        embed_elapsed_time = None
        avg_embed = None
        if self.with_elapsed_time:
            elapsed_time = x[:, -1].float()
            elapsed_time = elapsed_time.unsqueeze(dim=-1)
            elapsed_time = elapsed_time.unsqueeze(dim=-1)
            embed_elapsed_time = self.elapsed_time_embed(elapsed_time)
            x = x[:, :-1]
        if self.with_avg_diff:
            time_range = int(x.shape[1] / 2)
            avg_diff = x[:, time_range:]
            avg_embed = self.avg_diff_embed(avg_diff)
            x = x[:, :time_range]
        x = self.remain_embed(x)
        if self.with_avg_diff:
            x = torch.concat((x, avg_embed), dim=1)
        if self.with_elapsed_time:
            x = torch.concat((x, embed_elapsed_time), dim=1)
        r_out, (_, _) = self.lstm(x)
        outs = r_out[:, -1, :]
        outs = self.seq(outs)
        return outs


class RegressionDataset(Dataset):
    def __init__(self, dataset_path, with_elapsed_time, with_avg_diff):
        with open(dataset_path, 'rb') as f:
            self.remain_info = pickle.load(f)
        self.with_elapsed_time = with_elapsed_time
        self.with_avg_diff = with_avg_diff

    def __getitem__(self, idx):
        data_info = self.remain_info[idx]
        remain = data_info["remain"]
        remain = [max(0, min(100, rem)) for rem in remain]
        if self.with_avg_diff:
            avg = int(sum(remain) / len(remain))
            avg_dif = [int(rem - avg) + 100 for rem in remain]
            remain.extend(avg_dif)
        if self.with_elapsed_time:
            elapsed_time = data_info["elapsed_time"]
            remain.append(elapsed_time)
        label = data_info["label"]
        return {'remain': remain, 'label': label}

    def __len__(self):
        return len(self.remain_info)

    @staticmethod
    def collate_fn(batch):
        remains = list()
        labels = list()
        for info in batch:
            remains.append(info["remain"])
            labels.append(info["label"])
        remains = torch.Tensor(remains).long()
        labels = torch.Tensor(labels)
        result = {'remains': remains, 'labels': labels}
        return result


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lstm-input-size', type=int, default=32)
    parser.add_argument('--lstm-hidden-size', type=int, default=64)
    parser.add_argument('--lstm-num-layers', type=int, default=2)

    # 添加已經過時間
    parser.add_argument('--with-elapsed-time', type=bool, default=True)
    # 添加每個值對整段的平均差異
    parser.add_argument('--with-avg-diff', type=bool, default=True)

    parser.add_argument('--training-dataset-path', type=str, default='train_set.pickle')
    parser.add_argument('--val-dataset-path', type=str, default='val_set.pickle')
    parser.add_argument('--save-path', type=str, default='regression_model.pth')
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epoch = args.epoch
    lr = args.lr
    training_dataset_path = args.training_dataset_path
    val_dataset_path = args.val_dataset_path
    save_path = args.save_path
    with_elapsed_time = args.with_elapsed_time
    with_avg_diff = args.with_avg_diff
    model = RegressionModel(
        input_size=args.lstm_input_size,
        hidden_size=args.lstm_hidden_size,
        num_layers=args.lstm_num_layers,
        with_elapsed_time=with_elapsed_time,
        with_avg_diff=with_avg_diff
    )
    model = model.to(device)
    assert os.path.exists(training_dataset_path), '給定訓練資料不存在'
    train_dataset = RegressionDataset(training_dataset_path, with_elapsed_time, with_avg_diff)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=1, pin_memory=True,
                                  collate_fn=train_dataset.collate_fn)
    val_dataset = RegressionDataset(val_dataset_path, with_elapsed_time, with_avg_diff)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=1, pin_memory=True,
                                  collate_fn=val_dataset.collate_fn)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epo in range(1, epoch + 1):
        train_one_epoch(model, epo, epoch, device, train_dataloader, loss_function, optimizer, save_path)
        val_one_epoch(model, epo, epoch, device, val_dataloader, loss_function)
        epo += 1


def train_one_epoch(model, epo, epoch, device, dataloader, loss_function, optimizer, save_path):
    total_loss = 0
    acc = 0
    tot = 0
    model.train()
    pbar = tqdm(total=len(dataloader), desc=f'Epoch {epo} / {epoch}', postfix=dict, miniters=0.3)
    for iteration, data in enumerate(dataloader):
        remains = data["remains"]
        labels = data["labels"]
        remains = remains.to(device)
        labels = labels.to(device)
        prediction = model(remains).squeeze(dim=-1)
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        for pre, lab in zip(prediction, labels):
            pre = pre.item()
            lab = lab.item()
            if abs(pre - lab) < 10:
                acc += 1
        tot += labels.numel()
        pbar.set_postfix(**{
            'loss': total_loss / (iteration + 1),
            'acc': acc / tot
        })
        pbar.update(1)
    pbar.close()
    torch.save(model.state_dict(), save_path)


def val_one_epoch(model, epo, epoch, device, dataloader, loss_function):
    total_loss = 0
    acc = 0
    tot = 0
    model.eval()
    pbar = tqdm(total=len(dataloader), desc=f'Epoch {epo} / {epoch}', postfix=dict, miniters=0.3)
    with torch.no_grad():
        for iteration, data in enumerate(dataloader):
            remains = data["remains"]
            labels = data["labels"]
            remains = remains.to(device)
            labels = labels.to(device)
            prediction = model(remains).squeeze(dim=-1)
            loss = loss_function(prediction, labels)
            total_loss += loss.item()
            for pre, lab in zip(prediction, labels):
                pre = pre.item()
                lab = lab.item()
                if abs(pre - lab) < 10:
                    acc += 1
            tot += labels.numel()
            pbar.set_postfix(**{
                'loss': total_loss / (iteration + 1),
                'acc': acc / tot
            })
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    main()
