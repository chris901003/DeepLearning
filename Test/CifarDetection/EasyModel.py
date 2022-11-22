import torch
from torch import nn
import os
import numpy as np
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Cifar100Dataset


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride):
        super(BaseConv, self).__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(Focus, self).__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize=1, stride=1)

    def forward(self, x):
        top_left = x[..., ::2, ::2]
        bot_left = x[..., 1::2, ::2]
        top_right = x[..., ::2, 1::2]
        bot_right = x[..., 1::2, 1::2]
        x = torch.cat((top_left, bot_left, top_right, bot_right), dim=1)
        return self.conv(x)


class StageLayer(nn.Module):
    def __init__(self, in_channels):
        super(StageLayer, self).__init__()
        layer1 = BaseConv(in_channels=in_channels, out_channels=in_channels // 2, ksize=3, stride=1)
        layer2 = BaseConv(in_channels=in_channels // 2, out_channels=in_channels, ksize=3, stride=2)
        layer3 = BaseConv(in_channels=in_channels, out_channels=in_channels * 2, ksize=3, stride=1)
        self.stage1 = nn.Sequential(layer1, layer2, layer3)

        layer4 = BaseConv(in_channels=in_channels * 2, out_channels=in_channels, ksize=3, stride=1)
        layer5 = BaseConv(in_channels=in_channels, out_channels=in_channels, ksize=3, stride=1)
        layer6 = BaseConv(in_channels=in_channels, out_channels=in_channels * 2, ksize=3, stride=1)
        self.stage2 = nn.Sequential(layer4, layer5, layer6)

        layer7 = BaseConv(in_channels=in_channels * 2, out_channels=in_channels, ksize=3, stride=1)
        layer8 = BaseConv(in_channels=in_channels, out_channels=in_channels, ksize=3, stride=1)
        layer9 = BaseConv(in_channels=in_channels, out_channels=in_channels * 2, ksize=3, stride=1)
        self.stage3 = nn.Sequential(layer7, layer8, layer9)

        conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
        )
        conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU()
        )
        conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 4, out_channels=in_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU()
        )
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_stage = nn.Sequential(conv1, conv2, conv3, pool, nn.Dropout(0.2))

    def forward(self, x):
        return self.conv_stage(x)
        out1 = self.stage1(x)
        out2 = self.stage2(out1)
        out3 = self.stage3(out2)
        return out3


class Net(nn.Module):
    def __init__(self, base_channels=64, layers=3, num_classes=100):
        super(Net, self).__init__()
        self.focus = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=1)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.act = nn.ReLU()
        self.layers = nn.ModuleList()
        channels = base_channels
        for _ in range(layers):
            self.layers.append(StageLayer(in_channels=channels))
            channels *= 2
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels * 4 * 4, channels)
        self.dropout = nn.Dropout(0.3)
        self.cls = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.act(self.bn(self.focus(x)))
        for layer in self.layers:
            x = layer(x)
        # out = self.avg_pool(x)
        out = x.reshape(x.size(0), -1)
        out = self.act(self.dropout(self.fc(out)))
        out = self.cls(out)
        return out


def train(model, dataloader, optimizer, loss_func, device, epoch, Epoch):
    model.train()
    loss, total, correct = 0, 0, 0
    pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch}/{Epoch}', postfix=dict, miniters=0.3)
    for batch_idx, (images, labels, images_name) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        acc = round(100 * (correct / total), 2)
        pbar.set_postfix(**{'acc': acc})
        pbar.update(1)
    print(f'Epoch: {epoch}, Train loss: {loss / len(dataloader)}')
    print(f'Epoch: {epoch}, Train accuracy: {round(100 * (correct / total), 2)}')
    pbar.close()


def eval(model, dataloader, device, epoch, Epoch):
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch}/{Epoch}', postfix=dict, miniters=0.3)
    with torch.no_grad():
        for batch_idx, (images, labels, images_name) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            acc = round(100 * (correct / total), 2)
            pbar.set_postfix(**{'acc': acc})
            pbar.update(1)
    acc = round(100 * (correct / total), 2)
    print(f'Epoch: {epoch}, Eval accuracy: {acc}')
    pbar.close()


def predict_answer():
    from PIL import Image
    folder_path = '410985048'
    pretrained = './model.pth'
    answer_file = './410985048.txt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    support_image_format = ['.png']
    images_name = [image_name for image_name in os.listdir(folder_path)
                   if os.path.splitext(image_name)[1] in support_image_format]
    images_info = {int(os.path.splitext(image_name)[0]): image_name for image_name in images_name}
    images_info = sorted(images_info.items())
    transforms_data = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model = Net()
    model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    model.eval()
    model = model.to(device)
    results = list()
    for _, image_name in tqdm(images_info):
        image = Image.open(os.path.join(folder_path, image_name))
        image = transforms_data(image)
        image = image.unsqueeze(dim=0)
        with torch.no_grad():
            image = image.to(device)
            preds = model(image)
        preds = preds.squeeze(dim=0)
        preds = preds.argmax().item()
        info = os.path.splitext(image_name)[0] + ' ' + str(preds)
        results.append(info)
    with open(answer_file, 'w') as f:
        for result in results:
            f.write(result)
            f.write('\n')
    print(f'Total {len(results)} pictures')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Total_Epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Net()
    model = model.to(device)
    model.load_state_dict(torch.load('model.pth'))
    batch_size = args.batch_size
    train_dataset = Cifar100Dataset(annotation_path='./train_annotation.txt')
    eval_dataset = Cifar100Dataset(annotation_path='./train_annotation.txt')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.9)
    loss_func = torch.nn.CrossEntropyLoss()
    for epoch in range(1, args.Total_Epoch + 1):
        train(model, train_dataloader, optimizer, loss_func, device, epoch, args.Total_Epoch)
        eval(model, eval_dataloader, device, epoch, args.Total_Epoch)
        torch.save(model.state_dict(), './model.pth')
        scheduler.step()


if __name__ == '__main__':
    predict_answer()
