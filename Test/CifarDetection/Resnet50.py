import torch
from torch import nn
import argparse
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from DatasetSource import cifar100_dataset_from_official


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--Epoch', type=int, default=50)
    args = parser.parse_args()
    return args


def train(model, dataloader, optimizer, loss_func, device, epoch, Epoch):
    model.train()
    loss, total, correct = 0, 0, 0
    pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch}/{Epoch}', postfix=dict, miniters=0.3)
    for batch_idx, (images, labels) in enumerate(dataloader):
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
        for batch_idx, (images, labels) in enumerate(dataloader):
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


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    transform_data = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = cifar100_dataset_from_official(root='cifar100', train=True, download=True,
                                                   transform_data=transform_data)
    eval_dataset = cifar100_dataset_from_official(root='cifar100', train=False, download=True,
                                                  transform_data=transform_data)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True,
                                  num_workers=args.num_workers)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                 num_workers=args.num_workers)
    model = models.resnet50(pretrained=True)
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features=fc_inputs, out_features=512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=100)
    )
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(args.Epoch):
        train(model, train_dataloader, optimizer, loss_func, device, epoch + 1, args.Epoch)
        eval(model, eval_dataloader, device, epoch + 1, args.Epoch)


if __name__ == '__main__':
    main()
    print('Finish')
