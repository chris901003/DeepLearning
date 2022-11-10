import os
import torch
from torch import nn
import argparse
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from dataset import Cifar100Dataset
from DatasetSource import cifar100_dataset_from_official


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--Epoch', type=int, default=50)
    args = parser.parse_args()
    return args


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
    folder_path = './Training_data/1'
    pretrained = './resnet50.pth'
    answer_file = './410985048.txt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    support_image_format = ['.png']
    images_name = [image_name for image_name in os.listdir(folder_path)
                   if os.path.splitext(image_name)[1] in support_image_format]
    images_info = {int(os.path.splitext(image_name)[0]): image_name for image_name in images_name}
    images_info = sorted(images_info.items())
    transforms_data = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model = models.resnet50(pretrained=False)
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


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    # transform_data = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # train_dataset = cifar100_dataset_from_official(root='cifar100', train=True, download=True,
    #                                                transform_data=transform_data)
    # eval_dataset = cifar100_dataset_from_official(root='cifar100', train=False, download=True,
    #                                               transform_data=transform_data)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True,
    #                               num_workers=args.num_workers)
    # eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
    #                              num_workers=args.num_workers)
    train_dataset = Cifar100Dataset(annotation_path='./train_annotation.txt')
    eval_dataset = Cifar100Dataset(annotation_path='./train_annotation.txt')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
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
        torch.save(model.state_dict(), './resnet50.pth')


if __name__ == '__main__':
    predict_answer()
    print('Finish')
