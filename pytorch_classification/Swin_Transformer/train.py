import os

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import swin_base_patch4_window7_224


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])

train_dataset = torchvision.datasets.CIFAR100(root='../dataset', train=True, transform=train_transform, download=True)
val_dataset = torchvision.datasets.CIFAR100(root='../dataset', train=False, transform=val_transform, download=True)

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = swin_base_patch4_window7_224(num_classes=100).to(device)
if os.path.exists(os.path.join('weight', 'swin_base_patch4_window7_224.pth')):
    weights_dict = torch.load(os.path.join('weight', 'swin_base_patch4_window7_224.pth'), map_location=device)['models']
    for k in list(weights_dict.keys()):
        if 'head' in k:
            del weights_dict[k]
    print(model.load_state_dict(weights_dict, strict=False))
    for name, para in model.named_parameters():
        if 'head' not in name:
            para.requires_grad_(False)
        else:
            print(f'training {name}')
pg = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(pg, lr=0.001, weight_decay=5E-2)
loss_function = torch.nn.CrossEntropyLoss()

for epoch in range(1, 31):
    total_accuracy = 0
    total_loss = 0
    total = 0
    model.train()
    for idx, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        outputs = torch.argmax(outputs, dim=1)
        total_accuracy += (labels == outputs).sum().item()
        total += len(labels)
        acc = round(100*(total_accuracy/total), 2)
        print(f'Epoch {epoch}, {idx+1}/{len(train_dataloader)}, loss => {total_loss/(idx+1)}, accuracy => {acc}')
        exit()
    total_accuracy = 0
    model.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)
            total_accuracy += (outputs == labels).sum().item()
            print(f'Epoch {epoch}, Test {idx+1}/{len(val_dataloader)}')
        print(f'Epoch {epoch}, Test accuracy {round(100 * (total_accuracy/len(val_dataloader)), 2)}')
        torch.save(model.state_dict(), f'weight/Model_{epoch}.pkl')
print('Finish training')
