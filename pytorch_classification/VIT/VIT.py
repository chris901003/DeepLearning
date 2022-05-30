import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models
from model import vit_base_patch16_224

train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])
test_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])

train_dataset = torchvision.datasets.CIFAR10(root='../dataset', train=True, transform=train_transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False, transform=train_transform, download=True)

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = models.vit_b_16(pretrained=True)
for k, v in net.named_parameters():
    v.requires_grad = False
net.conv_proj = nn.Conv2d(3, 768, kernel_size=2, stride=2)
net.heads = nn.Linear(768, 10)
net.patch_size = 2
net.image_size = 28
net = net.to(device)
# exit()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

for epoch in range(1, 31):
    total_accuracy = 0
    total_loss = 0
    total = 0
    net.train()
    for idx, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        # print(outputs.shape)
        # print(labels.shape)
        # exit()
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        outputs = torch.argmax(outputs, dim=1)
        total_accuracy += (labels == outputs).sum().item()
        total += len(labels)
        acc = round(100*(total_accuracy/total), 2)
        print(f'Epoch {epoch}, {idx+1}/{len(train_dataloader)}, loss => {total_loss/(idx+1)}, accuracy => {acc}')
    total_accuracy = 0
    net.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            outputs = torch.argmax(outputs, dim=1)
            total_accuracy += (outputs == labels).sum().item()
            print(f'Epoch {epoch}, Test {idx+1}/{len(test_dataloader)}')
        print(f'Epoch {epoch}, Test accuracy {round(100 * (total_accuracy/len(test_dataset)), 2)}')
        torch.save(net.state_dict(), f'Model/Model_{epoch}.pkl')
print('Finish training')

