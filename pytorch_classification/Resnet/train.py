import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from model import *
from alive_progress import alive_bar
from torch.utils.tensorboard import SummaryWriter
# from Copy import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device => {device}')

transform_data = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_data = torchvision.datasets.CIFAR100(
    root='../dataset',
    train=True,
    transform=transform_data,
    download=True
)
test_data = torchvision.datasets.CIFAR100(
    root='../dataset',
    train=False,
    transform=transform_data,
    download=True
)

every_batch_size = 640
train_dataloader = DataLoader(train_data, batch_size=every_batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=every_batch_size, shuffle=True)

net34 = resnet34(num_classes=100, include_top=True)
net34 = net34.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net34.parameters(), lr=0.001)
writer = SummaryWriter('Log')

for epoch in range(0, 200):
    total_loss = 0
    total_accuracy = 0
    net34.train()
    with alive_bar(len(train_dataloader)) as bar:
        for images, labels in train_dataloader:
            bar()
            images = images.to(device)
            labels = labels.to(device)
            outputs = net34(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
            outputs = torch.argmax(outputs, dim=1)
            total_accuracy += (outputs == labels).sum()
        print(f'Train epoch {epoch+1}, '
              f'Average loss => {total_loss/len(train_data)}, Accuracy => {total_accuracy/len(train_data)}')
        writer.add_scalar('Train average loss', total_loss.item()/len(train_data), epoch+1)
        writer.add_scalar('Train accuracy', total_accuracy.item()/len(train_data), epoch+1)
    total_accuracy = 0
    net34.eval()
    with alive_bar(len(test_dataloader)) as bar:
        for images, labels in test_dataloader:
            bar()
            images = images.to(device)
            labels = labels.to(device)
            outputs = net34(images)
            outputs = torch.argmax(outputs, dim=1)
            total_accuracy += (outputs == labels).sum()
        print(f'Test epoch {epoch+1}, Accuracy => {total_accuracy/len(test_data)}')
        writer.add_scalar('Test accuracy', total_accuracy.item()/len(test_data), epoch+1)
    torch.save(net34, f'Model/Model_{epoch+1}')

writer.close()
print('Finish Training')
