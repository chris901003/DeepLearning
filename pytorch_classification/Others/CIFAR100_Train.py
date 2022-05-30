import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

mean = [0.5, 0.5, 0.5]
std = [0.1, 0.1, 0.1]
transform_data = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_data = torchvision.datasets.CIFAR100(
    root='dataset',
    train=True,
    transform=transform_data,
    download=True
)
test_data = torchvision.datasets.CIFAR100(
    root='dataset',
    train=False,
    transform=transform_data,
    download=True
)

every_batch_size = 128
train_dataloader = DataLoader(train_data, batch_size=every_batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=every_batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=64, momentum=0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=64, momentum=0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=128, momentum=0.5),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=128, momentum=0.5),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=256, momentum=0.5),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=256, momentum=0.5),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=256, momentum=0.5),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=512, momentum=0.5),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=512, momentum=0.5),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=512, momentum=0.5),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=512, momentum=0.5),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=512, momentum=0.5),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=512, momentum=0.5),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=1*1*512, out_features=4096),
            nn.BatchNorm1d(4096, momentum=0.5),
            nn.Dropout(0.4),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(4096, momentum=0.5),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.BatchNorm1d(4096, momentum=0.5),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(in_features=4096, out_features=100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc4(x)


net = Net()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
writer = SummaryWriter('CIFAR100_Log')

for epoch in range(0, 200):
    total_loss = 0
    total_accuracy = 0
    # net.train()
    with alive_bar(len(train_dataloader)) as bar:
        for images, labels in train_dataloader:
            bar()
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
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
    # net.eval()
    with alive_bar(len(test_dataloader)) as bar:
        for images, labels in test_dataloader:
            bar()
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            outputs = torch.argmax(outputs, dim=1)
            total_accuracy += (outputs == labels).sum()
        print(f'Test epoch {epoch+1}, Accuracy => {total_accuracy/len(test_data)}')
        writer.add_scalar('Test accuracy', total_accuracy.item()/len(test_data), epoch+1)
    torch.save(net, f'CIFAR100_Model/Model_{epoch+1}')

writer.close()
print('Finish Training')
