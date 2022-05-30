import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Sequential, Flatten
from torch.utils.tensorboard import SummaryWriter
from alive_progress import alive_bar
import torch.nn.functional as F
from torch.autograd import Variable

# 加入資料集

train_data = torchvision.datasets.MNIST(
    root='dataset',
    train=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ]),
    download=True
)
test_data = torchvision.datasets.MNIST(
    root='dataset',
    train=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ]),
    download=True
)

# 資料集大小
train_data_len = len(train_data)
test_data_len = len(test_data)

# 資料集加載方式
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32)
        )
        self.conv2 = Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64)
        )
        self.fc1 = Sequential(
            nn.Linear(in_features=64*5*5, out_features=512),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512)
        )
        self.fc2 = Sequential(
            nn.Linear(in_features=512, out_features=84),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=84)
        )
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


net = Net()
opt = torch.optim.Adam(params=net.parameters(), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

for epoch in range(10):
    total_loss = 0
    with alive_bar(len(train_dataloader)) as bar:
        for images, labels in train_dataloader:
            bar()
            images = images.to(device)
            labels = labels.to(device)
            pre_label = net(images)
            loss = F.cross_entropy(input=pre_label, target=labels)
            total_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch == 9:
            torch.save(net, f'LeNet_Model/ReLU_CrossEntropy_{epoch+1}')
        print(f'Epoch {epoch+1}, Average loss => {total_loss/len(train_data)}')
    total_accuracy = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            # outputs = torch.squeeze(outputs)
            # print(outputs)
            outputs = torch.argmax(outputs, dim=1)
            # print(outputs)
            # print(labels)
            total_accuracy += (outputs == labels).sum()
    print(f'Epoch  {epoch+1}, Accuracy => {total_accuracy/len(test_data)}')
