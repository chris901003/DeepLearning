import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from alive_progress import alive_bar

train_data = torchvision.datasets.EMNIST(
    root='dataset',
    split='bymerge',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_data = torchvision.datasets.EMNIST(
    root='dataset',
    split='bymerge',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

every_batch_size = 640
train_dataloader = DataLoader(
    train_data,
    batch_size=every_batch_size
)
test_dataloader = DataLoader(
    test_data,
    batch_size=every_batch_size
)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.out = nn.Linear(in_features=32 * 7 * 7, out_features=47)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        return self.out(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net()
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    with alive_bar(len(train_dataloader)) as bar:
        for images, labels in train_dataloader:
            bar()
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = F.cross_entropy(input=outputs, target=labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print(f'Train, Epoch {epoch + 1}, Average loss => {total_loss / len(train_data)}')
    total_accuracy = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            outputs = torch.argmax(outputs, dim=1)
            total_accuracy += (outputs == labels).sum()
    print(f'Test, Epoch {epoch + 1}, Accuracy => {total_accuracy / len(test_data)}')
    # torch.save(net, f'MNIST_Bymerge_Model/Model_{epoch+1}')

print('Training Finish')
