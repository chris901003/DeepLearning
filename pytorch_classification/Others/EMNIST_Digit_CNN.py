import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.EMNIST(
    root='dataset',
    split='digits',
    transform=torchvision.transforms.ToTensor(),
    train=True,
    download=True
)
test_data = torchvision.datasets.EMNIST(
    root='dataset',
    split='digits',
    transform=torchvision.transforms.ToTensor(),
    train=False,
    download=True
)

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=128*3*3, out_features=1024),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=84),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cnn = CNN()
cnn = cnn.to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

for epoch in range(0, 10):
    total_loss = 0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        output = cnn(images)
        loss = loss_function(output, labels)
        total_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Average loss => {total_loss/len(train_data)}')
    total_accuracy = 0
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        output = cnn(images)
        output = torch.argmax(output, dim=1)
        total_accuracy += (output == labels).sum()
    print(f'Epoch {epoch+1}, Accuracy => {total_accuracy/len(test_data)}')
    torch.save(cnn, f'EMNIST_Digit_CNN/Model_{epoch+1}')
