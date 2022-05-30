import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from alive_progress import alive_bar

transfrom_data = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_data = torchvision.datasets.EMNIST(
    root='dataset',
    split='letters',
    transform=transfrom_data,
    download=True,
    train=True
)

test_data = torchvision.datasets.EMNIST(
    root='dataset',
    split='letters',
    transform=transfrom_data,
    download=True,
    train=False
)

every_batch_size = 640
train_dataloader = DataLoader(train_data, batch_size=every_batch_size)
test_dataloader = DataLoader(test_data, batch_size=every_batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # 輸入通道數
                out_channels=16,  # 輸出通道數
                kernel_size=5,   # 卷積核大小
                stride=1,  #卷積步數
                padding=2,  # 如果想要 con2d 出來的圖片長寬沒有變化,
                            # padding=(kernel_size-1)/2 當 stride=1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 在 2x2 空間裡向下採樣, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 37)  # 全連接層，A/Z,a/z一共37個類

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展平多維的卷積圖成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

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
    print(f'Train, Epoch {epoch+1}, Average loss => {total_loss/len(train_data)}')
    total_accuracy = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            outputs = torch.argmax(outputs, dim=1)
            total_accuracy += (outputs == labels).sum()
    print(f'Test, Epoch {epoch+1}, Accuracy => {total_accuracy/len(test_data)}')
    torch.save(net, f'EMNIST_Letters_Model/Model_{epoch+1}')

print('Training Finish')
