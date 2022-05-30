import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torch import nn

transform_data = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
BATCH_SIZE = 640
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

net = models.resnet34(pretrained=False)

fc_inputs = net.fc.in_features
net.fc = nn.Sequential(
    nn.Linear(in_features=fc_inputs, out_features=100)
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load('Model/resnet34cifar100_60.pkl', map_location='cpu'))
net.to(device)
for k, v in net.named_parameters():
    if k != 'fc.0.weight' and k != 'fc.0.bias':
        v.requires_grad = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        acc = round(100 * (correct / total), 2)
        print(batch_idx, '/', len(train_dataloader), f'Epoch {epoch + 1}, Loss : {train_loss / (batch_idx + 1)}, '
                                                     f'Acc : {acc}')


def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        acc = round(100 * (correct / total), 2)
        print(f'Train accuracy {acc}')


for epoch in range(0, 5):
    train(epoch)
    test()
    torch.save(net.state_dict(), f'Model/resnet34cifar100_{epoch + 1}.pkl')
