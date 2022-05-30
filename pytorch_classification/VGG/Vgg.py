import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torch import nn
from torch.utils.tensorboard import SummaryWriter

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

net = models.vgg16(pretrained=True)

for k, v in net.named_parameters():
    v.requires_grad = False
more_layer = nn.Sequential(
    nn.Linear(in_features=4096, out_features=1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Linear(in_features=1024, out_features=512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Linear(in_features=512, out_features=100)
)
net.classifier[6] = more_layer
# print(net)
exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
writer = SummaryWriter('Log')


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

        acc = round(100*(correct/total), 2)
        print(batch_idx+1, '/', len(train_dataloader), f'Epoch {epoch+1}, Loss : {train_loss/(batch_idx+1)}, '
                                                     f'Acc : {acc}')
    writer.add_scalar('Vgg16 Train loss', train_loss/len(train_dataloader), epoch+1)
    writer.add_scalar('Vgg16 Train accuracy', round(100*(correct/total), 2), epoch+1)


def test(epoch):
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
        acc = round(100*(correct/total), 2)
        print(f'Train accuracy {acc}')
        writer.add_scalar('Vgg16 Test accuracy', acc, epoch + 1)
        torch.save(net.state_dict(), f'Model/vgg16cifar100_{epoch + 1}.pkl')


for epoch in range(0, 5):
    train(epoch)
    test(epoch)

print('Finish Training')
