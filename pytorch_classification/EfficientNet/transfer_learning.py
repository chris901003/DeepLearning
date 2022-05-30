import torch
import torchvision
from utility import preprocess_data
from tiny_image_dataset import TinyImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
from torch.utils.tensorboard import SummaryWriter


root = '/Users/huanghongyan/Documents/LeNet/Pytorch/dataset/tiny-imagenet-200'
train_images_path, train_images_label, test_images_path, test_images_label = preprocess_data(root)
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = TinyImageDataset(train_images_path, train_images_label, data_transform)
test_dataset = TinyImageDataset(test_images_path, test_images_label, data_transform)
BATCH_SIZE = 200
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TinyImageDataset.collate_fn
)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=TinyImageDataset.collate_fn
)

net = models.resnet34(pretrained=True)
for k, v in net.named_parameters():
    v.requires_grad = False
fc_inputs = net.fc.in_features
net.fc = nn.Sequential(
    nn.Linear(in_features=fc_inputs, out_features=512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Linear(in_features=512, out_features=256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=200)
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
writer = SummaryWriter('Log')

for epoch in range(1, 51):
    total_accuracy = 0
    total_loss = 0
    total = 0
    net.train()
    for idx, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
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
    writer.add_scalar('Template train loss', total_loss / len(train_dataloader), epoch)
    writer.add_scalar('Template train accuracy', round(100 * (total_accuracy / total), 2), epoch)
    total_accuracy = 0
    net.eval()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            outputs = torch.argmax(outputs, dim=1)
            total_accuracy += (outputs == labels).sum().item()
            print(f'Epoch {epoch}, Test {idx}/{len(test_dataloader)}')
        print(f'Epoch {epoch}, Test accuracy {round(100 * (total_accuracy/len(test_dataset)), 2)}')
        writer.add_scalar('Template test accuracy', round(100 * (total_accuracy/len(test_dataset)), 2), epoch)
        torch.save(net.state_dict(), f'Model/Model_{epoch}.pkl')
writer.close()
print('Finish training')
