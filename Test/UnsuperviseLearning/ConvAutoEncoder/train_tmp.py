import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import models
from functools import cmp_to_key
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
from kmeans_pytorch import kmeans
from PIL import Image
from torch.nn import functional as F
from u2Net import u2net_full
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans


class StlDataset(Dataset):
    def __init__(self):
        super(StlDataset, self).__init__()
        self.dataRoot = r'C:\Dataset\Stl10\train_images'
        self.imagePath = list()
        for imageName in os.listdir(self.dataRoot):
            self.imagePath.append(os.path.join(self.dataRoot, imageName))
        self.imagePath = sorted(self.imagePath, key=cmp_to_key(self.image_path_cmp))

    def __getitem__(self, idx):
        idx = idx % len(self.imagePath)
        image = Image.open(self.imagePath[idx])
        image = transforms.ToTensor()(image)
        result = {'image': image, 'imagePath': self.imagePath[idx]}
        return result

    def __len__(self):
        return len(self.imagePath)

    @staticmethod
    def image_path_cmp(lhs, rhs):
        lhsBase = os.path.basename(lhs)
        rhsBase = os.path.basename(rhs)
        lhsBase = int(os.path.splitext(lhsBase)[0])
        rhsBase = int(os.path.splitext(rhsBase)[0])
        if lhsBase < rhsBase:
            return -1
        return 1

    @staticmethod
    def collate_fn(batch):
        images = list()
        imagePath = list()
        for info in batch:
            images.append(info['image'])
            imagePath.append(info['imagePath'])
        images = torch.stack(images, dim=0)
        return images, imagePath


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.resnet50 = models.resnet50()
        # 48x48
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(True)
        )
        # 24x24
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(True)
        )
        # 12x12
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(True)
        )
        # 6x6
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1, stride=2),
        #     nn.BatchNorm2d(num_features=32),
        #     nn.LeakyReLU(True)
        # )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, padding=1, stride=2),
        #     nn.BatchNorm2d(num_features=32),
        #     nn.LeakyReLU(True)
        # )
        self.fc = nn.Linear(in_features=12 * 12 * 16, out_features=1024)
        self.deFc = nn.Linear(in_features=1000, out_features=12 * 12 * 16)

        # self.deConv0 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(num_features=256),
        #     nn.ReLU(True)
        # )
        self.deConv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True)
        )
        self.deConv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True)
        )
        self.deConv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=3),
            nn.Tanh()
        )
        # self.deConv4 = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(num_features=3),
        #     nn.Tanh()
        # )

    def forward(self, x, feature=False):
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = x.reshape(x.size(0), 1, -1)
        # x = self.fc(x)
        # x = F.tanh(x)

        # x = self.conv4(x)
        # x = self.conv5(x)
        x = self.resnet50(x)
        if feature:
            x = x.reshape(x.size(0), -1)
            return x
        # x = self.deConv0(x)
        x = self.deFc(x)
        x = x.reshape(x.size(0), -1, 12, 12)
        x = self.deConv1(x)
        x = self.deConv2(x)
        x = self.deConv3(x)
        # x = self.deConv4(x)
        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = u2net_full()
    model = model.to(device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    train_data = StlDataset()
    num_workers = 0
    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                                               collate_fn=train_data.collate_fn, shuffle=True)
    # transform = transforms.ToTensor()
    # train_data2 = datasets.MNIST('data', train=True, download=True, transform=transform)
    # train_loader = torch.utils.data.DataLoader(train_data2, batch_size=batch_size, num_workers=num_workers,
    #                                            shuffle=True)
    n_epochs = 1000

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        for idx, data in enumerate(train_loader):
            images, _ = data
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if idx == 15:
                transformImage = transforms.ToPILImage()
                img = transformImage(outputs[0])
                img.show()
                img1 = transformImage(images[0])
                img1.show()

        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
        ))
        torch.save(model, 'pretrained3.pth')


def cluster():
    model = u2net_full()
    model.load_state_dict(torch.load('pretrained3.pth', map_location='cpu').state_dict())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    train_data = StlDataset()
    num_workers = 0
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,
                                               collate_fn=train_data.collate_fn)
    featuresList = list()
    imagesPathList = list()

    for idx, data in enumerate(train_loader):
        images, imagesPath = data
        imagesPathList.append(imagesPath)
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images, feature=True)
            # outputs = F.softmax(outputs, dim=1)
        featuresList.append(outputs)
    features = torch.concat(featuresList, dim=0)

    X = features.cpu().numpy()
    # scaler = StandardScaler().fit(X)
    # X = scaler.transform(X)
    # features = torch.from_numpy(X)
    kmeans_fit = KMeans(n_clusters=10, random_state=42)
    k_pred = kmeans_fit.fit_predict(X)
    cluster_labels = kmeans_fit.labels_
    cluster_ids_x, cluster_centers = kmeans(
        X=features, num_clusters=10, distance='cosine', device=torch.device('cuda:0')
    )
    predict_cls = cluster_ids_x.cpu().numpy()
    # clustering = DBSCAN(eps=10, min_samples=2).fit(X)
    # t = clustering.labels_
    print('f')


if __name__ == '__main__':
    cluster()
