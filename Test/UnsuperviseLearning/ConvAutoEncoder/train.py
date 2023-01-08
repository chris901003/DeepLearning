import torch
from torch import nn
from functools import cmp_to_key
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
from kmeans_pytorch import kmeans
from PIL import Image
from sklearn.cluster import DBSCAN
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
        # 48x48
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        # 24x24
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        # 12x12
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        # 6x6
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        # 6x6
        # self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

        self.act = nn.ReLU()

    def forward(self, x, feature=False):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = x.reshape(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvAutoEncoder()
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    from torchvision import datasets
    train_data = datasets.STL10('data', split='train', transform=transforms.ToTensor())
    test_data = datasets.STL10('data', split='test', transform=transforms.ToTensor())
    num_workers = 0
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    n_epochs = 1000

    for epoch in range(1, n_epochs + 1):
        train_loss = 0.0
        correct = 0.0
        for idx, data in enumerate(train_loader):
            images, label = data
            images = images.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += torch.eq(outputs.argmax(dim=1), label).sum() / label.size(0)
            # if idx == 30:
            #     transformImage = transforms.ToPILImage()
            #     img = transformImage(outputs[0])
            #     img.show()
            #     img1 = transformImage(images[0])
            #     img1.show()

        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch,
            train_loss
        ))
        print(f'Acc {correct / len(train_loader)}')
        # torch.save(model, 'pretrained2.pth')


def cluster():
    model = ConvAutoEncoder()
    model.load_state_dict(torch.load('pretrained2.pth', map_location='cpu').state_dict())
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
        featuresList.append(outputs)
    features = torch.concat(featuresList, dim=0)
    cluster_ids_x, cluster_centers = kmeans(
        X=features, num_clusters=10, distance='euclidean', device=torch.device('cuda:0')
    )
    predict_cls = cluster_ids_x.cpu().numpy()
    X = torch.nn.functional.softmax(features)
    X = X.cpu().numpy()
    clustering = DBSCAN(eps=0.3, min_samples=2).fit(X)
    t = clustering.labels_
    clustering_kmeans = KMeans(n_clusters=10).fit(X)
    t1 = clustering_kmeans.labels_
    print('f')


if __name__ == '__main__':
    cluster()
