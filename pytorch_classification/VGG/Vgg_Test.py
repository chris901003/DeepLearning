import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models
from torch import nn
from PIL import Image

transform_data = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

net = models.vgg16(pretrained=True)
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load('Model/vgg16cifar100_34.pkl', map_location='cpu'))
net.to(device)


def picture_chart(idx):
    data = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
            'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
            'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 'cup', 'dinosaur', 'dolphin',
            'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp',
            'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
            'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
            'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
            'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
            'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
    return data[idx]


image_path = '../CustomizePicture/shark.png'
image = Image.open(image_path)
image = transform_data(image)
image = torch.unsqueeze(image, dim=0)
image = torch.cat((image, image))


net.eval()
with torch.no_grad():
    y = net(image)
    output = torch.squeeze(y)
    predict = torch.softmax(output, dim=1)
    predict_cla = torch.argmax(predict, dim=1)
    print(picture_chart(predict_cla[0].item()), str(round(predict[0][predict_cla[0]].item() * 100, 2)) + '%')
