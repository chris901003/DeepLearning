import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential, Flatten
import torch.nn.functional as F
import PIL

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = Sequential(
            nn.Linear(in_features=64*5*5, out_features=512),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc2 = Sequential(
            nn.Linear(in_features=512, out_features=84),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

# train_data = torchvision.datasets.MNIST(root='dataset', train=True, download=True)
#
# v1, v2 = train_data[4]

image_path = './CustomizePicture/123.png'
image = Image.open(image_path)
image = image.convert('L')
image = PIL.ImageOps.invert(image)
# image.show()

transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((28, 28)),
    torchvision.transforms.ToTensor()
])

# print(v2)
# image = v1
# image.show()

image = transform(image)
# print(image.shape)
image = torch.unsqueeze(image, dim=0)
# print(image.shape)

model = torch.load('LeNet_Model/ReLU_CrossEntropy_10_Google', map_location='cpu')
model.eval()

with torch.no_grad():
    y = model(image)
    # print(y.shape)
    output = torch.squeeze(y)
    # print(output)
    predict = torch.softmax(output, dim=0)
    # print(predict)
    predict_cla = torch.argmax(predict).numpy()
print(predict_cla, predict[predict_cla].numpy())
