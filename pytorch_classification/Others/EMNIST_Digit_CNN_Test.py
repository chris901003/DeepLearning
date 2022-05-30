import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from alive_progress import alive_bar
import PIL
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
from torchvision import transforms


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

file_name = 'CustomizePicture/0.png'  # 導入自己的圖片
img = Image.open(file_name)
img = img.convert('L')

img = PIL.ImageOps.invert(img)
img = img.transpose(Image.FLIP_LEFT_RIGHT)
img = img.rotate(90)

# plt.imshow(img)
# plt.show()

train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

img = train_transform(img)
img = torch.unsqueeze(img, dim=0)

model = CNN()
model = torch.load('EMNIST_Digit_CNN/Model_10', map_location='cpu')
model.eval()


with torch.no_grad():
    y = model(img)
    # print(y)
    # print(y.shape)
    output = torch.squeeze(y)
    # print(output)
    # print(output)
    predict = torch.softmax(output, dim=0)
    # predict = output
    # print(predict)
    predict_cla = torch.argmax(predict).numpy()
print(predict_cla, predict[predict_cla].numpy())
