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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # 輸入通道數
                out_channels=16,  # 輸出通道數
                kernel_size=5,  # 卷積核大小
                stride=1,  # 卷積步數
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

file_name = 'CustomizePicture/8.png'  # 導入自己的圖片
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

model = Net()
model = torch.load('EMNIST_Bymerge_Model/Model_10', map_location='cpu')
model.eval()

def get_mapping(num, with_type='bymerge'):
    """
    根據 mapping，由傳入的 num 計算 UTF8 字符。
    """
    if with_type == 'byclass':
        if num <= 9:
            return chr(num + 48)  # 數字
        elif num <= 35:
            return chr(num + 55)  # 大寫字母
        else:
            return chr(num + 61)  # 小寫字母
    elif with_type == 'letters':
        return chr(num + 64) + " / " + chr(num + 96)  # 大寫/小寫字母
    elif with_type == 'digits':
        return chr(num + 96)
    elif with_type == 'bymerge':
        if num <= 9:
            return chr(num + 48)
        elif num <= 35:
            return chr(num + 55)
        elif num == 36:
            return chr(97)
        elif num == 37:
            return chr(98)
        elif num == 38:
            return chr(100)
        elif num == 39:
            return chr(101)
        elif num == 40:
            return chr(102)
        elif num == 41:
            return chr(103)
        elif num == 42:
            return chr(104)
        elif num == 43:
            return chr(110)
        elif num == 44:
            return chr(113)
        elif num == 45:
            return chr(114)
        elif num == 46:
            return chr(116)
    else:
        return num


img = torch.cat((img, img))

with torch.no_grad():
    y = model(img)
    # print(y)
    # print(y.shape)
    output = torch.squeeze(y)
    # print(output.shape)
    # print(output)
    predict = torch.softmax(output, dim=1)
    # predict = output
    print(predict[0] == predict[1])
    predict_cla = torch.argmax(predict, dim=1)
    print(predict_cla)
    print(predict[0][predict_cla[0]], predict[1][predict_cla[1]])
    # predict_cla = torch.argmax(predict).numpy()
# print(get_mapping(predict_cla), predict[predict_cla].numpy())
