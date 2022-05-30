import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from alive_progress import alive_bar
import PIL
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
from torchvision import transforms

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.run = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.run(x, None)
        out = self.out(r_out[:, -1, :])
        return out

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

model = RNN()
model = torch.load('EMNIST_Digit_RNN/Model_10', map_location='cpu')
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

