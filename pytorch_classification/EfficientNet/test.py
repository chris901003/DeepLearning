import torch
import torchvision
from model import resnet34
from torchvision import models
from PIL import Image
from torchvision import transforms
from torch import nn
import json
import time

net_1 = models.resnet34(pretrained=False)
fc_inputs = net_1.fc.in_features
net_1.fc = nn.Sequential(
    nn.Linear(in_features=fc_inputs, out_features=200)
)

net_1.load_state_dict(torch.load('Model/TransferLearning_224.pkl', map_location='cpu'))
net_2 = resnet34(num_class=200)
net_2.load_state_dict(torch.load('Model/AddLayer.pkl', map_location='cpu'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_1 = net_1.to(device)
net_2 = net_2.to(device)

net_3 = models.efficientnet_b0(pretrained=False)
net_3.classifier[1] = nn.Linear(in_features=1280, out_features=200)
net_3.load_state_dict(torch.load('Model/EfficientNet.pkl', map_location='cpu'))
net_3 = net_3.to(device)

transform_data = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
])

image_path = '../CustomizePicture/123.png'
image = Image.open(image_path)
image = transform_data(image)
image = torch.unsqueeze(image, dim=0)
image = torch.cat((image, image))

with open('label_table.json') as f:
    label_table = json.load(f)

start = time.time()
net_1.eval()
with torch.no_grad():
    y = net_1(image)
    output = torch.squeeze(y)
    predict = torch.softmax(output, dim=1)
    predict_cla = torch.argmax(predict, dim=1)
    print(label_table[str(predict_cla[0].item())], str(round(predict[0][predict_cla[0]].item() * 100, 2)) + '%')
end = time.time()
print(f'Template pending time = {end-start}')

start = time.time()
net_2.eval()
with torch.no_grad():
    y = net_2(image)
    output = torch.squeeze(y)
    predict = torch.softmax(output, dim=1)
    predict_cla = torch.argmax(predict, dim=1)
    print(label_table[str(predict_cla[0].item())], str(round(predict[0][predict_cla[0]].item() * 100, 2)) + '%')
end = time.time()
print(f'Add layer pending time = {end-start}')

start = time.time()
net_3.eval()
with torch.no_grad():
    y = net_3(image)
    output = torch.squeeze(y)
    predict = torch.softmax(output, dim=1)
    predict_cla = torch.argmax(predict, dim=1)
    print(label_table[str(predict_cla[0].item())], str(round(predict[0][predict_cla[0]].item() * 100, 2)) + '%')
end = time.time()
print(f'Efficient net pending time = {end-start}')
