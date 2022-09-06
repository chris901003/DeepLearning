import os
from torchvision import models
import torch
from torch import nn
import cv2
import numpy as np


def main():
    num_classes = 4
    pretrained = './best_model.pkt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet34(pretrained=True)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(in_features=fc_inputs, out_features=num_classes)
    model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    model = model.to(device)
    model.eval()
    mean = np.array([127.5, 127.5, 127.5], dtype=np.float32).reshape(1, 1, -1)
    std = np.array([127.5, 127.5, 127.5], dtype=np.float32).reshape(1, 1, -1)
    img_path = '/Users/huanghongyan/Documents/DeepLearning/pytorch_gan/MMGeneration_StyleGan2_Milk/Image/0.jpg'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    img = (img - mean) / std
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)
    img = img.unsqueeze(dim=0)
    with torch.no_grad():
        output = model(img)
    output = torch.softmax(output, dim=1)
    pred = torch.argmax(output, dim=1).item()
    output = output.cpu().detach().numpy()[0]
    score = output[pred]
    print(f'Classes = {pred}, Score = {round(score * 100, 2)}')


if __name__ == '__main__':
    main()
