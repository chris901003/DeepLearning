import torch
from torch import nn
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_features=28*28*64, out_features=10)

    def forward(self, images, labels):
        output = self.conv(images)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)
        output = output + labels
        return output


def main():
    net = Net()
    net = net.eval()
    x = torch.randn(1, 3, 28, 28)
    y = torch.tensor([3])
    onnx_save_path = 'test.onnx'
    torch.onnx.export(net, (x, y), onnx_save_path, input_names=['x', 'y'], output_names=['pred'])


if __name__ == '__main__':
    print('Testing onnx create')
    main()
