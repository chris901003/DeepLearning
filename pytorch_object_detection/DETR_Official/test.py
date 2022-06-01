import torch
from torch import nn


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=1, stride=1)

    def forward(self, x):
        return self.conv1(x)


class Backbone2(nn.Module):
    def __init__(self):
        super(Backbone2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=100, out_channels=10, kernel_size=1, stride=1)

    def forward(self, x):
        return self.conv1(x)


class Combine(nn.Sequential):
    def __init__(self, backbone1, backbone2):
        super(Combine, self).__init__(backbone1, backbone2)

    def forward(self, x):
        return self[1](x)


b1 = Backbone()
b2 = Backbone2()
c = Combine(b1, b2)
a = torch.randn((1, 100, 10, 10))
b = c(a)
print(b.shape)
