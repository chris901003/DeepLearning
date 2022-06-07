import torch
from collections import defaultdict
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


def update(**kwargs):
    print(kwargs)


class SmoothedValue(object):
    def __init__(self):
        self.tot = 0

    def update(self, value, n=1):
        self.tot += value


meters = defaultdict(SmoothedValue)
meters['score'].update(10)
meters['score'].update(1000)
meters['total_score'] = 100
meters['total_score'] = 10000
print(meters['score'])
