from torch import nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv_add = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_add = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.down_sample = down_sample

    def forward(self, x):
        identify = x
        if self.down_sample is not None:
            identify = self.down_sample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_add(x)
        x = self.bn_add(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identify
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, down_sample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel * self.expansion, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def forward(self, x):
        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_class=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Sequential(
                nn.Linear(in_features=512 * block.expansion, out_features=1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=num_class)
            )

    def _make_layer(self, block, channel, block_num, stride=1):
        down_sample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channel, out_channels=channel*block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, channel, down_sample=down_sample, stride=stride))
        self.in_channel = channel * block.expansion
        for i in range(1, block_num):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.average_pool(x)
            x = x.view(x.size()[0], -1)
            x = self.fc(x)
        return x


def resnet34(num_class=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_class=num_class, include_top=include_top)


def resnet152(num_class=1000, include_top=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_class=num_class, include_top=include_top)
