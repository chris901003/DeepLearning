import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_downsample=False):
        """
        :param c_in: 輸入channel
        :param c_out: 輸出channel
        :param is_downsample: 是否使用下採樣
        """
        # 已看過
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        # 判斷是否要下採樣
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        # 使用BN來進行標準化
        self.bn1 = nn.BatchNorm2d(c_out)
        # 使用Relu當作激活函數
        self.relu = nn.ReLU(True)
        # 普通卷積
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            # 如果有下採樣，那在殘差結構上面也會需要下採樣
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(c_out)
            )
        elif c_in != c_out:
            # 如果輸入與輸出的channel不同，那在殘差結構上面需要對channel進行擴維
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=1, bias=False),
                nn.BatchNorm2d(c_out)
            )
            self.is_downsample = True

    def forward(self, basic_in):
        # 已看過
        basic_out = self.conv1(basic_in)
        basic_out = self.bn1(basic_out)
        basic_out = self.relu(basic_out)
        basic_out = self.conv2(basic_out)
        basic_out = self.bn2(basic_out)
        if self.is_downsample:
            basic_in = self.downsample(basic_in)
        return F.relu(basic_in.add(basic_out), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    # 已看過
    # 構建多層層結構
    blocks = []
    for i in range(repeat_times):
        # 如果需要下採樣就只有在第一層會進行下採樣，剩下的就不會下採樣了
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample), ]
        else:
            blocks += [BasicBlock(c_out, c_out), ]
    return nn.Sequential(*blocks)


class Net(nn.Module):
    def __init__(self, num_classes=751, reid=False):
        """
        :param num_classes: 分類類別數，預設會是751表示數據集裡面有751種不同的人
        :param reid: 預設會是False
        """
        # 已看過
        super(Net, self).__init__()
        # 下面的標示都沒有batch_size維度
        # [3, 128, 64] -> [64, 64, 32]
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(32,32,3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),
        )
        # make_layers(in_channel, out_channel, repeat_time, down_sample)
        # [64, 64, 32] -> [64, 64, 32]
        self.layer1 = make_layers(64, 64, 2, False)
        # [64, 64, 32] -> [128, 32, 16]
        self.layer2 = make_layers(64, 128, 2, True)
        # [128, 32, 16] -> [256, 16, 8]
        self.layer3 = make_layers(128, 256, 2, True)
        # [256, 16, 8] -> [512, 8, 4]
        self.layer4 = make_layers(256, 512, 2, True)
        # [512, 8, 4] -> [512, 8 * 4]
        self.avgpool = nn.AvgPool2d((8, 4), 1)
        # reid預設為False
        self.reid = reid

        # 製造分類頭
        # [512, 8 * 4] -> [num_classes]
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        # 已看過
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x shape [batch_size, channel, height, width]
        x = self.avgpool(x)
        # x shape [batch_size, channel * height * width]
        x = x.view(x.size(0), -1)
        # B x 128
        if self.reid:
            # norm = 求范數
            # x.norm shape [batch_size, 1]
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            # x shape [batch_size, channel * height * width]
            return x
        # classifier
        x = self.classifier(x)
        # x shape [batch_size, num_classes]
        return x


if __name__ == '__main__':
    # 已看過
    # 測試使用
    net = Net()
    x = torch.randn(4, 3, 128, 64)
    y = net(x)
    print(y.shape)
