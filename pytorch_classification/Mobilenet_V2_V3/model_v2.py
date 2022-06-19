from torch import nn
import torch


def _make_divisible(ch, divisor=8, min_ch=None):
    # 已看過
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    # 用來對輸入的數字變成最接近8的倍數的數字
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        # 已看過
        # 構建卷積加上標準化加上激活函數
        padding = (kernel_size - 1) // 2
        # 當groups大小與輸入channel相同時就會使用dw卷積
        # 這裡使用的激活函數是Relu6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    # 到殘差結構
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        # 已看過
        # expand_ratio中間層的channel深度
        super(InvertedResidual, self).__init__()
        # 計算中間層channel深度
        hidden_channel = in_channel * expand_ratio
        # 只有當輸入channel與輸出channel深度相同且卷積核步距為1時才會有捷徑分支
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        if expand_ratio != 1:
            # 1x1 pointwise conv，先升維
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv，進行特徵提取，dw卷積在這裡
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)，再降維
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        # 輸出成Sequential
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # 已看過
        # 如果有捷徑分支就加上去
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    # 搭建完整模型
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        # 已看過
        # alpha = 在基礎channel深度上做倍率調整
        # round_nearset = 將各層的channel深度變成round_nearset的倍數
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        # 初始深度以及最後特徵提取完的深度
        input_channel = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        # 多層到殘差結構的超參數
        # t = 中間channel深度要是輸入channel的多少倍
        # c = 輸入channel
        # n = 要重複堆疊幾次
        # s = 第一層的步距
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                # stride只會影響第一層
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))
        # combine feature layers
        # 將特徵提取層放入Sequential中
        self.features = nn.Sequential(*features)

        # 進入分類層
        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # 初始化權重
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 已看過
        # 就像前傳播
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
