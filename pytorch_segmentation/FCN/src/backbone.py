import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    # 已看過
    # 一般卷積或是膨脹卷積
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # 已看過
    # 升維或降維使用
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    # 通過一個bottleneck在channel維度上面會加深多少倍
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """
        :param inplanes: 輸入channel深度
        :param planes: 瓶頸結構中中間層的channel深度
        :param stride: 步距
        :param downsample: 下採樣一系列層結構
        :param groups: 組卷積數，這裡不會使用組卷積
        :param base_width: 組卷積數，這裡不會使用組卷積
        :param dilation: 膨脹係數
        :param norm_layer: 標準化層
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # width = planes，base_width=64且groups=1
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        # 降維
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 卷積
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 升維
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # 殘差邊
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 已看過
        # 瓶頸結構，根據stride以及dilation會知道高和寬有沒有進行下採樣
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        """
        :param block: 使用到的瓶頸結構
        :param layers: 每個瓶頸結構要堆疊幾層
        :param num_classes: 分類類別數
        :param zero_init_residual: 權重初始化用的
        :param groups: 組卷積，這裡不會使用到組卷積
        :param width_per_group: 組卷積，這裡不會使用到組卷積
        :param replace_stride_with_dilation: 哪幾個部分要使用到膨脹卷積
        :param norm_layer: norm_layer的實例對象，預設為None
        """
        # 已看過
        super(ResNet, self).__init__()
        # 如果沒有標準化層就創建一個BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        # 如果沒有傳入哪些要用膨脹卷積就全部設定為False
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # 最一開始的7*7卷積，圖像會下採樣兩倍
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 第一層殘差結構
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 第二層殘差結構，不使用膨脹卷積
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        # 第三層殘差結構，使用膨脹卷積
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        # 第四層殘差結構，使用膨脹卷積
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # 下面這兩個都會被拿掉，因為這裡不需要做分類
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 設定初始化權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the models by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        :param block: 瓶頸結構
        :param planes: 輸入channel深度
        :param blocks: 堆疊層數
        :param stride: 第一層的步距
        :param dilate: 是否使用膨脹卷積
        :return:
        """
        # 已看過
        # 構建每層殘差結構
        norm_layer = self._norm_layer
        downsample = None
        # 保留一開始的膨脹係數
        previous_dilation = self.dilation
        # 計算下次的膨脹係數
        if dilate:
            self.dilation *= stride
            stride = 1
        # 構建殘差結構的邊，進行下採樣以及channel擴維
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # 第一層的殘差結構
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        # 剩下的殘差結構
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # 已看過
        # See note [TorchScript super()]
        # 輸入 x shape [batch_size, 3, 480, 480]

        # x shape [batch_size, 64, 240, 240]，高寬下採樣兩倍
        x = self.conv1(x)
        # 標準化後接上激活函數
        x = self.bn1(x)
        x = self.relu(x)
        # x shape [batch_size, 64, 120, 120]，高寬下採樣兩倍
        x = self.maxpool(x)

        # x shape [batch_size, 256, 120, 120]
        x = self.layer1(x)
        # x shape [batch_size, 512, 60, 60]
        x = self.layer2(x)
        # x shape [batch_size, 1024, 60, 60]
        x = self.layer3(x)
        # x shape [batch_size, 2048, 60, 60]
        x = self.layer4(x)

        # 這裡會被捨棄掉，我們不會用到
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        # 已看過
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    # 已看過
    model = ResNet(block, layers, **kwargs)
    return model


def resnet50(**kwargs):
    # 已看過
    # **kwargs裡面會有說明哪幾層要使用膨脹卷積
    # 構建resnet50
    r"""ResNet-50 models from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # 傳入每一層要用的基礎層以及堆疊層數以及膨脹卷積要用在哪幾層
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    # 已看過
    # **kwargs裡面會有說明哪幾層要使用膨脹卷積
    # 構建resnet101
    r"""ResNet-101 models from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
