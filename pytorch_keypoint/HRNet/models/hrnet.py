import torch.nn as nn

BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):
    # resnet的基礎結構
    # expansion為擴張係數
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        :param inplanes: 輸入channel深度
        :param planes: 中間層channel深度
        :param stride: 步距
        :param downsample: 下採樣方式
        """
        super(BasicBlock, self).__init__()
        # 擴維
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 已看過
        # 與resnet相同
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # resnet中的瓶頸結構
    # 輸出channel擴張係數
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        :param inplanes: 輸入channel
        :param planes: 中間層的channel
        :param stride: 步距
        :param downsample: 殘差結構中的下採樣方式
        """
        # 已看過
        super(Bottleneck, self).__init__()
        # 使用1*1卷積進行擴維
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # 通過BN進行標準化
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # 中間層卷積
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # 最後輸出channel會是planes*expansion
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        # 激活函數使用Relu
        self.relu = nn.ReLU(inplace=True)
        # 看是否使用下採樣方式
        self.downsample = downsample
        # 步距
        self.stride = stride

    def forward(self, x):
        # 已看過
        # 與resnet中相同
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        构建对应stage，即用来融合不同尺度的实现
        :param input_branches: 輸入的分支數，每個分支表示一種尺度，也就是下採樣度
        :param output_branches: 輸出的分支數
        :param c: 輸入的第一个分支通道數，也就是基底的channel深度
        """
        # 已看過
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        # 遍歷所有的分支
        for i in range(self.input_branches):  # 每个分支上都先通过4个BasicBlock
            # 每個通道之間channel差別會是2倍
            w = c * (2 ** i)  # 对应第i个分支的通道数
            # 這裡會通過4層的basic block
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            # 放入ModuleList當中
            self.branches.append(branch)

        # 用於融合每個分支上的輸出
        self.fuse_layers = nn.ModuleList()
        # 遍歷要輸出的層數
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    # 当输入、输出为同一个分支时不做任何处理
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    # 当输入分支j大于输出分支i时(即输入分支下采样率大于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及上采样，方便后续相加
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    # 当输入分支j小于输出分支i时(即输入分支下采样率小于输出分支下采样率)，
                    # 此时需要对输入分支j进行通道调整以及下采样，方便后续相加
                    # 注意，这里每次下采样2x都是通过一个3x3卷积层实现的，4x就是两个，8x就是三个，总共i-j个
                    ops = []
                    # 前i-j-1个卷积层不用变通道，只进行下采样
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    # 最后一个卷积层不仅要调整通道，还要进行下采样
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        # 設定激活函數
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 已看過
        # x會是一個list，其中的每一個表示不同下採樣的倍率
        # 每个分支通过对应的block，這些branch不會改變特徵圖的高寬以及深度
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 接着融合不同尺寸信息
        x_fused = []
        # self.fuse_layers長度就是輸出的數量，在最後一個stage只會一個輸出
        for i in range(len(self.fuse_layers)):
            # 做相加後再通過激活函數
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )

        # x_fused = [[batch_size, channel, height, width], ..., [batch_size, channel, height, width]]
        # 長度就是要出書的個數
        return x_fused


class HighResolutionNet(nn.Module):
    # 由train.py構建
    def __init__(self, base_channel: int = 32, num_joints: int = 17):
        # 已看過
        # 在coco數據集中關節點數量為17
        super().__init__()
        # Stem
        # BN_MOMENTUM預設為0.1，這是一個超參數
        # 最一開始的卷積，高寬會少一半
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # 這裡使用的是BN標準化
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # 第二個卷積會將高寬再減半
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        # 用Relu作為激活函數
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        # down sample的模板
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        # layer1層結構，使用bottleneck作為基底
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )

        # Transition 1層結構
        # 會將上層輸出分成兩條路，一條不會繼續下採樣，另一條會再下採樣兩倍
        self.transition1 = nn.ModuleList([
            # 這條路不會再進行下採樣
            nn.Sequential(
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            # 進行兩倍下採樣同時進行兩倍擴維
            nn.Sequential(
                nn.Sequential(  # 这里又使用一次Sequential是为了适配原项目中提供的权重
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage2
        # 在stage2中只需要堆疊一層
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel)
        )

        # transition2
        self.transition2 = nn.ModuleList([
            # 下採樣4倍以及8倍不需做任合動作
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            # 多一個下採樣16倍的，透過下採樣8倍的到的
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage3
        # 在stage3中需要堆疊4層
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel)
        )

        # transition3
        # 可以對照架構圖就可以了
        self.transition3 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage4
        # 注意，最后一个StageModule只输出分辨率最高的特征层
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=1, c=base_channel)
        )

        # Final layer
        # 最後一層將輸出channel變成關節點數量
        self.final_layer = nn.Conv2d(base_channel, num_joints, kernel_size=1, stride=1)

    def forward(self, x):
        # 已看過
        # x shape [batch_size, channel, height, width]
        # 最一開始輸入shape [batch_size, 3, 256, 192]

        # x shape [batch_size, 64, 128, 96]，經過一次下採樣(2倍)
        x = self.conv1(x)
        # 標準化
        x = self.bn1(x)
        # 激活
        x = self.relu(x)
        # x shape [batch_size, 64, 64, 48]，再經過一次下採樣(4倍)
        x = self.conv2(x)
        # 標準化
        x = self.bn2(x)
        # 激活
        x = self.relu(x)

        # x shape [batch_size, 256, 64, 48]
        x = self.layer1(x)
        # x會變成list型態 = [[batch_size, 32, 64, 48], [batch_size, 64, 32, 24]]
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list

        # x shape [[batch_size, 32, 64, 48], [batch_size, 64, 32, 24]]
        x = self.stage2(x)
        # x shape [[batch_size, 32, 64, 48], [batch_size, 64, 32, 24], [batch_size, 128, 16, 12]]
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        # x shape [[batch_size, 32, 64, 48], [batch_size, 64, 32, 24], [batch_size, 128, 16, 12]]
        x = self.stage3(x)
        # x shape [[batch_size, 32, 64, 48], [batch_size, 64, 32, 24],
        # [batch_size, 128, 16, 12], [batch_size, 256, 8, 6]]
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]  # New branch derives from the "upper" branch only

        # x shape [[batch_size, 32, 64, 48]]
        x = self.stage4(x)

        # x shape [batch_size, 關節點數量, 64, 48]
        x = self.final_layer(x[0])

        return x
