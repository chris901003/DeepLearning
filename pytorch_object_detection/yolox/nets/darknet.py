#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
from torch import nn


class SiLU(nn.Module):
    # 已看過
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    # 獲取指定的激活函數
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class Focus(nn.Module):
    # 已看過
    # 這裡就是YOLOv5特別提出來的部分，有點Swim Transformer的感覺，跨像素提取，高寬少一半通道數乘以4倍
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # 直接暴力拆解
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        return self.conv(x)


class BaseConv(nn.Module):
    # 已看過
    # 構建基礎捲積(Conv + BN + Activation)
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        # groups == 1 時就是普通卷積
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        # 沒有BN層
        return self.act(self.conv(x))


class DWConv(nn.Module):
    # 已看過
    # DW卷積，這裏沒有用到所以不用管，除非要把backbone改成mobilenet的話就會需要用到
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class SPPBottleneck(nn.Module):
    # 已看過
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        # 先透過1 * 1的卷積讓channel縮減成一半
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        # 透過循環弄出3個kernel不同的池化層
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        # pooling後的channel要拼接再一起，還有一個是什麼都不做直接拿conv1的結果進行拼接
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class Bottleneck(nn.Module):
    # --------------------------------------------------#
    #   残差结构的构建，小的残差结构
    # --------------------------------------------------#
    # 注意一下這裡是小殘差結構
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        # 這裡基本跟yolov3, v4一樣
        # --------------------------------------------------#
        #   利用1x1卷积进行通道数的缩减。缩减率一般是50%
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   利用3x3卷积进行通道数的拓张。并且完成特征提取
        # --------------------------------------------------#
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    # 已看過
    # 這裡構建的殘差結構就是大殘差結構
    # YOLOv5提出的
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        # 主幹與大殘差邊都會需要先經過一次卷積
        # 而且第一次卷機會讓channel減半
        # --------------------------------------------------#
        #   主干部分的初次卷积
        # --------------------------------------------------#
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # --------------------------------------------------#
        #   大的残差边部分的初次卷积
        # --------------------------------------------------#
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        # 堆疊後channel會變成2 * hidden_channels
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

        # --------------------------------------------------#
        #   根据循环的次数构建上述Bottleneck残差结构
        # --------------------------------------------------#
        # 用在主幹上面的多層小殘差結構
        # 這裡論文有些道我們沒有用bottleneck結構所以在裡面不會有降維操作，所以會是1.0
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0,
                                  depthwise, act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        # -------------------------------#
        #   x_1是主干部分
        # -------------------------------#
        x_1 = self.conv1(x)
        # -------------------------------#
        #   x_2是大的残差边部分
        # -------------------------------#
        x_2 = self.conv2(x)

        # -----------------------------------------------#
        #   主干部分利用残差结构堆叠继续进行特征提取
        # -----------------------------------------------#
        x_1 = self.m(x_1)
        # -----------------------------------------------#
        #   主干部分和大的残差边部分进行堆叠
        # -----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        # -----------------------------------------------#
        #   对堆叠的结果进行卷积的处理
        # -----------------------------------------------#
        # 最後再通過一次卷積完成CSP結構
        return self.conv3(x)


class CSPDarknet(nn.Module):
    # 已看過
    # 構建網路
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu",):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        # dep_mul以及wid_mul預設都是1

        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#
        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        
        # -----------------------------------------------#
        #   利用focus网络结构进行特征提取
        #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
        # -----------------------------------------------#
        # focus裡面會先將圖像大小除以2然後channel翻4倍
        # 之後再通過卷積，卷積出來的channel就是base_channel
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # 基本上dark2, 3, 4, 5都是差不多的
        # 就是先下採樣之後再透過CSPLayer
        # -----------------------------------------------#
        #   完成卷积之后，320, 320, 64 -> 160, 160, 128
        #   完成CSPlayer之后，160, 160, 128 -> 160, 160, 128
        # -----------------------------------------------#
        # CSPLayer中的n就表示小殘差邊要輪幾次
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        # -----------------------------------------------#
        #   完成卷积之后，160, 160, 128 -> 80, 80, 256
        #   完成CSPlayer之后，80, 80, 256 -> 80, 80, 256
        # -----------------------------------------------#
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # -----------------------------------------------#
        #   完成卷积之后，80, 80, 256 -> 40, 40, 512
        #   完成CSPlayer之后，40, 40, 512 -> 40, 40, 512
        # -----------------------------------------------#
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        # -----------------------------------------------#
        #   完成卷积之后，40, 40, 512 -> 20, 20, 1024
        #   完成SPP之后，20, 20, 1024 -> 20, 20, 1024
        #   完成CSPlayer之后，20, 20, 1024 -> 20, 20, 1024
        # -----------------------------------------------#
        # 這裡有多一個SPP結構喔
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        # -----------------------------------------------#
        #   dark3的输出为80, 80, 256，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark3(x)
        outputs["dark3"] = x
        # -----------------------------------------------#
        #   dark4的输出为40, 40, 512，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark4(x)
        outputs["dark4"] = x
        # -----------------------------------------------#
        #   dark5的输出为20, 20, 1024，是一个有效特征层
        # -----------------------------------------------#
        x = self.dark5(x)
        outputs["dark5"] = x
        # dark3, 4, 5後面都會用到所以回傳回去
        return {k: v for k, v in outputs.items() if k in self.out_features}


if __name__ == '__main__':
    print(CSPDarknet(1, 1))
