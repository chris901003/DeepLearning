# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential

from mmocr.models.builder import BACKBONES


@BACKBONES.register_module()
class VeryDeepVgg(BaseModule):
    """Implement VGG-VeryDeep backbone for text recognition, modified from
    `VGG-VeryDeep <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        leaky_relu (bool): Use leakyRelu or not.
        input_channels (int): Number of channels of input image tensor.
    """

    def __init__(self,
                 leaky_relu=True,
                 input_channels=3,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Uniform', layer='BatchNorm2d')
                 ]):
        """ 已看過，構建專門設計給CRNN的VGG特徵提取網路
        Args:
            leaky_relu: 是否將激活函數從ReLU換成leaky_relu激活函數
            input_channels: 輸入的channel深度
            init_cfg: 初始化config設定方式
        """
        # 繼承自BaseModule，對繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)

        # 這裡會有7層的卷積結構，以下紀錄的每一層的配置信息
        # ks = kernel_size
        ks = [3, 3, 3, 3, 3, 3, 2]
        # ps = padding
        ps = [1, 1, 1, 1, 1, 1, 0]
        # ss = stride
        ss = [1, 1, 1, 1, 1, 1, 1]
        # nm = out_channels
        nm = [64, 128, 256, 256, 512, 512, 512]

        # 將nm資料保存
        self.channels = nm

        # cnn = nn.Sequential()
        # 創建一連串的cnn層結構
        cnn = Sequential()

        def conv_relu(i, batch_normalization=False):
            # 卷積加上激活層，會根據傳入的i構建對應的卷積參數，第二個部分會是決定是否使用標準化層結構
            # 輸入的channel深度就會是上一層的輸出channel深度，如果是第一層就直接拿input_channels
            n_in = input_channels if i == 0 else nm[i - 1]
            # 獲取輸出的channel深度
            n_out = nm[i]
            # 透過add_module將層結構出入進去
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i]))
            if batch_normalization:
                # 如果有需要標準化層結構就會到這裡添加上去
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(n_out))
            if leaky_relu:
                # 如果選擇的激活函數式leaky_relu就會到這裡
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                # 否則就會直接使用ReLU作為激活函數
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # 透過conv_relu構建卷積加上激活函數，裡面傳的數字表示第幾個卷積層結構的資訊
        conv_relu(0)
        # 會透過最大池化進行2倍下採樣
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        # 以下就是重複進行堆疊
        conv_relu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        conv_relu(2, True)
        conv_relu(3)
        # 這裡的下採樣方式有點不同稍微注意一下
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        conv_relu(4, True)
        conv_relu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv_relu(6, True)  # 512x1x16

        # 最終將cnn多層結構保存下來
        self.cnn = cnn

    def out_channels(self):
        return self.channels[-1]

    def forward(self, x):
        """
        Args:
            x (Tensor): Images of shape :math:`(N, C, H, W)`.

        Returns:
            Tensor: The feature Tensor of shape :math:`(N, 512, H/32, (W/4+1)`.
        """
        # 已看過，直接進行向前傳遞
        # output shape = [batch_size, channel, height, width]
        output = self.cnn(x)

        # 將最後的特徵圖進行回傳
        return output
