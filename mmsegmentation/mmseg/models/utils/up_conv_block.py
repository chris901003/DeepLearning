# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_upsample_layer


class UpConvBlock(nn.Module):
    """Upsample convolution block in decoder for UNet.

    This upsample convolution block consists of one upsample module
    followed by one convolution block. The upsample module expands the
    high-level low-resolution feature map and the convolution block fuses
    the upsampled high-level low-resolution feature map and the low-level
    high-resolution feature map from encoder.

    Args:
        conv_block (nn.Sequential): Sequential of convolutional layers.
        in_channels (int): Number of input channels of the high-level
        skip_channels (int): Number of input channels of the low-level
        high-resolution feature map from encoder.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers in the conv_block.
            Default: 2.
        stride (int): Stride of convolutional layer in conv_block. Default: 1.
        dilation (int): Dilation rate of convolutional layer in conv_block.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv'). If the size of
            high-level feature map is the same as that of skip feature map
            (low-level feature map from encoder), it does not need upsample the
            high-level feature map and the upsample_cfg is None.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 conv_block,
                 in_channels,
                 skip_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 dcn=None,
                 plugins=None):
        """
        :param conv_block: BasicConvBlock Class，在unet.py當中
        :param in_channels: 輸入的channel深度
        :param skip_channels: 混合較低層次的特徵層的channel深度
        :param out_channels: 輸出的channel深度
        :param num_convs: 卷積需要堆疊多少次
        :param stride: 步距
        :param dilation: 膨脹係數
        :param with_cp: 是否使用checkpoint
        :param conv_cfg: 卷積層的設定，預設為None
        :param norm_cfg: 使用的標準化層
        :param act_cfg: 使用的激活函數
        :param upsample_cfg: 上採樣的方式
        :param dcn: dcn卷積，這裡沒有實現，不可以使用
        :param plugins: 這裡沒有實現，不可以使用
        """
        # 已看過
        # 這裡繼承的就是torch.nn.Module，也就是最底層了
        # 這個class是專門給unet的upsample專用的，其他模型上面不會使用到

        super(UpConvBlock, self).__init__()
        # dcn以及plugins都沒有實現，所以這裡如果用到會報錯
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        # 將參數丟入到conv_block當中構建conv_block
        self.conv_block = conv_block(
            in_channels=2 * skip_channels,
            out_channels=out_channels,
            num_convs=num_convs,
            stride=stride,
            dilation=dilation,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dcn=None,
            plugins=None)
        if upsample_cfg is not None:
            # 如果需要上採樣就會到這裡來
            # self.upsample = 上採樣實例對象
            self.upsample = build_upsample_layer(
                cfg=upsample_cfg,
                in_channels=in_channels,
                out_channels=skip_channels,
                with_cp=with_cp,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            # 如果不需要進行上採樣就到這裡使用一般卷積
            # self.upsample = 上採樣實例對象
            self.upsample = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, skip, x):
        """Forward function."""
        # 已看過
        # skip = encoder的特徵層，會與上一層decoder的輸出進行融合

        # 會需要先將上層decoder的輸出進行上採樣，這樣高寬才可以對上
        x = self.upsample(x)
        # 將encoder與當前decoder進行維度上面的拼接，這裡channel會有變化其他不會
        out = torch.cat([skip, x], dim=1)
        # 透過一個卷積模塊將channel變回下層需要的channel同時也進有融合的工作
        out = self.conv_block(out)

        return out
