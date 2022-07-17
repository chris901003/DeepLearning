# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (UPSAMPLE_LAYERS, ConvModule, build_activation_layer,
                      build_norm_layer)
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.ops import Upsample
from ..builder import BACKBONES
from ..utils import UpConvBlock


class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        """
        :param in_channels: 輸入的channel深度
        :param out_channels: 輸出的channel深度
        :param num_convs: 卷積層堆疊數量
        :param stride: 步距
        :param dilation: 膨脹係數
        :param with_cp: 是否使用checkpoint
        :param conv_cfg: 卷積相關設定
        :param norm_cfg: 標準化層相關設定
        :param act_cfg: 使用的激活函數
        :param dcn: dcn相關內容，這裡沒有實作所以無法使用
        :param plugins: 這裡沒有實作所以無法使用
        """
        # 已看過

        # 這裡繼承於torch.nn.Module
        super(BasicConvBlock, self).__init__()
        # dcn與plugins都沒有實作，所以如果使用的話會報錯
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        # 卷積
        convs = []
        # 重複需要的堆疊層數
        for i in range(num_convs):
            convs.append(
                # 這裡調用MMCV當中構建卷積加標準化加激活類
                ConvModule(
                    # 只有在第一層的輸入channel深度會是in_channels，其他層就會是out_channels
                    # 因為第一層就會將channel變成out_channel
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    # 只有第一層的stride會是指定的stride其他都會是1，這樣最多只會下採樣一次
                    stride=stride if i == 0 else 1,
                    # 第一層的dilation以及padding都一定會是1其他層就依據輸入給定
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        # 用nn.Sequential將convs給到self.convs
        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""
        # 已看過

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            # 基本上會直接用這裡，out就是已經通過卷積以及標準化以及激活的結果
            out = self.convs(x)
        return out


@UPSAMPLE_LAYERS.register_module()
class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 *,
                 kernel_size=4,
                 scale_factor=2):
        super(DeconvModule, self).__init__()

        assert (kernel_size - scale_factor >= 0) and\
               (kernel_size - scale_factor) % 2 == 0,\
               f'kernel_size should be greater than or equal to scale_factor '\
               f'and (kernel_size - scale_factor) should be even numbers, '\
               f'while the kernel size is {kernel_size} and scale_factor is '\
               f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.with_cp = with_cp
        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        activate = build_activation_layer(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.deconv_upsamping, x)
        else:
            out = self.deconv_upsamping(x)
        return out


@UPSAMPLE_LAYERS.register_module()
class InterpConv(nn.Module):
    """Interpolation upsample module in decoder for UNet.

    This module uses interpolation to upsample feature map in the decoder
    of UNet. It consists of one interpolation upsample layer and one
    convolutional layer. It can be one interpolation upsample layer followed
    by one convolutional layer (conv_first=False) or one convolutional layer
    followed by one interpolation upsample layer (conv_first=True).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        conv_first (bool): Whether convolutional layer or interpolation
            upsample layer first. Default: False. It means interpolation
            upsample layer followed by one convolutional layer.
        kernel_size (int): Kernel size of the convolutional layer. Default: 1.
        stride (int): Stride of the convolutional layer. Default: 1.
        padding (int): Padding of the convolutional layer. Default: 1.
        upsample_cfg (dict): Interpolation config of the upsample layer.
            Default: dict(
                scale_factor=2, mode='bilinear', align_corners=False).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 *,
                 conv_cfg=None,
                 conv_first=False,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 upsample_cfg=dict(
                     scale_factor=2, mode='bilinear', align_corners=False)):
        super(InterpConv, self).__init__()

        self.with_cp = with_cp
        conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        upsample = Upsample(**upsample_cfg)
        if conv_first:
            self.interp_upsample = nn.Sequential(conv, upsample)
        else:
            self.interp_upsample = nn.Sequential(upsample, conv)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.interp_upsample, x)
        else:
            out = self.interp_upsample(x)
        return out


@BACKBONES.register_module()
class UNet(BaseModule):
    """UNet backbone.

    This backbone is the implementation of `U-Net: Convolutional Networks
    for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_.

    Args:
        in_channels (int): Number of input image channels. Default" 3.
        base_channels (int): Number of base channels of each stage.
            The output channels of the first stage. Default: 64.
        num_stages (int): Number of stages in encoder, normally 5. Default: 5.
        strides (Sequence[int 1 | 2]): Strides of each stage in encoder.
            len(strides) is equal to num_stages. Normally the stride of the
            first stage in encoder is 1. If strides[i]=2, it uses stride
            convolution to downsample in the correspondence encoder stage.
            Default: (1, 1, 1, 1, 1).
        enc_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence encoder stage.
            Default: (2, 2, 2, 2, 2).
        dec_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence decoder stage.
            Default: (2, 2, 2, 2).
        downsamples (Sequence[int]): Whether use MaxPool to downsample the
            feature map after the first stage of encoder
            (stages: [1, num_stages)). If the correspondence encoder stage use
            stride convolution (strides[i]=2), it will never use MaxPool to
            downsample, even downsamples[i-1]=True.
            Default: (True, True, True, True).
        enc_dilations (Sequence[int]): Dilation rate of each stage in encoder.
            Default: (1, 1, 1, 1, 1).
        dec_dilations (Sequence[int]): Dilation rate of each stage in decoder.
            Default: (1, 1, 1, 1).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Notice:
        The input image size should be divisible by the whole downsample rate
        of the encoder. More detail of the whole downsample rate can be found
        in UNet._check_input_divisible.
    """

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 norm_eval=False,
                 dcn=None,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None):
        """
        :param in_channels: 輸入的channel深度
        :param base_channels: 基礎channel
        :param num_stages: encoder的層數
        :param strides: 每層encoder的步距，len(strides)=num_stages
        :param enc_num_convs: 每層的encoder裏面要堆疊多少層(encoder=下採樣層)
        :param dec_num_convs: 每層的decoder裏面要堆疊多少層(decoder=上採樣層)
        :param downsamples: 每層encoder後是否需要進行下採樣，如果該層的strides=2就會用卷積進行下採樣，否則就會用MaxPool
        :param enc_dilations: encoder當中卷積的膨脹係數
        :param dec_dilations: decoder當中卷積的膨脹係數
        :param with_cp: 是否使用checkpoint，如果設定成有的話就會自動凍結部分層結構，可以節省記憶體使用量
        :param conv_cfg: 對於conv層結構的設定，預設為None
        :param norm_cfg: 標準化層結構
        :param act_cfg: 激活函數選擇
        :param upsample_cfg: decoder中進行upsample時的設定
        :param norm_eval: 是否要將標準化層設定成驗證模式，設定成驗證模式就會凍結均值以及方差，這裡預設為False
        :param dcn: 是否啟用dcn卷積，這裡默認為False
        :param plugins:
        :param pretrained: 模型預訓練權重地址(已經廢棄，直接將預訓練權重放到init_cfg當中)
        :param init_cfg: 初始化設定的config，如果有預訓練權重就直接放這裡來，pretrained已經廢棄
        """
        # 已看過

        # 這裡也是繼承於BaseModule
        super(UNet, self).__init__(init_cfg)

        # 將pretrained保存
        self.pretrained = pretrained
        # init_cfg與pretrained不可以同時設定，否則這裡會直接報錯
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        # pretrained已經被廢棄，要改用init_cfg，也就是預訓練權重要直接放到init_cfg當中
        if isinstance(pretrained, str):
            # 這裡只會給一個warning不會中斷訓練
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            # 將pretrained的內容放到init_cfg當中，這樣才符合新版本的設定
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                # 如果沒有給定pretrained也沒有init_cfg，就在這裡使用一些比較有效率的隨機生成權重的方式
                self.init_cfg = [
                    # 卷積層就使用Kaiming隨機生成權重
                    dict(type='Kaiming', layer='Conv2d'),
                    # 標準化層就全部設定成1
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            # 其他狀況就直接報錯
            raise TypeError('pretrained must be a str or None')

        # 正常的config文件配置在下面檢查都不會出問題

        # 如果需要使用dcn會報錯，這裡還沒有實現dcn的功能
        assert dcn is None, 'Not implemented yet.'
        # 如果需要使用plugins會報錯，這裡還沒有實現plugins功能
        assert plugins is None, 'Not implemented yet.'
        # len(strides)必須與num_stages相同，否則這裡會報錯
        assert len(strides) == num_stages, \
            'The length of strides should be equal to num_stages, '\
            f'while the strides is {strides}, the length of '\
            f'strides is {len(strides)}, and the num_stages is '\
            f'{num_stages}.'
        # len(enc_num_convs)必須與num_stages相同，否則這裡會報錯
        assert len(enc_num_convs) == num_stages, \
            'The length of enc_num_convs should be equal to num_stages, '\
            f'while the enc_num_convs is {enc_num_convs}, the length of '\
            f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        # len(dec_num_convs)會比num_stages少一層，可以去看架構圖就可以對照
        assert len(dec_num_convs) == (num_stages-1), \
            'The length of dec_num_convs should be equal to (num_stages-1), '\
            f'while the dec_num_convs is {dec_num_convs}, the length of '\
            f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        # len(downsamples)會比num_stages少一層，因為最後一層不會需要再下採樣
        assert len(downsamples) == (num_stages-1), \
            'The length of downsamples should be equal to (num_stages-1), '\
            f'while the downsamples is {downsamples}, the length of '\
            f'downsamples is {len(downsamples)}, and the num_stages is '\
            f'{num_stages}.'
        # len(enc_dilations)必須跟num_stages相同
        assert len(enc_dilations) == num_stages, \
            'The length of enc_dilations should be equal to num_stages, '\
            f'while the enc_dilations is {enc_dilations}, the length of '\
            f'enc_dilations is {len(enc_dilations)}, and the num_stages is '\
            f'{num_stages}.'
        # len(dec_dilations)會比num_stages少一層
        assert len(dec_dilations) == (num_stages-1), \
            'The length of dec_dilations should be equal to (num_stages-1), '\
            f'while the dec_dilations is {dec_dilations}, the length of '\
            f'dec_dilations is {len(dec_dilations)}, and the num_stages is '\
            f'{num_stages}.'

        # 將一系列變數進行保存
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.base_channels = base_channels

        # encoder以及decoder都是nn.ModuleList格式，目前都是空
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # 遍歷層結構數，在這裡會同時構造encoder以及decoder
        for i in range(num_stages):
            # enc_conv_block = 暫時存放encoder層結構的地方
            enc_conv_block = []
            if i != 0:
                # 第一層不會進來，因為decoder會比encoder少一層
                # 如過需要進行下採樣但是這層的步距又是1的話就會需要透過MaxPool進行下採樣
                if strides[i] == 1 and downsamples[i - 1]:
                    # 這裡的MaxPool的kernel_size為2，進行2倍下採樣
                    enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                # 當本層有經過下採樣，那們在decoder部分就要進行上採樣，透過upsample進行標記
                upsample = (strides[i] != 1 or downsamples[i - 1])
                # 添加decoder層結構，傳入一些超參數
                # 透過UpConvBlock就會構建出完整的上採樣流程
                self.decoder.append(
                    UpConvBlock(
                        conv_block=BasicConvBlock,
                        in_channels=base_channels * 2**i,
                        skip_channels=base_channels * 2**(i - 1),
                        out_channels=base_channels * 2**(i - 1),
                        num_convs=dec_num_convs[i - 1],
                        stride=1,
                        dilation=dec_dilations[i - 1],
                        with_cp=with_cp,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        upsample_cfg=upsample_cfg if upsample else None,
                        dcn=None,
                        plugins=None))

            # 構建encoder層
            enc_conv_block.append(
                # 這裡就是調用BasicConvBlock構建卷積加上標準化加上激活函數
                BasicConvBlock(
                    in_channels=in_channels,
                    out_channels=base_channels * 2**i,
                    num_convs=enc_num_convs[i],
                    stride=strides[i],
                    dilation=enc_dilations[i],
                    with_cp=with_cp,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    dcn=None,
                    plugins=None))
            # 將enc_conv_block內容先放到nn.Sequential當中再放到encoder當中
            self.encoder.append((nn.Sequential(*enc_conv_block)))
            # 更新in_channels深度
            in_channels = base_channels * 2**i

    def forward(self, x):
        # 已看過，UNet的向前傳播
        # x = shape [batch_size=4, channel=3, height=128, width=128]

        # 檢查一下高寬數值在下採樣時不會遇到無法整除的情況
        self._check_input_divisible(x)
        # 保存encoder的輸出
        enc_outs = []
        # 遍歷整個encoder層結構，encoder是由nn.ModuleList組成
        for enc in self.encoder:
            # x = 通過卷積加上標準化加上激活函數後的tensor
            x = enc(x)
            enc_outs.append(x)
        # 通過所有encoder層後enc_outs就會保存每層的特徵圖，enc_outs長度就會是encoder的層數
        # dec_outs = 存放decoder的輸出特徵層，第一個先把encoder最後一層的輸出放進去
        dec_outs = [x]
        # 遍歷decoder層結構，這裡因為在構建decoder時是倒順序的所以這裡要返過來，這裡的i會從大到小
        for i in reversed(range(len(self.decoder))):
            # 將encoder的輸出與上一層decoder的輸出傳入進行卷積操作
            x = self.decoder[i](enc_outs[i], x)
            # 添加到decoder輸出當中
            dec_outs.append(x)

        # 最後將decoder層的特徵圖全部返回
        return dec_outs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(UNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        # 已看過，主要功能是要檢查是否合法
        # x shape = [batch_size=4, channel=3, height=128, width=128]
        # 取出高寬
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        # 遍歷所有層數
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                # 如果有經過步距為2或是有明確說要進行下採樣，就會在whole_downsample_rate上面乘以2
                whole_downsample_rate *= 2
        # 如果原始高寬無法被下採樣倍率整除就會報錯
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}, when num_stages is '\
            f'{self.num_stages}, strides is {self.strides}, and downsamples '\
            f'is {self.downsamples}.'
