# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger
from ..common import get_module_device


@MODULES.register_module()
class DCGANGenerator(nn.Module):
    """Generator for DCGAN.

    Implementation Details for DCGAN architecture:

    #. Adopt transposed convolution in the generator;
    #. Use batchnorm in the generator except for the final output layer;
    #. Use ReLU in the generator in addition to the final output layer.

    More details can be found in the original paper:
    Unsupervised Representation Learning with Deep Convolutional
    Generative Adversarial Networks
    http://arxiv.org/abs/1511.06434

    Args:
        output_scale (int | tuple[int]): Output scale for the generated
            image. If only an integer is provided, the output image will
            be a square shape. The tuple of two integers will set the
            height and width for the output image, respectively.
        out_channels (int, optional): The channel number of the output feature.
            Default to 3.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this number.
            Default to 1024.
        input_scale (int | tuple[int], optional): Input scale for the
            generated image. If only an integer is provided, the input feature
            ahead of the convolutional generator will be a square shape. The
            tuple of two integers will set the height and width for the input
            convolutional feature, respectively. Defaults to 4.
        noise_size (int, optional): Size of the input noise
            vector. Defaults to 100.
        default_norm_cfg (dict, optional): Norm config for all of layers
            except for the final output layer. Defaults to ``dict(type='BN')``.
        default_act_cfg (dict, optional): Activation config for all of layers
            except for the final output layer. Defaults to
            ``dict(type='ReLU')``.
        out_act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to ``dict(type='Tanh')``.
        pretrained (str, optional): Path for the pretrained model. Default to
            ``None``.
    """

    def __init__(self,
                 output_scale,
                 out_channels=3,
                 base_channels=1024,
                 input_scale=4,
                 noise_size=100,
                 default_norm_cfg=dict(type='BN'),
                 default_act_cfg=dict(type='ReLU'),
                 out_act_cfg=dict(type='Tanh'),
                 pretrained=None):
        """ 對DCGAN進行初始化
        Args:
            output_scale: 輸出圖像的大小，如果只有給一個數字就會是正方形圖像
            out_channels: 輸出的channel深度
            base_channels: 對於生成器的基底channel深度
            input_scale: 輸入到生成器當中的圖像大小，如果只有給一個數字就會是正方形圖像
            noise_size: 輸入的向量維度，最一開始傳入的會是[batch_size, channel=noise_size, height=1, width=1]的資料
            default_norm_cfg: 表準化層相關設定
            default_act_cfg: 激活函數相關設定
            out_act_cfg: 最後一層輸出的激活函數設定
            pretrained: 預訓練權重資料
        """
        # 繼承自nn.Module，對繼承對象進行初始化
        super().__init__()
        # 保存傳入參數
        self.output_scale = output_scale
        self.base_channels = base_channels
        self.input_scale = input_scale
        self.noise_size = noise_size

        # the number of times for upsampling
        # 計算總共需要上採樣多少次，這裡每次上採樣只會放大2倍
        self.num_upsamples = int(np.log2(output_scale // input_scale))

        # output 4x4 feature map
        # 透過轉置卷積將一開始數入的資料從[B, channel, 1, 1]變成[B, channel, 4, 4]
        self.noise2feat = ConvModule(
            # 給定噪聲的向量維度
            noise_size,
            # 輸出的channel深度
            base_channels,
            # 使用卷積核大小為4x4
            kernel_size=4,
            stride=1,
            padding=0,
            # 這裡使用的是轉置卷積
            conv_cfg=dict(type='ConvTranspose2d'),
            norm_cfg=default_norm_cfg,
            act_cfg=default_act_cfg)

        # build up upsampling backbone (excluding the output layer)
        # 一系列上採樣轉置卷積層實例化對象存放地方
        upsampling = []
        # 將當前channel深度設定成base_channels
        curr_channel = base_channels
        # 開始遍歷總共需要構建的層結構數量
        for _ in range(self.num_upsamples - 1):
            upsampling.append(
                # 持續使用轉置卷積模塊，將圖像進行2倍放大
                ConvModule(
                    curr_channel,
                    # 這裡每進行一次上採樣就會將channel深度減半
                    curr_channel // 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    conv_cfg=dict(type='ConvTranspose2d'),
                    norm_cfg=default_norm_cfg,
                    act_cfg=default_act_cfg))

            # 更新當前channel深度
            curr_channel //= 2

        # 最後使用Sequential將多層結構進行包裝
        self.upsampling = nn.Sequential(*upsampling)

        # output layer
        # 構建最後輸出層結構
        self.output_layer = ConvModule(
            curr_channel,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            conv_cfg=dict(type='ConvTranspose2d'),
            # 這裡不會添加標準化層
            norm_cfg=None,
            # 使用的是特別的激活函數
            act_cfg=out_act_cfg)

        # 進行權重初始化
        self.init_weights(pretrained=pretrained)

    def forward(self, noise, num_batches=0, return_noise=False):
        """Forward function.

        Args:
            noise (torch.Tensor | callable | None): You can directly give a
                batch of noise through a ``torch.Tensor`` or offer a callable
                function to sample a batch of noise data. Otherwise, the
                ``None`` indicates to use the default noise sampler.
            num_batches (int, optional): The number of batch size.
                Defaults to 0.
            return_noise (bool, optional): If True, ``noise_batch`` will be
                returned in a dict with ``fake_img``. Defaults to False.

        Returns:
            torch.Tensor | dict: If not ``return_noise``, only the output image
                will be returned. Otherwise, a dict contains ``fake_img`` and
                ``noise_batch`` will be returned.
        """
        """ DCGAN的生成器的forward函數
        Args:
            noise: 最初始的噪聲，這裡如果沒有傳入就會是None
            num_batches: 要生成的圖像數量
            return_noise: 是否需要將noise進行返回，如果有需要的話最終回傳的會是dict格式
        """
        # receive noise and conduct sanity check.
        if isinstance(noise, torch.Tensor):
            # 如果傳入的noise是tensor就會到這裡
            # 檢查傳入的noise的channel與初始化設定的是否相同
            assert noise.shape[1] == self.noise_size
            if noise.ndim == 2:
                # 如果noise是兩個維度的資料會到這裡，在後面加上高寬維度
                noise_batch = noise[:, :, None, None]
            elif noise.ndim == 4:
                # 如果noise是四個維度就會到這裡，直接進行使用就可以
                noise_batch = noise
            else:
                # 其他的shape就會直接報錯
                raise ValueError('The noise should be in shape of (n, c) or '
                                 f'(n, c, 1, 1), but got {noise.shape}')
        # receive a noise generator and sample noise.
        elif callable(noise):
            # 如果傳入的noise是可以進行呼叫的就會到這裡，也就是傳入的可能是函數
            noise_generator = noise
            assert num_batches > 0
            # 透過可呼叫對象進行創建noise資訊，這裡傳入的是最後noise的shape
            noise_batch = noise_generator((num_batches, self.noise_size, 1, 1))
        # otherwise, we will adopt default noise sampler.
        else:
            # 其他就會到這裡，包括noise是None的情況
            assert num_batches > 0
            # 直接用亂數的方式生成，tensor shape [num_batches, noise_channel, height=1, width=1]
            noise_batch = torch.randn((num_batches, self.noise_size, 1, 1))

        # dirty code for putting data on the right device
        # 將獲取的tensor轉到設備上，可以提升運行速度
        noise_batch = noise_batch.to(get_module_device(self))

        # 通過第一次轉置卷積將noise轉成圖像
        x = self.noise2feat(noise_batch)
        # 進行上採樣操作
        x = self.upsampling(x)
        # 最後通過一層轉置卷積
        x = self.output_layer(x)

        if return_noise:
            # 如果有需要將原始noise進行回傳就會到這裡
            return dict(fake_img=x, noise_batch=noise_batch)

        # 如果不需要回傳noise就直接回傳x，tensor shape [batch_size, channel, height, width]
        return x

    def init_weights(self, pretrained=None):
        """Init weights for models.

        We just use the initialization method proposed in the original paper.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    normal_init(m, 0, 0.02)
                elif isinstance(m, _BatchNorm):
                    nn.init.normal_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0)
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')


@MODULES.register_module()
class DCGANDiscriminator(nn.Module):
    """Discriminator for DCGAN.

    Implementation Details for DCGAN architecture:

    #. Adopt convolution in the discriminator;
    #. Use batchnorm in the discriminator except for the input and final \
       output layer;
    #. Use LeakyReLU in the discriminator in addition to the output layer.

    Args:
        input_scale (int): The scale of the input image.
        output_scale (int): The final scale of the convolutional feature.
        out_channels (int): The channel number of the final output layer.
        in_channels (int, optional): The channel number of the input image.
            Defaults to 3.
        base_channels (int, optional): The basic channel number of the
            generator. The other layers contains channels based on this number.
            Defaults to 128.
        default_norm_cfg (dict, optional): Norm config for all of layers
            except for the final output layer. Defaults to ``dict(type='BN')``.
        default_act_cfg (dict, optional): Activation config for all of layers
            except for the final output layer. Defaults to
            ``dict(type='ReLU')``.
        out_act_cfg (dict, optional): Activation config for the final output
            layer. Defaults to ``dict(type='Tanh')``.
        pretrained (str, optional): Path for the pretrained model. Default to
            ``None``.
    """

    def __init__(self,
                 input_scale,
                 output_scale,
                 out_channels,
                 in_channels=3,
                 base_channels=128,
                 default_norm_cfg=dict(type='BN'),
                 default_act_cfg=dict(type='LeakyReLU'),
                 out_act_cfg=None,
                 pretrained=None):
        """ DCGAN的鑑別器初始化部分
        Args:
            input_scale: 輸入圖像大小，如果只有一個數字表示輸入的是正方形圖像
            output_scale: 輸入到最後一個卷積層前的圖像大小
            out_channels: 輸出的channel深度
            in_channels: 輸入的channel深度
            base_channels: 基底的channel深度
            default_norm_cfg: 標準化層設定
            default_act_cfg: 激活函數層設定
            out_act_cfg: 最後一層的激活函數設定
            pretrained: 預訓練權重設定
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        # 將傳入資料進行保存
        self.input_scale = input_scale
        self.output_scale = output_scale
        self.out_channels = out_channels
        self.base_channels = base_channels

        # the number of times for downsampling
        # 計算需要通過多少下採樣層可以到達指定高寬，這裡每次會進行2倍下採樣
        self.num_downsamples = int(np.log2(input_scale // output_scale))

        # build up downsampling backbone (excluding the output layer)
        # 下採樣層結構保存list
        downsamples = []
        # 遍歷需要下採樣層數
        for i in range(self.num_downsamples):
            # remove norm for the first conv
            # 第一層卷積會將標準化層去除
            norm_cfg_ = None if i == 0 else default_norm_cfg
            # 獲取當前channel深度
            in_ch = in_channels if i == 0 else base_channels * 2**(i - 1)

            # 透過步距為2的卷積進行下採樣
            downsamples.append(
                ConvModule(
                    in_ch,
                    # 這裡每層會將channel放大2倍
                    base_channels * 2**i,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=norm_cfg_,
                    act_cfg=default_act_cfg))
            # 更新當前channel深度
            curr_channels = base_channels * 2**i

        # 將多層下採樣層用Sequential進行包裝
        self.downsamples = nn.Sequential(*downsamples)

        # define output layer
        # 最後再通過一層卷積層，將圖像變成1x1大小，channel調整成指定的深度
        self.output_layer = ConvModule(
            curr_channels,
            out_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=None,
            # 這裡使用的激活函數是特別的
            act_cfg=out_act_cfg)

        # 進行權重初始化設定
        self.init_weights(pretrained=pretrained)

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Fake or real image tensor.

        Returns:
            torch.Tensor: Prediction for the reality of the input image.
        """
        # DCGAN的鑑別器forward函數
        # x = tensor shape [batch_size, channel, height, width]

        # 獲取傳入的batch_size
        n = x.shape[0]
        # 進行一系列下採樣
        x = self.downsamples(x)
        # 這裡的num_classes只有分成真偽，所以channel=1
        # 最後通過一層下採樣，x shape [batch_size, channel=num_classes, 1, 1]
        x = self.output_layer(x)

        # reshape to a flatten feature
        # 進行通道調整，x shape [batch_size, channel=num_classes]
        return x.view(n, -1)

    def init_weights(self, pretrained=None):
        """Init weights for models.

        We just use the initialization method proposed in the original paper.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    normal_init(m, 0, 0.02)
                elif isinstance(m, _BatchNorm):
                    nn.init.normal_(m.weight.data)
                    nn.init.constant_(m.bias.data, 0)
        else:
            raise TypeError('pretrained must be a str or None but'
                            f' got {type(pretrained)} instead.')
