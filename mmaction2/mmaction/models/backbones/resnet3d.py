# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, NonLocal3d, build_activation_layer,
                      constant_init, kaiming_init)
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
from torch.nn.modules.utils import _ntuple, _triple

from ...utils import get_root_logger
from ..builder import BACKBONES

try:
    from mmdet.models import BACKBONES as MMDET_BACKBONES
    from mmdet.models.builder import SHARED_HEADS as MMDET_SHARED_HEADS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


class BasicBlock3d(nn.Module):
    """BasicBlock 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        inflate (bool): Whether to inflate kernel. Default: True.
        non_local (bool): Determine whether to apply non-local module in this
            block. Default: False.
        non_local_cfg (dict): Config for non-local module. Default: ``dict()``.
        conv_cfg (dict): Config dict for convolution layer.
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type``,
            Default: ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 inflate=True,
                 non_local=False,
                 non_local_cfg=dict(),
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 **kwargs):
        super().__init__()
        assert style in ['pytorch', 'caffe']
        # make sure that only ``inflate_style`` is passed into kwargs
        assert set(kwargs).issubset(['inflate_style'])

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        self.conv1_stride_s = spatial_stride
        self.conv2_stride_s = 1
        self.conv1_stride_t = temporal_stride
        self.conv2_stride_t = 1

        if self.inflate:
            conv1_kernel_size = (3, 3, 3)
            conv1_padding = (1, dilation, dilation)
            conv2_kernel_size = (3, 3, 3)
            conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 3, 3)
            conv1_padding = (0, dilation, dilation)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            padding=conv1_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes * self.expansion,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s),
            padding=conv2_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

        if self.non_local:
            self.non_local_block = NonLocal3d(self.conv2.norm.num_features,
                                              **self.non_local_cfg)

    def forward(self, x):
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        if self.non_local:
            out = self.non_local_block(out)

        return out


class Bottleneck3d(nn.Module):
    """Bottleneck 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        inflate (bool): Whether to inflate kernel. Default: True.
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        non_local (bool): Determine whether to apply non-local module in this
            block. Default: False.
        non_local_cfg (dict): Config for non-local module. Default: ``dict()``.
        conv_cfg (dict): Config dict for convolution layer.
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type``,
            Default: ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 inflate=True,
                 inflate_style='3x1x1',
                 non_local=False,
                 non_local_cfg=dict(),
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        """ 已看過，ResNet當中的Bottleneck3D的初始化函數
        Args:
            inflate: 輸入的channel深度
            planes: 當前block的基礎channel深度
            spatial_stride: 空間部分的步距
            temporal_stride: 時間部分的步距
            dilation: 膨脹係數
            downsample: 下採樣的層結構，如果殘差結構不需要通過downsample就會是None
            style: 模型風格
            inflate: 是否需要膨脹到3D
            inflate_style: 膨脹的風格
            non_local: 是否需要用non-local模型
            non_local_cfg: non-local模型的設定資料
            conv_cfg: 卷積配置資訊
            act_cfg: 激活函數配置資訊
            with_cp: 是否使用checkpoint
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        # 傳入的style需要是pytorch或是caffe的風格
        assert style in ['pytorch', 'caffe']
        # 傳入的膨脹風格需要是3x1x1或是3x3x3的
        assert inflate_style in ['3x1x1', '3x3x3']

        # 保存傳入的參數
        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        if self.style == 'pytorch':
            # 如果底層的風格是用pytorch就會到這裡
            # 將conv1的時間與空間的步距都設定成1
            # 將conv2的時間與空間的步距根據傳入的資料進行設定
            self.conv1_stride_s = 1
            self.conv2_stride_s = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            # 如果底層的風格是用caffe就會到這裡
            # 將conv1的時間與空間步距都根據傳入的資料進行設定
            # 將conv2的時間與空間步距都設定成1
            self.conv1_stride_s = spatial_stride
            self.conv2_stride_s = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1

        if self.inflate:
            # 如果有需要進行膨脹就會到這裡
            if inflate_style == '3x1x1':
                # 如果傳入的膨脹風格是3x1x1就會到這裡
                # 將conv1卷積核設定成(3, 1, 1)，通過padding後經過conv1後shape不會產生變化
                conv1_kernel_size = (3, 1, 1)
                # 將conv1的padding設定成(1, 0, 0)
                conv1_padding = (1, 0, 0)
                # 將conv2的卷積核設定成(1, 3, 3)，通過padding後經過conv2後shape不會產生變化
                conv2_kernel_size = (1, 3, 3)
                # 設定conv2的padding
                conv2_padding = (0, dilation, dilation)
            else:
                # 如果傳入的膨脹風格是3x3x3就會到這裡
                # 主要是在第二個卷積將卷積核設定成(3, 3, 3)
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            # 如果沒有設定膨脹就會到這裡，這裡在時間維度上面都不會進行融合
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        # 構建卷積以及標準化以及激活函數多層結構
        self.conv1 = ConvModule(
            inplanes,
            planes,
            # 使用的卷積核就是上面設定好的
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s,
                    self.conv1_stride_s),
            # padding大小就會是上面設定好的
            padding=conv1_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # 構建中間層卷積
        self.conv2 = ConvModule(
            planes,
            planes,
            # 使用的卷積核就是上面設定好的
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s,
                    self.conv2_stride_s),
            # padding大小就會是上面設定好的
            padding=conv2_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # 最後通過第三個卷積將channel進行擴維
        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            # 卷積核就設定成1x1x1
            1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            # No activation in the third ConvModule for bottleneck
            # 這裡將激活函數取消
            act_cfg=None)

        # 保存downsample的層結構
        self.downsample = downsample
        # 構建激活函數，將主幹與殘差邊相加後會需要經過激活函數
        self.relu = build_activation_layer(self.act_cfg)

        if self.non_local:
            # 如果有需要使用到non_local就會實例化non_local模型
            # non_local主要是可以提供類似注意力的行為，可以獲取到非局部的特徵
            self.non_local_block = NonLocal3d(self.conv3.norm.num_features,
                                              **self.non_local_cfg)

    def forward(self, x):
        """Defines the computation performed at every call."""
        # 已看過，ResNet3D的Bottleneck的forward部分
        # x = 輸入的影像特徵資料，tensor shape [batch_size * num_clip, channel, clip_len, height, width]

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            # 已看過，向前傳遞

            # 保留傳入的x作為之後殘差結構的值
            identity = x

            # 通過第一層卷積以及標準化以及激活層，如果是pytorch格式就會是進行時空的融合
            out = self.conv1(x)
            # 通過第二層卷積以及標準化以及激活層，如果是pytorch格式就會是進行空間的融合
            out = self.conv2(out)
            # 通過第三層卷積以及標準化，不管任何格式就是進行channel深度調整
            out = self.conv3(out)

            if self.downsample is not None:
                # 如果主幹提取時有改變特徵圖大小或是深度有改變，殘差邊上的資料也會需要調整
                identity = self.downsample(x)

            # 進行特徵融合
            out = out + identity
            # 回傳結果
            return out

        if self.with_cp and x.requires_grad:
            # 如果有啟用checkpoint且x需要進行反向傳遞就會到這裡
            out = cp.checkpoint(_inner_forward, x)
        else:
            # 其他就會到這裡
            out = _inner_forward(x)
        # 將經過多層結構的結果放入到激活層當中
        out = self.relu(out)

        if self.non_local:
            # 如果有啟用non_local模塊就會到這裡進行正向傳遞
            out = self.non_local_block(out)

        # 將提取特徵後的特徵圖進行回傳
        return out


@BACKBONES.register_module()
class ResNet3d(nn.Module):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str | None): Name of pretrained model.
        stage_blocks (tuple | None): Set number of stages for each res layer.
            Default: None.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Default: True.
        in_channels (int): Channel num of input features. Default: 3.
        base_channels (int): Channel num of stem output features. Default: 64.
        out_indices (Sequence[int]): Indices of output feature. Default: (3, ).
        num_stages (int): Resnet stages. Default: 4.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
            Default: ``(1, 2, 2, 2)``.
        temporal_strides (Sequence[int]):
            Temporal strides of residual blocks of each stage.
            Default: ``(1, 1, 1, 1)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Default: ``(3, 7, 7)``.
        conv1_stride_s (int): Spatial stride of the first conv layer.
            Default: 2.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Default: 1.
        pool1_stride_s (int): Spatial stride of the first pooling layer.
            Default: 2.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Default: 1.
        with_pool2 (bool): Whether to use pool2. Default: True.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default: -1.
        inflate (Sequence[int]): Inflate Dims of each block.
            Default: (1, 1, 1, 1).
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        conv_cfg (dict): Config for conv layers. required keys are ``type``
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type`` and
            ``requires_grad``.
            Default: ``dict(type='BN3d', requires_grad=True)``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU', inplace=True)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        non_local (Sequence[int]): Determine whether to apply non-local module
            in the corresponding block of each stages. Default: (0, 0, 0, 0).
        non_local_cfg (dict): Config for non-local module. Default: ``dict()``.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
        kwargs (dict, optional): Key arguments for "make_res_layer".
    """

    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained,
                 stage_blocks=None,
                 pretrained2d=True,
                 in_channels=3,
                 num_stages=4,
                 base_channels=64,
                 out_indices=(3, ),
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 conv1_kernel=(3, 7, 7),
                 conv1_stride_s=2,
                 conv1_stride_t=1,
                 pool1_stride_s=2,
                 pool1_stride_t=1,
                 with_pool1=True,
                 with_pool2=True,
                 style='pytorch',
                 frozen_stages=-1,
                 inflate=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_eval=False,
                 with_cp=False,
                 non_local=(0, 0, 0, 0),
                 non_local_cfg=dict(),
                 zero_init_residual=True,
                 **kwargs):
        """ 已看過，ResNet3D模型的初始化函數
        Args:
            depth: 指定ResNet的深度，通常會是使用50的深度
            pretrained: 預訓練權重設定
            stage_blocks: 對於resnet當中每層需要堆疊的數量
            pretrained2d: 是否需要載入2D的預訓練權重
            in_channels: 輸入的channel深度
            num_stages: resnet當中的主要stage數量
            base_channels: 基礎的channel深度，每到下一個stage就會是(2 ^ stage) * base_channels的channel深度
            out_indices: 哪幾個stages的結果需要進行輸出
            spatial_strides: 每個stage在空間維度上面的步距，也就是原先2D上面的步距，對於圖像的高寬步距
            temporal_strides: 每個stage在時間維度上面的步距(這個是在3D卷積當中新出來的)
            dilations: 膨脹係數，通常在resnet當中不會用到膨脹卷積
            conv1_kernel: 第一層卷積層的卷積核設定
            conv1_stride_s: 第一層卷積層在空間維度上的步距
            conv1_stride_t: 第一層卷積層在時間維度上的步距
            pool1_stride_s: 第一層池化層在空間維度上的步距
            pool1_stride_t: 第一層池化層在時間維度上的步距
            with_pool1: 是否使用第一層池化層
            with_pool2: 是否使用第二層池化層
            style: 模型架構風格，這裡常用pytorch
            frozen_stages: 哪部分的層結構需要進行凍結，如果設定為-1就表示都不進行凍結
            inflate: 每個stage的膨脹深度(這是在action中才有的)
            inflate_style: 決定在每個stage的conv1與conv2的kernel_size與padding_stride
            conv_cfg: 卷積的配置，這裡默認會使用Conv3D
            norm_cfg: 標準化的配置，這裡默認會使用BN3D
            act_cfg: 激活函數的配置，這裡默認使用ReLU
            norm_eval: 在訓練過程是否凍結標準化層的均值以及標準差
            with_cp: 是否使用checkpoint，如果啟用可以減少memory的使用，但是會使模型訓練速度變慢
            non_local: 是否使用non-local模型
            non_local_cfg: non-local模型的設定參數
            zero_init_residual: 使用0進行初始化殘差邊，這裡是訓練resnet的小技巧
            kwargs: 其他參數，通常為空
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        if depth not in self.arch_settings:
            # 如果指定的depth不在合法的深度就會直接報錯，這裡支援的深度有[18, 34, 50, 101, 152]
            raise KeyError(f'invalid depth {depth} for resnet')
        # 保存傳入的參數
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        # 這裡的num_stages範圍會在[1, 4]之間
        assert 1 <= num_stages <= 4
        self.stage_blocks = stage_blocks
        self.out_indices = out_indices
        # 指定輸出的層數不會超過stage的數量，最多就是將每個stage的輸出進行輸出
        assert max(out_indices) < num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        # 這裡檢查一下(空間上的步距, 時間上的步距, 膨脹係數)的長度需要一樣，相同index的資訊會湊成一個stage的配置
        assert len(spatial_strides) == len(temporal_strides) == len(
            dilations) == num_stages
        if self.stage_blocks is not None:
            # 如果有指定stage_blocks的話長度就會需要與num_stages相同，這樣每個index表示該stage當中的block需要堆疊多少次
            assert len(self.stage_blocks) == num_stages

        # 保存傳入參數
        self.conv1_kernel = conv1_kernel
        self.conv1_stride_s = conv1_stride_s
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_s = pool1_stride_s
        self.pool1_stride_t = pool1_stride_t
        self.with_pool1 = with_pool1
        self.with_pool2 = with_pool2
        self.style = style
        self.frozen_stages = frozen_stages
        # 每個stage的膨脹設定，這裡使用的_ntuple是torch官方的函數
        # stage_inflations = tuple(tuple)，第一個tuple長度會是num_stage，第二個tuple會是每個block的inflation參數
        # 以resnet50為例就會是 = tuple(tuple(1, 1, 1), tuple(1, 0, 1, 0), tuple(1, 0, 1, 0, 1, 0), tuple(0, 1, 0))
        self.stage_inflations = _ntuple(num_stages)(inflate)
        # non_local_stages相關設定
        self.non_local_stages = _ntuple(num_stages)(non_local)
        self.inflate_style = inflate_style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        # 根據指定的depth，獲取相配對的block以及每個stage當中要堆疊多少個block
        self.block, stage_blocks = self.arch_settings[depth]

        if self.stage_blocks is None:
            # 如果原先沒有指定的stage_blocks就會到這裡
            # 提取剛剛獲得的stage_blocks前num_stages的堆疊數量
            self.stage_blocks = stage_blocks[:num_stages]

        # 將inplanes設定成base_channels，因為在第一個stage輸入的channel會已經是base_channels
        self.inplanes = self.base_channels

        self.non_local_cfg = non_local_cfg

        # 構建stage之前的卷積層
        self._make_stem_layer()

        # 剩下的resnet的多層stage層模塊
        self.res_layers = []
        # 遍歷每層stage模塊進行實例化
        for i, num_blocks in enumerate(self.stage_blocks):
            # 獲取該層stage在空間上的步距
            spatial_stride = spatial_strides[i]
            # 獲取該層stage在時間上的步距
            temporal_stride = temporal_strides[i]
            # 獲取該層stage的膨脹係數
            dilation = dilations[i]
            # 該層基底的channel深度
            planes = self.base_channels * 2**i
            # 透過make_res_layer構建stage實例化對象
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                non_local=self.non_local_stages[i],
                non_local_cfg=self.non_local_cfg,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                with_cp=with_cp,
                **kwargs)
            # 輸出的channel深度，透過輸入的深度乘上中間擴大的channel深度
            self.inplanes = planes * self.block.expansion
            # 給層結構一個名稱
            layer_name = f'layer{i + 1}'
            # 透過add_module將名稱與實例化對象進行配對方到模型當中
            self.add_module(layer_name, res_layer)
            # 保存層結構名稱，使用時就可以透過呼叫名字使用
            self.res_layers.append(layer_name)

        # 最終輸出的channel深度
        self.feat_dim = self.block.expansion * self.base_channels * 2**(
            len(self.stage_blocks) - 1)

    @staticmethod
    def make_res_layer(block,
                       inplanes,
                       planes,
                       blocks,
                       spatial_stride=1,
                       temporal_stride=1,
                       dilation=1,
                       style='pytorch',
                       inflate=1,
                       inflate_style='3x1x1',
                       non_local=0,
                       non_local_cfg=dict(),
                       norm_cfg=None,
                       act_cfg=None,
                       conv_cfg=None,
                       with_cp=False,
                       **kwargs):
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature
                in each block.
            planes (int): Number of channels for the output feature
                in each block.
            blocks (int): Number of residual blocks.
            spatial_stride (int | Sequence[int]): Spatial strides in
                residual and conv layers. Default: 1.
            temporal_stride (int | Sequence[int]): Temporal strides in
                residual and conv layers. Default: 1.
            dilation (int): Spacing between kernel elements. Default: 1.
            style (str): ``pytorch`` or ``caffe``. If set to ``pytorch``,
                the stride-two layer is the 3x3 conv layer, otherwise
                the stride-two layer is the first 1x1 conv layer.
                Default: ``pytorch``.
            inflate (int | Sequence[int]): Determine whether to inflate
                for each block. Default: 1.
            inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines
                the kernel sizes and padding strides for conv1 and conv2
                in each block. Default: '3x1x1'.
            non_local (int | Sequence[int]): Determine whether to apply
                non-local module in the corresponding block of each stages.
                Default: 0.
            non_local_cfg (dict): Config for non-local module.
                Default: ``dict()``.
            conv_cfg (dict | None): Config for norm layers. Default: None.
            norm_cfg (dict | None): Config for norm layers. Default: None.
            act_cfg (dict | None): Config for activate layers. Default: None.
            with_cp (bool | None): Use checkpoint or not. Using checkpoint
                will save some memory while slowing down the training speed.
                Default: False.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        # 已看過，構建resnet當中的stage模塊
        # block = stage當中的基礎層結構，一個stage當中需要堆疊多層block模塊
        # inplanes = 輸入的channel深度
        # planes = 該層stage的基底channel深度
        # blocks = 當中的block需要堆疊多少次
        # spatial_stride = 空間上的步距
        # temporal_stride = 時間上的步距
        # dilation = 膨脹係數
        # style = 底層使用的架構
        # inflate = 是否需要對該block進行膨脹到3D卷積
        # inflate_style = 膨脹的型態
        # non_local = 是否需要在指定的block上使用non-local的層結構
        # non_local_cfg = non-local模型的設定config資料
        # norm_cfg = 標準化層的設定
        # act_cfg = 激活函數的設定
        # conv_cfg = 卷積層的設定
        # with_cp = 是否有啟用checkpoint
        # kwargs = 其他參數

        # 如果inflate是int格式就會拓展成tuple型態，且tuple長度會與堆疊的block數量相同，數值都會是一樣
        # 如果傳入時不是int就不會有任合變動
        inflate = inflate if not isinstance(inflate, int) else (inflate, ) * blocks
        # 這裡會做出與inflate一樣的動作
        non_local = non_local if not isinstance(non_local, int) else (non_local, ) * blocks
        # 檢查inflate與non_local的長度需要一致
        assert len(inflate) == blocks and len(non_local) == blocks
        # 先將下採樣的部分設定成None
        downsample = None
        if spatial_stride != 1 or inplanes != planes * block.expansion:
            # 當通過第一個block會使得channel變化或是高寬產生變化，就會需要downsample讓殘差邊的特徵圖進行downsample使其可以相加
            # 構建downsample的卷積以及標準化以及激活函數層結構
            downsample = ConvModule(
                # 輸入的channel深度
                inplanes,
                # 輸出的channel深度
                planes * block.expansion,
                # 透過1*1*1的卷積核
                kernel_size=1,
                # 這裡的步距就會與主幹上的步距相同
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)

        # stage當中的block結構存放位置
        layers = []
        layers.append(
            # 先構建第一個block實例對象
            block(
                inplanes,
                planes,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                downsample=downsample,
                style=style,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp,
                **kwargs))
        # 更新輸入的channel深度
        inplanes = planes * block.expansion
        # 構建剩下的block層結構的實例對象
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    spatial_stride=1,
                    temporal_stride=1,
                    dilation=dilation,
                    style=style,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                    **kwargs))

        # 最後統一放到nn.Sequential後回傳
        return nn.Sequential(*layers)

    @staticmethod
    def _inflate_conv_params(conv3d, state_dict_2d, module_name_2d,
                             inflated_param_names):
        """Inflate a conv module from 2d to 3d.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        # 已看過，將2D的卷積預訓練權重轉到3D上
        # conv3d = 3D卷積的實例化對象
        # state_dict_2d = 預訓練的權重值，這裡會是個OrderedDict格式
        # module_name_2d = 在2D模型當中的層結構名稱
        # inflated_param_names = 那些參數有被進行膨脹

        # 獲取weight_2d_name = 原始2d層結構名稱再加上.weight
        weight_2d_name = module_name_2d + '.weight'

        # 從預訓練權重字典當中找到我們指定的層結構權重值
        conv2d_weight = state_dict_2d[weight_2d_name]
        # 獲取時空維度的深度
        kernel_t = conv3d.weight.data.shape[2]

        # 獲取新的權重資料，會在2D權重上的第二個維度擴維，深度會是時空維度的深度，並且會將權重值變成原先的kernel_t分之一
        # 也就是因為增加了時空的維度，所以需要將原先的權重進行平均化，經過計算可以證明出通過這樣的卷積與通過2D卷積後獲得的值會相同
        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(
            conv3d.weight) / kernel_t
        # 將權重值拷貝一份到3D卷積當中
        conv3d.weight.data.copy_(new_weight)
        # 將卷積的名稱保存到inflated_param_names當中
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            # 如果conv3d當中有bias就會到這裡
            bias_2d_name = module_name_2d + '.bias'
            # 偏置部分就直接放到3D卷積上就可以了
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            # 保存下名稱
            inflated_param_names.append(bias_2d_name)

    @staticmethod
    def _inflate_bn_params(bn3d, state_dict_2d, module_name_2d,
                           inflated_param_names):
        """Inflate a norm module from 2d to 3d.

        Args:
            bn3d (nn.Module): The destination bn3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding bn module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        # 已看過，將2D的BN預訓練權重膨脹到3D的BN上
        # bn3d = 3D的BN實例化對象
        # state_dict_2d = 整個預訓練權重資料，這裡會是dict格式
        # module_name_2d = 在2D模型當中層結構名稱，需要這個名稱才可以在state_dict_2d當中找到對應的權重資料
        # inflated_param_names = 有進行膨脹的層結構名稱

        # 遍歷bn3d當中的層結構名稱，一個bn層當中會有兩個可學習參數
        for param_name, param in bn3d.named_parameters():
            # param_name = 該參數的名稱
            # param = 參數本身，會是torch的Parameter型態
            # 獲取在2D模型下的層結構名稱，會是層結構名稱加上權重名稱
            param_2d_name = f'{module_name_2d}.{param_name}'
            # 獲取在權重字典當中對應的權重資料
            param_2d = state_dict_2d[param_2d_name]
            if param.data.shape != param_2d.shape:
                # 如果兩個的shape不同就表示有問題，會跳出警告表示匹配不正確
                warnings.warn(f'The parameter of {module_name_2d} is not'
                              'loaded due to incompatible shapes. ')
                return

            # 如果shape可以匹配上就會直接將值複製上去
            param.data.copy_(param_2d)
            # 並添加名稱到inflated_param_names當中
            inflated_param_names.append(param_2d_name)

        # 這裡會通過named_buffers獲取其他參數，有些buffer參數不會在上面的地方被提取出來，這裡會透過named_buffers被提取出來
        for param_name, param in bn3d.named_buffers():
            param_2d_name = f'{module_name_2d}.{param_name}'
            # some buffers like num_batches_tracked may not exist in old checkpoints
            if param_2d_name in state_dict_2d:
                param_2d = state_dict_2d[param_2d_name]
                param.data.copy_(param_2d)
                inflated_param_names.append(param_2d_name)

    @staticmethod
    def _inflate_weights(self, logger):
        """Inflate the resnet2d parameters to resnet3d.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging information.
        """
        # 已看過，使用該函數將2D的預訓練參數放到3D的模型當中

        # 使用_load_checkpoint將指定的預訓練模型權重載入
        state_dict_r2d = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_r2d:
            # 如果模型權重當中有state_dict就將其提取出來，這個才會是我們需要的權重資料
            state_dict_r2d = state_dict_r2d['state_dict']

        # 保存膨脹後的權重名稱
        inflated_param_names = []
        # 遍歷當前模型的所有層結構的名稱
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                # 如果當前結構是ConvModule類型的就會到這裡
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if 'downsample' in name:
                    # 如果當中有downsample就會到這裡
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    # original_conv_name = 原始層結構名稱 + '.0'
                    original_conv_name = name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    # original_bn_name = 原始層結構名稱 + '.1'
                    original_bn_name = name + '.1'
                else:
                    # 如果不是downsample上的ConvModule就會到這裡
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    # original_conv_name = 原始名稱
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    # original_bn_name = 原始名稱，並且將conv換成bn，原先會是.conv.bn變成.bn
                    original_bn_name = name.replace('conv', 'bn')
                if original_conv_name + '.weight' not in state_dict_r2d:
                    # 如果沒有在預訓練權重檔案找到對應的層結構就會報錯，表示有問題
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_conv_name}')
                else:
                    # 獲取在預訓練資料上面對於該層結構的shape
                    shape_2d = state_dict_r2d[original_conv_name + '.weight'].shape
                    # 獲取在3D模型上該層結構的shape
                    shape_3d = module.conv.weight.data.shape
                    # 這裡在shape_3d的[2]是時空上的維度，所以去除時空上的維度其他地方應該需要與shape_2d相同
                    if shape_2d != shape_3d[:2] + shape_3d[3:]:
                        # 如果去除時空維度還是不同表示shape有問題，這裡會跳出警告
                        logger.warning(f'Weight shape mismatch for '
                                       f': {original_conv_name} : '
                                       f'3d weight shape: {shape_3d}; '
                                       f'2d weight shape: {shape_2d}. ')
                    else:
                        # 如果只有時空上需要對齊就會到這裡，透過_inflate_conv_params將2D的權重膨脹到3D
                        self._inflate_conv_params(module.conv, state_dict_r2d,
                                                  original_conv_name,
                                                  inflated_param_names)

                if original_bn_name + '.weight' not in state_dict_r2d:
                    # 如果該BN沒有對應到的預訓練權重資料就會跳出警告，表示這裡有問題
                    logger.warning(f'Module not exist in the state_dict_r2d'
                                   f': {original_bn_name}')
                else:
                    # 如果有對應的預訓練權重就會透過_inflate_bn_params將預訓練權重調整到可以轉到3D的模型上
                    self._inflate_bn_params(module.bn, state_dict_r2d,
                                            original_bn_name,
                                            inflated_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        # 獲取哪些層結構沒有被從2D模型加載到3D模型上
        remaining_names = set(state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            # 將沒有加載到的層結構打印出來
            logger.info(f'These parameters in the 2d checkpoint are not loaded'
                        f': {remaining_names}')

    def inflate_weights(self, logger):
        # 已看過，將預訓練權重資料擴展成3D可以接受
        self._inflate_weights(self, logger)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        # 已看過，構建resnet當中最一開始的幾個層結構

        # 構建第一個卷積模塊，這裡使用的ConvModule會與一般2D圖像在使用的卷積會有所不同
        # 這裡的模塊依舊會包含[卷積, 標準化層, 激活函數]三個層結構
        self.conv1 = ConvModule(
            # 給定輸入的channel
            self.in_channels,
            # 輸出的channel
            self.base_channels,
            # 卷積核大小
            kernel_size=self.conv1_kernel,
            # 步距，這裡分別給的會是(時間方面的步距, 空間方面的步距, 空間方面的步距)
            stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            # 根據kernel大小決定需要padding的大小，主要是不讓維度有所減小
            padding=tuple([(k - 1) // 2 for k in _triple(self.conv1_kernel)]),
            # 是否啟用bias
            bias=False,
            # 卷積設定
            conv_cfg=self.conv_cfg,
            # 標準化層設定
            norm_cfg=self.norm_cfg,
            # 激活函數設定
            act_cfg=self.act_cfg)

        # 通過一個最大池化下採樣，這裡用的是3D的池化核，這裡在時間維度上面核心大小是1
        # 這裡會對空間上面進行2倍下採樣，時間維度上面不會發生變化
        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            # 步距
            stride=(self.pool1_stride_t, self.pool1_stride_s,
                    self.pool1_stride_s),
            # padding設定
            padding=(0, 1, 1))

        # 這裡還會有先實例化第二個池化層結構，看起來是要將時間維度進行2倍下採樣用的，空間維度上面不會發生變化
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @staticmethod
    def _init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch.

        Args:
            pretrained (str | None): The path of the pretrained weight. Will
                override the original `pretrained` if set. The arg is added to
                be compatible with mmdet. Default: None.
        """
        # 已看過，進行Resnet3D的權重初始化
        if pretrained:
            # 如果有傳入pretrained就會將其放入到self當中，如果self當中已經有指定就會被覆蓋
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            # 如果pretrained是str就會到這裡
            # 構建logger紀錄
            logger = get_root_logger()
            # 記錄下要從哪裡載入預訓練權重
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                # 如果預訓練權重是從2D獲得的就會到這裡
                # Inflate 2D model into 3D model.
                # 透過inflate_weights將權重變成3D模型可以接受的格式
                self.inflate_weights(logger)

            else:
                # Directly load 3D model.
                # 如果預訓練權重是從3D模型來的就可以直接載入就可以
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            # 如果沒有傳入pretrained資料就會到這裡，使用比較好的隨機方式進行初始化
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck3d):
                        constant_init(m.conv3.bn, 0)
                    elif isinstance(m, BasicBlock3d):
                        constant_init(m.conv2.bn, 0)
        else:
            # 其他的pretrained就會到這裡直接報錯
            raise TypeError('pretrained must be a str or None')

    def init_weights(self, pretrained=None):
        # 已看過，Resnet3D的初始化權重部分
        self._init_weights(self, pretrained)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        # 已看過，ResNet3D的forward函數
        # x = 一個batch的影像資料，tensor shape [batch_size * num_clips, channel, clip_len, height, width]

        # 通過第一個卷積
        x = self.conv1(x)
        if self.with_pool1:
            # 如果有實例化)
            x = self.maxpool(x)
        # 構建保存輸出的list
        outs = []
        # 遍歷剩下resnet當中的層結構
        for i, layer_name in enumerate(self.res_layers):
            # 透過當前需要層結構的名稱，透過層結構名稱獲取實例對象
            res_layer = getattr(self, layer_name)
            # 將x放入到層結構中進行向前傳遞
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                # 如果當前是在第0層的stage且有pool2就會進來
                x = self.pool2(x)
            if i in self.out_indices:
                # 如果當前的層數是需要進行輸出的就會到這裡進行保留
                outs.append(x)
        if len(outs) == 1:
            # 如果輸出的層數只有一層就直接提取出來進行輸出
            return outs[0]

        # 如果有多層就用tuple包起來
        return tuple(outs)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module()
class ResNet3dLayer(nn.Module):
    """ResNet 3d Layer.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str | None): Name of pretrained model.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Default: True.
        stage (int): The index of Resnet stage. Default: 3.
        base_channels (int): Channel num of stem output features. Default: 64.
        spatial_stride (int): The 1st res block's spatial stride. Default 2.
        temporal_stride (int): The 1st res block's temporal stride. Default 1.
        dilation (int): The dilation. Default: 1.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        all_frozen (bool): Frozen all modules in the layer. Default: False.
        inflate (int): Inflate Dims of each block. Default: 1.
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        conv_cfg (dict): Config for conv layers. required keys are ``type``
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type`` and
            ``requires_grad``.
            Default: ``dict(type='BN3d', requires_grad=True)``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU', inplace=True)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
        kwargs (dict, optional): Key arguments for "make_res_layer".
    """

    def __init__(self,
                 depth,
                 pretrained,
                 pretrained2d=True,
                 stage=3,
                 base_channels=64,
                 spatial_stride=2,
                 temporal_stride=1,
                 dilation=1,
                 style='pytorch',
                 all_frozen=False,
                 inflate=1,
                 inflate_style='3x1x1',
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True,
                 **kwargs):

        super().__init__()
        self.arch_settings = ResNet3d.arch_settings
        assert depth in self.arch_settings

        self.make_res_layer = ResNet3d.make_res_layer
        self._inflate_conv_params = ResNet3d._inflate_conv_params
        self._inflate_bn_params = ResNet3d._inflate_bn_params
        self._inflate_weights = ResNet3d._inflate_weights
        self._init_weights = ResNet3d._init_weights

        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.stage = stage
        # stage index is 0 based
        assert 0 <= stage <= 3
        self.base_channels = base_channels

        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation

        self.style = style
        self.all_frozen = all_frozen

        self.stage_inflation = inflate
        self.inflate_style = inflate_style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        block, stage_blocks = self.arch_settings[depth]
        stage_block = stage_blocks[stage]
        planes = 64 * 2**stage
        inplanes = 64 * 2**(stage - 1) * block.expansion

        res_layer = self.make_res_layer(
            block,
            inplanes,
            planes,
            stage_block,
            spatial_stride=spatial_stride,
            temporal_stride=temporal_stride,
            dilation=dilation,
            style=self.style,
            norm_cfg=self.norm_cfg,
            conv_cfg=self.conv_cfg,
            act_cfg=self.act_cfg,
            inflate=self.stage_inflation,
            inflate_style=self.inflate_style,
            with_cp=with_cp,
            **kwargs)

        self.layer_name = f'layer{stage + 1}'
        self.add_module(self.layer_name, res_layer)

    def inflate_weights(self, logger):
        self._inflate_weights(self, logger)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.all_frozen:
            layer = getattr(self, self.layer_name)
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        self._init_weights(self, pretrained)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        res_layer = getattr(self, self.layer_name)
        out = res_layer(x)
        return out

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


if mmdet_imported:
    MMDET_SHARED_HEADS.register_module()(ResNet3dLayer)
    MMDET_BACKBONES.register_module()(ResNet3d)
