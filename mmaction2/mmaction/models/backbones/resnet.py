# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
from torch.utils import checkpoint as cp

from ...utils import get_root_logger
from ..builder import BACKBONES


class BasicBlock(nn.Module):
    """Basic block for ResNet.

    Args:
        inplanes (int): Number of channels for the input in first conv2d layer.
        planes (int): Number of channels produced by some norm/conv2d layers.
        stride (int): Stride in the conv layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False):
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.style = style
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        assert not with_cp

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.

    Args:
        inplanes (int):
            Number of channels for the input feature in first conv layer.
        planes (int):
            Number of channels produced by some norm layes and conv layers
        stride (int): Spatial stride in the conv layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_cp=False):
        super().__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   conv_cfg=None,
                   norm_cfg=None,
                   act_cfg=None,
                   with_cp=False):
    """Build residual layer for ResNet.

    Args:
        block: (nn.Module): Residual module to be built.
        inplanes (int): Number of channels for the input feature in each block.
        planes (int): Number of channels for the output feature in each block.
        blocks (int): Number of residual blocks.
        stride (int): Stride in the conv layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        conv_cfg (dict | None): Config for norm layers. Default: None.
        norm_cfg (dict | None): Config for norm layers. Default: None.
        act_cfg (dict | None): Config for activate layers. Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        nn.Module: A residual layer for the given config.
    """
    # 已看過，構建stage層結構模型
    # block = 使用的block模塊類
    # inplanes = 輸入的channel深度
    # planes = 當前stage的基礎channel深度
    # blocks = block的堆疊數量
    # stride = 第一個block的
    # dilation = 膨脹係數
    # style = 底層使用的模組
    # conv_cfg = 卷積層配置
    # norm_cfg = 標準化層配置
    # act_cfg = 激活函數層配置
    # with_cp = 是否啟用checkpoint

    # 先將downsample部分設定成None
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        # 如果步距設定非1或是channel有改變就需要透過downsample調整殘差部分
        # downsample會是卷積標準化
        downsample = ConvModule(
            inplanes,
            planes * block.expansion,
            kernel_size=1,
            stride=stride,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    # block保存地方
    layers = list()
    layers.append(
        # 構建第一個block資料
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            style=style,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_cp=with_cp))
    # 更新channel深度
    inplanes = planes * block.expansion
    # 構建剩下的層結構資料
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation,
                style=style,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp))

    # 最後包裝到Sequential上
    return nn.Sequential(*layers)


@BACKBONES.register_module()
class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str | None): Name of pretrained model. Default: None.
        in_channels (int): Channel num of input features. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (Sequence[int]): Indices of output feature. Default: (3, ).
        dilations (Sequence[int]): Dilation of each stage.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: ``pytorch``.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default: -1.
        conv_cfg (dict): Config for norm layers. Default: dict(type='Conv').
        norm_cfg (dict):
            Config for norm layers. required keys are `type` and
            `requires_grad`. Default: dict(type='BN2d', requires_grad=True).
        act_cfg (dict): Config for activate layers.
            Default: dict(type='ReLU', inplace=True).
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        partial_bn (bool): Whether to use partial bn. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 pretrained=None,
                 torchvision_pretrain=True,
                 in_channels=3,
                 num_stages=4,
                 out_indices=(3, ),
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 style='pytorch',
                 frozen_stages=-1,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_eval=False,
                 partial_bn=False,
                 with_cp=False):
        """ 已看過，普通resnet的構建
        Args:
            depth: resnet的深度
            pretrained: 預訓練權重資料
            torchvision_pretrain: 是否使用torchvision提供的預訓練全權重
            in_channels: 輸入的channel深度
            num_stages: 總共堆疊的stage數量
            out_indices: 哪些stage的特徵圖要進行輸出
            strides: 每個stage第一個卷積層的步距
            dilations: 每個stage的膨脹係數
            style: 模型的風格，這裡通常預設都會是pytorch
            frozen_stages: 設定哪些層結構進行凍結
            conv_cfg: 卷積層的相關設定
            norm_cfg: 標準化相關設定
            act_cfg: 激活函數相關設定
            norm_eval: 是否需要將標準差以及均值固定
            partial_bn: 是否使用部分標準化層
            with_cp: 是否啟用checkpoint
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        if depth not in self.arch_settings:
            # 如果設定的resnet深度不在指定的幾種當中就會報錯
            raise KeyError(f'invalid depth {depth} for resnet')
        # 將傳入的參數保存
        self.depth = depth
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.torchvision_pretrain = torchvision_pretrain
        self.num_stages = num_stages
        # 檢查num_stages的值是否合法
        assert 1 <= num_stages <= 4
        self.out_indices = out_indices
        # 輸出的數量不可以比stage數量多
        assert max(out_indices) < num_stages
        self.strides = strides
        self.dilations = dilations
        # strides與dilations與num_stages是配套的，所以數量需要一樣
        assert len(strides) == len(dilations) == num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.partial_bn = partial_bn
        self.with_cp = with_cp

        # 根據設定的resnet深度，獲取適用的block與每個stage的block數量
        self.block, stage_blocks = self.arch_settings[depth]
        # 根據需要的stage數量獲取前blocks的設定
        self.stage_blocks = stage_blocks[:num_stages]
        # 通過stem層後的channel深度
        self.inplanes = 64

        # 構建stem層結構
        self._make_stem_layer()

        # 已看過，構建剩下的stage層結構
        self.res_layers = []
        # 構建stage層結構
        for i, num_blocks in enumerate(self.stage_blocks):
            # 獲取當前stage當中的block需要堆疊多少層
            stride = strides[i]
            # 獲取膨脹係數
            dilation = dilations[i]
            # 獲取當層stage的channel基礎深度
            planes = 64 * 2**i
            # 構建stage層結構實例對象
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp)
            # 更新下層輸入的channel深度
            self.inplanes = planes * self.block.expansion
            # 構建層結構名稱
            layer_name = f'layer{i + 1}'
            # 透過add_module將層結構放到模型當中
            self.add_module(layer_name, res_layer)
            # 將層結構名稱放到res_layers當中
            self.res_layers.append(layer_name)

        # 紀錄最後輸出的channel深度
        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a pooling layer."""
        # 已看過，構建resnet當中的stem層結構，卷積以及激活以及標準化最後通過池化下採樣
        self.conv1 = ConvModule(
            self.in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    @staticmethod
    def _load_conv_params(conv, state_dict_tv, module_name_tv,
                          loaded_param_names):
        """Load the conv parameters of resnet from torchvision.

        Args:
            conv (nn.Module): The destination conv module.
            state_dict_tv (OrderedDict): The state dict of pretrained
                torchvision model.
            module_name_tv (str): The name of corresponding conv module in the
                torchvision model.
            loaded_param_names (list[str]): List of parameters that have been
                loaded.
        """

        weight_tv_name = module_name_tv + '.weight'
        if conv.weight.data.shape == state_dict_tv[weight_tv_name].shape:
            conv.weight.data.copy_(state_dict_tv[weight_tv_name])
            loaded_param_names.append(weight_tv_name)

        if getattr(conv, 'bias') is not None:
            bias_tv_name = module_name_tv + '.bias'
            if conv.bias.data.shape == state_dict_tv[bias_tv_name].shape:
                conv.bias.data.copy_(state_dict_tv[bias_tv_name])
                loaded_param_names.append(bias_tv_name)

    @staticmethod
    def _load_bn_params(bn, state_dict_tv, module_name_tv, loaded_param_names):
        """Load the bn parameters of resnet from torchvision.

        Args:
            bn (nn.Module): The destination bn module.
            state_dict_tv (OrderedDict): The state dict of pretrained
                torchvision model.
            module_name_tv (str): The name of corresponding bn module in the
                torchvision model.
            loaded_param_names (list[str]): List of parameters that have been
                loaded.
        """

        for param_name, param in bn.named_parameters():
            param_tv_name = f'{module_name_tv}.{param_name}'
            param_tv = state_dict_tv[param_tv_name]
            if param.data.shape == param_tv.shape:
                param.data.copy_(param_tv)
                loaded_param_names.append(param_tv_name)

        for param_name, param in bn.named_buffers():
            param_tv_name = f'{module_name_tv}.{param_name}'
            # some buffers like num_batches_tracked may not exist
            if param_tv_name in state_dict_tv:
                param_tv = state_dict_tv[param_tv_name]
                if param.data.shape == param_tv.shape:
                    param.data.copy_(param_tv)
                    loaded_param_names.append(param_tv_name)

    def _load_torchvision_checkpoint(self, logger=None):
        """Initiate the parameters from torchvision pretrained checkpoint."""
        state_dict_torchvision = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_torchvision:
            state_dict_torchvision = state_dict_torchvision['state_dict']

        loaded_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if 'downsample' in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace('conv', 'bn')
                self._load_conv_params(module.conv, state_dict_torchvision,
                                       original_conv_name, loaded_param_names)
                self._load_bn_params(module.bn, state_dict_torchvision,
                                     original_bn_name, loaded_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(
            state_dict_torchvision.keys()) - set(loaded_param_names)
        if remaining_names:
            logger.info(
                f'These parameters in pretrained checkpoint are not loaded'
                f': {remaining_names}')

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            if self.torchvision_pretrain:
                # torchvision's
                self._load_torchvision_checkpoint(logger)
            else:
                # ours
                load_checkpoint(
                    self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input samples extracted
            by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.bn.eval()
            for m in self.conv1.modules():
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _partial_bn(self):
        logger = get_root_logger()
        logger.info('Freezing BatchNorm2D except the first one.')
        count_bn = 0
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                count_bn += 1
                if count_bn >= 2:
                    m.eval()
                    # shutdown update in frozen mode
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
        if mode and self.partial_bn:
            self._partial_bn()
