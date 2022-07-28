# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import ResLayer


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    """Bottleneck block for ResNet.

    If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
    it is "caffe", the stride-two layer is the first 1x1 conv layer.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """ 已看過，Resnet的瓶頸結構
        Args:
            inplanes: 輸入channel深度
            planes: 輸出channel深度
            stride: 步距
            dilation: 膨脹係數
            downsample: 捷徑分支的下採樣方式
            style: 使用型態
            with_cp: 是否使用checkpoint
            conv_cfg: 卷積層的設定
            norm_cfg: 表準化層的設定
            dcn: 是否使用dcn
            plugins: 是否需要插入一些層結構
            init_cfg: 初始化方式
        """

        # 繼承自BaseModule，初始化繼承對象
        super(Bottleneck, self).__init__(init_cfg)
        # 檢查傳入資料的合法性
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        # 保存傳入的參數
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            # 如果是pytorch型態第一個conv的步距就會是1第二個才是指定的步距
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        # 構建3個表準化層結構，並且這裡都將名稱以及實例對象進行接收
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        # 最後一個conv會將channel擴大expansion倍，所以標準化層也會需要調整
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        # 構建第一層卷積層
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            # 如果是pytorch版本就會是1
            stride=self.conv1_stride,
            bias=False)
        # 添加上標準化層結構
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            # 如果沒有dcn會是這裡
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                # 如果是pytorch版本就會是指定的步距
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            # 如果有使用dcn就會到這裡
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        # 添加標準化層
        self.add_module(self.norm2_name, norm2)
        # 最後一層卷積
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            # 在這裡我們會進行擴圍，會將channel變成輸入的expansion(4)倍
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        # 添加標準化層
        self.add_module(self.norm3_name, norm3)

        # 實例化激活函數
        self.relu = nn.ReLU(inplace=True)
        # 保存捷徑分支的下採樣方式
        self.downsample = downsample

        if self.with_plugins:
            # 如果需要添加其他層結構才會進來
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            # 這裡保存一下x之後用在捷徑分支上
            identity = x
            # 先經過第一層卷積層
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                # 如果有中間插入層會進來
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            # 經過第二層卷積層
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                # 如果有中間插入層會進來
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            # 經過第三層卷積層，這裡先不會進行激活
            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                # 如果有中間插入層會進來
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                # 如果捷徑分支需要downsample會到這裡
                identity = self.downsample(x)

            # 進行捷徑分支的相加
            out += identity

            # 最後輸出
            return out

        if self.with_cp and x.requires_grad:
            # 如果有checkpoint就會到這裡
            out = cp.checkpoint(_inner_forward, x)
        else:
            # 其他狀況會到這裡進行向前傳遞
            out = _inner_forward(x)

        # 將最後結果通過激活函數
        out = self.relu(out)

        # 最後進行回傳
        return out


@BACKBONES.register_module()
class ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
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
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        """ 已看過，Resnet的初始化位置
        Args:
            depth: Resnet的深度，大部分都會是用50
            in_channels: 輸入的channel深度，預設為3表示RGB圖像
            stem_channels: 主幹上面的channel深度，如果沒有要特別設定就用None就會自動與base_channels相同，預設為None
            base_channels: 基礎channel深度，每一層會根據基礎channel深度進行加倍
            num_stages: 總共會有幾個大模塊
            strides: 每個模塊的第一個卷積的步距
            dilations: 每個模塊的膨脹係數
            out_indices: 哪個模塊的輸出要放到最後的輸出
            style: 要用哪種深度學習的框架，預設為pytorch
            deep_stem: 將一開始的7*7卷積核換成3*3卷積層
            avg_down: 使用AvgPooling替代stride=2的卷積進行下採樣
            frozen_stages: 將哪一層進行凍結
            conv_cfg: 卷積的相關設定
            norm_cfg: 標準化層的設定
            norm_eval: 是否需要凍結標準化層當中的均值方差
            dcn: 是否要用dcn卷積
            stage_with_dcn: 哪些層需要用到dcn
            plugins:
            with_cp: 是否有checkpoint
            zero_init_residual: 是否需要將最後的標準化層先都設定成0
            pretrained: 預訓練權重
            init_cfg: 初始化模型設定
        """

        # 繼承自BaseModule，將繼承對象進行初始化
        super(ResNet, self).__init__(init_cfg)
        # 保存zero_init_residual
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            # 如果設定的深度不再arch_settings當中就會直接報錯，這裡提供[18, 34, 50, 101, 152]可以選擇
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        # 不可以同時設定init_cfg與pretrained，只能選擇一個進行模型初始化
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            # 如果將預訓練權重放在pretrained上面就會跳出警告，這裡我們希望放到init_cfg當中
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            # 將預訓練權重資料放到init_cfg當中
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                # 如果沒有設定pretrained也沒有init_cfg就會設定成某些特殊的初始化方式
                self.init_cfg = [
                    # 卷積層部分就會用Kaiming的方式隨機初始化
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        # 如果是標準化層就是全部設定成1
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        # 如果有要設定成0就會在這裡進行指定
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            # 其他的初始化狀態就會直接報錯
            raise TypeError('pretrained must be a str or None')

        # 保存一些傳入的參數
        self.depth = depth
        if stem_channels is None:
            # 如果沒有特別設定stem_channels就會與base_channels相同
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        # 模塊數量需要在1到4之間
        assert 1 <= num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        # 這幾個的數量要相同
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        # 輸出的層數不可以大於總層數
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        # 透過depth獲取block與stage_blocks資訊
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        # 構建第一層卷積層以及池化層，傳入輸入的channel與基礎channel深度
        self._make_stem_layer(in_channels, stem_channels)

        # 剩下的一系列層結構的名稱會到保存到這裡
        self.res_layers = []
        # 遍歷總共會實例化幾個大模塊
        for i, num_blocks in enumerate(self.stage_blocks):
            # 取出第一個卷積層的步距
            stride = strides[i]
            # 獲取膨漲係數，這裡基本上都會是1，也就是不啟用膨脹卷積
            dilation = dilations[i]
            # 是否要用dcn
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                # 如果沒有要添加其他東西就會是None
                stage_plugins = None
            # 通過此層後的channel深度
            planes = base_channels * 2**i
            # 透過make_res_layer構建層結構
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            # 下層的channel會是當前的channel加深expansion(4)倍
            self.inplanes = planes * self.block.expansion
            # 給當前層一個名稱
            layer_name = f'layer{i + 1}'
            # 將層結構放入到模型當中
            self.add_module(layer_name, res_layer)
            # 將名稱保留下來，這樣在forward當中才可以進行呼叫
            self.res_layers.append(layer_name)

        # 看哪些層結構需要進行參數凍結
        self._freeze_stages()

        # 透過resnet過後的輸出的channel深度
        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        # 已看過，構建resnet當中一層的結構
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        """ 已看過，構建resnet最一開始的卷積層
        Args:
            in_channels: 輸入的channel深度
            stem_channels: 輸出的channel深度
        """
        if self.deep_stem:
            # 如果有設定deep_stem就會使用3*3卷積而不是用7*7卷積
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            # 原版的resnet第一層就會使用7*7的卷積
            self.conv1 = build_conv_layer(
                # 透過build_conv_layer構建卷積層，這裡會進行2被下採樣
                # 這裡沒有設定conv_cfg就會直接使用Conv2d進行實例化
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            # 獲取標準化層
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            # 透過add_module添加到模型當中
            self.add_module(self.norm1_name, norm1)
            # 構建激活函數
            self.relu = nn.ReLU(inplace=True)
        # 根據原始resnet，這裡會透過MaxPool進行2倍下採樣
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        # 已看過，將指定的層數量進行學習率凍結
        if self.frozen_stages >= 0:
            # 如果frozen_stages大於0表示需要進行凍結
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                # 將第一層的標準化層變成驗證模式
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    # 將標準化層中的均值方差以及a,b進行凍結
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            # 凍結層結構的訓練
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        # 已看過，Resnet的forward函數
        if self.deep_stem:
            # 如果有使用deep_stem就會到這裡，stem會是一個3*3的卷積
            x = self.stem(x)
        else:
            # 正常版本的resnet會到這裡，conv1會是一個7*7的卷積
            # 這裡會先進行一次2倍下採樣
            x = self.conv1(x)
            # 經過標準化層
            x = self.norm1(x)
            # 經過激活函數
            x = self.relu(x)
        # 通過最大池化下採樣，這裡會進行一次2倍下採樣
        x = self.maxpool(x)
        # 後面就是一系列的層結構，並且會有我們需要的輸出
        outs = []
        # 遍歷每一層層結構
        for i, layer_name in enumerate(self.res_layers):
            # 獲取層結構實例對象
            res_layer = getattr(self, layer_name)
            # 將模型傳入進行正向傳遞
            x = res_layer(x)
            if i in self.out_indices:
                # 這裡我們只需要最後一層的輸出
                outs.append(x)
        # 將結果返回
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module()
class ResNetV1d(ResNet):
    r"""ResNetV1d variant described in `Bag of Tricks
    <https://arxiv.org/pdf/1812.01187.pdf>`_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
