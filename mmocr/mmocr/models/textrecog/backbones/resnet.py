# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import ConvModule, build_plugin_layer
from mmcv.runner import BaseModule, Sequential

import mmocr.utils as utils
from mmocr.models.builder import BACKBONES
from mmocr.models.textrecog.layers import BasicBlock


@BACKBONES.register_module()
class ResNet(BaseModule):
    """
    Args:
        in_channels (int): Number of channels of input image tensor.
        stem_channels (list[int]): List of channels in each stem layer. E.g.,
            [64, 128] stands for 64 and 128 channels in the first and second
            stem layers.
        block_cfgs (dict): Configs of block
        arch_layers (list[int]): List of Block number for each stage.
        arch_channels (list[int]): List of channels for each stage.
        strides (Sequence[int] | Sequence[tuple]): Strides of the first block
            of each stage.
        out_indices (None | Sequence[int]): Indices of output stages. If not
            specified, only the last stage will be returned.
        stage_plugins (dict): Configs of stage plugins
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 stem_channels,
                 block_cfgs,
                 arch_layers,
                 arch_channels,
                 strides,
                 out_indices=None,
                 plugins=None,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d'),
                 ]):
        """ 已看過，這裡是在MASTER的backbone部分，雖然類名是ResNet不過中間有添加其他層結構
        Args:
            in_channels: 輸入的channel深度
            stem_channels: 每層卷積層輸出的channel深度，list[int]
            block_cfgs: 使用到的block模塊，也就是在resnet當中主要的模塊選擇
            arch_layers: 每個stage堆疊block的數量
            arch_channels: 每個stage輸出channel深度
            strides: 每個stage的第一個block的步距
            out_indices: 哪些stage的特徵圖需要進行輸出，如果沒有是None就默認將最後一層特徵圖進行輸出
            plugins: 額外插入的特徵層
            init_cfg: 初始化設定
        """
        # 繼承自BaseModule，將繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)
        # 檢查傳入的in_channels是否為int格式
        assert isinstance(in_channels, int)
        # 檢查stem_channels要不是int就要是list且當中為int
        assert isinstance(stem_channels, int) or utils.is_type_list(
            stem_channels, int)
        # 檢查arch_layers需要是list且當中是int
        assert utils.is_type_list(arch_layers, int)
        # 檢查arch_channels需要是list且當中是int
        assert utils.is_type_list(arch_channels, int)
        # 檢查strides是否符合規定
        assert utils.is_type_list(strides, tuple) or utils.is_type_list(
            strides, int)
        # arch_layers與arch_channels與strides長度需要一樣，因為這是配套的
        assert len(arch_layers) == len(arch_channels) == len(strides)
        # out_indices檢查
        assert out_indices is None or isinstance(out_indices, (list, tuple))

        # 保存out_indices
        self.out_indices = out_indices
        # 構建stem層結構，將輸入的channel以及stem輸出的channel深度傳入
        # 這裡可以對應上論文的前兩個卷積部分
        self._make_stem_layer(in_channels, stem_channels)
        # 獲取總共會有多少個stage
        self.num_stages = len(arch_layers)
        # 先將use_plugins設定成False
        self.use_plugins = False
        # 保存arch_channels資料
        self.arch_channels = arch_channels
        # 保存res_layers的地方
        self.res_layers = []
        if plugins is not None:
            # 如果傳入的plugins有資料就會到這裡
            # 創建兩個空間
            # plugin_ahead_names當中的層結構會在執行stage前先執行，plugin_after_names就是在stage後執行
            self.plugin_ahead_names = []
            self.plugin_after_names = []
            # 將use_plugins設定成True
            self.use_plugins = True
        # 遍歷stage也就是arch層結構
        for i, num_blocks in enumerate(arch_layers):
            # 獲取當前stage的第一個block的步距
            stride = strides[i]
            # 獲取當前stage的輸出channel深度
            channel = arch_channels[i]

            if self.use_plugins:
                # 如果有需要插入層結構就會到這裡透過_make_stage_plugins添加
                # 根據指定執行的位置會放到plugin_ahead_names當中或是plugin_after_names
                self._make_stage_plugins(plugins, stage_idx=i)

            # 構建ResNet層結構
            res_layer = self._make_layer(
                block_cfgs=block_cfgs,
                inplanes=self.inplanes,
                planes=channel,
                blocks=num_blocks,
                stride=stride,
            )
            self.inplanes = channel
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def _make_layer(self, block_cfgs, inplanes, planes, blocks, stride):
        """ 已看過，構建ResNet當中的模塊
        Args:
            block_cfgs: 在resnet當中有BasicBlock或是BottleNeck兩種模塊
            inplanes: 輸入的channel深度
            planes: 輸出的channel深度
            blocks: 需要堆疊堆少層
            stride: 第一層block的步距
        """

        # 保存層結構的地方
        layers = []
        # 先將殘差邊的downsample部分設定成None
        downsample = None
        # 拷貝一份block的設定資料
        block_cfgs_ = block_cfgs.copy()
        if isinstance(stride, int):
            # 如果步距是int，就會用tuple進行包裝
            stride = (stride, stride)

        if stride[0] != 1 or stride[1] != 1 or inplanes != planes:
            # 如果步距不是1或是輸入的channel深度與輸出channel深度不同就會需要downsample層結構
            # 這裡就會透過卷積以及標準化將特徵圖的高寬以及channel對齊
            downsample = ConvModule(
                inplanes,
                planes,
                1,
                stride,
                norm_cfg=dict(type='BN'),
                act_cfg=None)

        if block_cfgs_['type'] == 'BasicBlock':
            # 如果是用的block是BasicBlock就會到這裡
            # 將block變成BasicBlock類
            block = BasicBlock
            # 將type彈出去
            block_cfgs_.pop('type')
        else:
            # 其他的block就會報錯，這裡只有支援BasicBlock，也就是沒有Bottleneck的版本
            raise ValueError('{} not implement yet'.format(block_cfgs_['type']))

        # 將實例化的層結構放到layers當中
        layers.append(
            # 構建第一個block
            block(
                inplanes,
                planes,
                # 這裡的stride會是指定的stride
                stride=stride,
                # 將downsample方式傳入
                downsample=downsample,
                # 剩餘的block_cfgs也會傳入
                **block_cfgs_))
        # 更新下個block的輸入channel深度
        inplanes = planes
        # 將剩下的多層結構創建
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, **block_cfgs_))

        # 最後用Sequential進行包裝
        return Sequential(*layers)

    def _make_stem_layer(self, in_channels, stem_channels):
        """ 已看過，構建stem層結構
        Args:
            in_channels: 輸入的channel深度
            stem_channels: 每層stem輸出的channel深度
        """
        if isinstance(stem_channels, int):
            # 如果傳入的stem_channels是int就會在外面加上list
            stem_channels = [stem_channels]
        # 保存層結構的list
        stem_layers = []
        # 遍歷總共要創建的層結構
        for _, channels in enumerate(stem_channels):
            # 構建卷積以及標準化以及激活函數層結構，這裡有透過padding，所以高寬不會發生變化
            stem_layer = ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU'))
            # 更新下個層結構的輸入channel深度
            in_channels = channels
            # 保存實例對象
            stem_layers.append(stem_layer)
        # 透過Sequential包裝
        self.stem_layers = Sequential(*stem_layers)
        # 記錄下透過stem輸出後的channel深度
        self.inplanes = stem_channels[-1]

    def _make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support inserting ``nn.Maxpooling``,
        ``mmcv.cnn.Convmodule``into the backbone. Originally designed
        for ResNet31-like architectures.

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type="Maxpooling", arg=(2,2)),
            ...          stages=(True, True, False, False),
            ...          position='before_stage'),
            ...     dict(cfg=dict(type="Maxpooling", arg=(2,1)),
            ...          stages=(False, False, True, Flase),
            ...          position='before_stage'),
            ...     dict(cfg=dict(
            ...              type='ConvModule',
            ...              kernel_size=3,
            ...              stride=1,
            ...              padding=1,
            ...              norm_cfg=dict(type='BN'),
            ...              act_cfg=dict(type='ReLU')),
            ...          stages=(True, True, True, True),
            ...          position='after_stage')]

        Suppose ``stage_idx=1``, the structure of stage would be:

        .. code-block:: none

            Maxpooling -> A set of Basicblocks -> ConvModule

        Args:
            plugins (list[dict]): List of plugins cfg to build.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        # 已看過，構建在resnet當中插入的層結構
        # plugins = 需要插入的層結構構建資料，list[ConfigDict]，list長度就會是新增的層結構數量，ConfigDict就是詳細內容
        # stage_idx = 當前是第幾個stage

        # 獲取輸入的channel深度
        in_channels = self.arch_channels[stage_idx]
        # 在plugin_ahead_names與plugin_after_names當中先添加一個空list
        self.plugin_ahead_names.append([])
        self.plugin_after_names.append([])
        # 遍歷plugins當中的層結構資料
        for plugin in plugins:
            # 將資料拷貝一份
            plugin = plugin.copy()
            # 將stages資料取出，如果當中沒有就會是None
            stages = plugin.pop('stages', None)
            # 將position資料取出，如果當中沒有就會是None
            position = plugin.pop('position', None)
            # stages要不是None就要與self.num_stages相同
            assert stages is None or len(stages) == self.num_stages
            if stages[stage_idx]:
                # 如果當前stage的部分是True就會進來，也就是不是每層都會有這個plugins的層結構
                if position == 'before_stage':
                    # 如果position的是設定before_stage就會到這裡，透過build_plugin_layer進行構建層實例化對象
                    name, layer = build_plugin_layer(
                        # 將config資料傳入
                        plugin['cfg'],
                        # 這是為了要命名用的
                        f'_before_stage_{stage_idx+1}',
                        # 給定輸入以及輸出的channel深度
                        in_channels=in_channels,
                        out_channels=in_channels)
                    # 將名稱保存到plugin_ahead_names的當前stage_idx下
                    self.plugin_ahead_names[stage_idx].append(name)
                    # 用add_module添加到模型當中，使用時用name進行呼叫
                    self.add_module(name, layer)
                elif position == 'after_stage':
                    # 如果是要放在after_stage就會到這裡
                    # 這裡與上面相同，只是命名的名稱不同
                    name, layer = build_plugin_layer(
                        plugin['cfg'],
                        f'_after_stage_{stage_idx+1}',
                        in_channels=in_channels,
                        out_channels=in_channels)
                    # 將其保存在after的部分
                    self.plugin_after_names[stage_idx].append(name)
                    self.add_module(name, layer)
                else:
                    # 其他的position就會直接報錯
                    raise ValueError('uncorrected plugin position')

    def forward_plugin(self, x, plugin_name):
        """ 已看過，plugin的層結構前向傳播位置
        Args:
            x: 特徵圖本身，這裡的shape會根據要通過的層結構會有所不同
            plugin_name: 要執行的層結構名稱
        """
        # 保留輸入的特徵圖
        out = x
        for name in plugin_name:
            # 遍歷插入的層結構名稱並且在模型當中找到實例化對象，之後進行正向傳播
            out = getattr(self, name)(out)
        # 最後將結果輸出
        return out

    def forward(self, x):
        """
        Args: x (Tensor): Image tensor of shape :math:`(N, 3, H, W)`.

        Returns:
            Tensor or list[Tensor]: Feature tensor. It can be a list of
            feature outputs at specific layers if ``out_indices`` is specified.
        """
        # 已看過，MASTER的backbone正向傳播部分
        # x = 圖像資料，tensor shape [batch_size, channel, height, width]

        # 先通過兩層的卷積層，這裡圖像高寬不會發生變化，只是進行特徵提取而已
        x = self.stem_layers(x)

        # 每層結構輸出的特徵圖保留
        outs = []
        # 遍歷每層模塊，這裡會獲取每層結構的名稱，在構建時是使用add_module方法添加層結構
        for i, layer_name in enumerate(self.res_layers):
            # 從模型當中獲取對應名子的實例化層結構對象
            res_layer = getattr(self, layer_name)
            if not self.use_plugins:
                # 如果沒有使用plugins的層結構就會到這裡
                # 直接將x放入到當前層結構當中
                x = res_layer(x)
                if self.out_indices and i in self.out_indices:
                    # 如果有指定哪些層結構需要進行輸出，且當前層數需要輸出就會到這裡
                    outs.append(x)
            else:
                # 如果有插入的層結構就會到這裡
                # 取得當前層的前置插入層結構，透過forward_plugin進行前向傳遞
                x = self.forward_plugin(x, self.plugin_ahead_names[i])
                # 進行當前層的前向傳遞
                x = res_layer(x)
                # 取得當前層的後置插入層結構，透過forward_plugin進行前向傳遞
                x = self.forward_plugin(x, self.plugin_after_names[i])
                if self.out_indices and i in self.out_indices:
                    # 如果有指定哪些層結構需要進行輸出，且當前層數需要輸出就會到這裡
                    outs.append(x)

        # 如果有指定輸出的層結構就會輸出outs，否則就是將最後一層結果輸出
        return tuple(outs) if self.out_indices else x
