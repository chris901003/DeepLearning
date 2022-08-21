# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      normal_init)
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .resnet import BasicBlock, Bottleneck, get_expansion
from .utils import load_checkpoint


class HRModule(nn.Module):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=False,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 upsample_cfg=dict(mode='nearest', align_corners=None)):
        """ 構建HRNet當中的層結構
        Args:
            num_branches: 當前stage的分支數量
            blocks: 使用的block類
            num_blocks: 個別分支需要重複堆疊的block數量
            in_channels: 個別分支輸入的channel深度
            num_channels: 個別分支輸出的channel深度
            multiscale_output:
            with_cp: 是否使用checkpoint
            conv_cfg: 卷積層設定資料
            norm_cfg: 標準化層設定資料
            upsample_cfg: 上採樣設定資料
        """

        # Protect mutable default arguments
        # 將標準化設定資料進行保存
        norm_cfg = copy.deepcopy(norm_cfg)
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        # 確認分支設定
        self._check_branches(num_branches, num_blocks, in_channels, num_channels)

        # 將傳入的餐數保留
        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.upsample_cfg = upsample_cfg
        self.with_cp = with_cp
        # 構建分支實例化對象
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        # 構建融合層結構
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _check_branches(num_branches, num_blocks, in_channels, num_channels):
        """ 檢查傳入的分支參數有沒有問題
        Args:
            num_branches: 分支數量
            num_blocks: 每個分支block的堆疊數量
            in_channels: 每個分支的輸入channel深度
            num_channels: 每個分支的數入channel輸出深度
        """
        """Check input to avoid ValueError."""
        # 檢查分支數量與num_blocks的長度是否相同
        if num_branches != len(num_blocks):
            # 如果不相同就會報錯
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)

        # 檢查分支數量與num_channels的長度是否相同
        if num_branches != len(num_channels):
            # 如果不相同就會報錯
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)

        # 檢查分支數量與in_channels的長度是否相同
        if num_branches != len(in_channels):
            # 如果不想同就會報錯
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        """ 構建一個分支結構
        Args:
            branch_index: 當前要構建的是第幾個分支
            block: 使用的層結構的類
            num_blocks: 每個分支需要堆疊block的數量
            num_channels: 每個分支的channel深度
            stride: 步距
        """
        """Make one branch."""
        # 會先將downsample設定成None
        downsample = None
        if stride != 1 or self.in_channels[branch_index] != num_channels[branch_index] * get_expansion(block):
            # 當圖像高寬會發生變化或是channel深度會發生變化就需要透過downsample將殘差邊進行調整
            downsample = nn.Sequential(
                # 構建卷積層
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    # 將channel深度調整到與主幹相同
                    num_channels[branch_index] * get_expansion(block),
                    kernel_size=1,
                    # 步距會與傳入的相同
                    stride=stride,
                    bias=False),
                # 構建標準化層
                build_norm_layer(
                    self.norm_cfg,
                    num_channels[branch_index] * get_expansion(block))[1])

        # 構建剩下的層結構
        layers = list()
        layers.append(
            # 構建第一層block結構
            block(
                # 透過branch_index獲取正確的輸入channel深度
                self.in_channels[branch_index],
                # 輸出的channel深度會是基礎channel深度乘上expansion
                num_channels[branch_index] * get_expansion(block),
                # 步距會是指定的步距
                stride=stride,
                # 殘差邊會需要進行downsample調整高寬以及深度
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        # 更新輸入的channel深度
        self.in_channels[branch_index] = num_channels[branch_index] * get_expansion(block)
        # 構建剩下的block結構
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    # 這裡的channel深度不會發生變化
                    self.in_channels[branch_index],
                    num_channels[branch_index] * get_expansion(block),
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        # 將多層block用Sequential包裝
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """ 構建分支結構
        Args:
            num_branches: 分支數量
            block: block類
            num_blocks: 每個分支需要堆疊的block數量
            num_channels: 每個分支的channel深度
        """
        """Make branches."""
        # 分支層結構保存的list
        branches = []

        # 遍歷分支數量
        for i in range(num_branches):
            # 將構建好的層結構保存
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        # 透過ModuleList將多層結構封裝
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        # 構建融合層
        if self.num_branches == 1:
            # 如果只有一個分支就不會需要融合
            return None

        # 獲取總共有多少分支
        num_branches = self.num_branches
        # 獲取每個分支輸出的channel深度
        in_channels = self.in_channels
        # 保存融合層的list
        fuse_layers = []
        # 如果有設定多尺度輸出最後輸出的分支數量就會是輸入的分支數量，否則就會是1
        num_out_branches = num_branches if self.multiscale_output else 1

        # 遍歷最終要輸出的分支數量
        for i in range(num_out_branches):
            # 保存融合層的list
            fuse_layer = []
            # 遍歷輸入的分支數量
            for j in range(num_branches):
                if j > i:
                    # 如果遍歷到的j大於i就會到這裡，這裡會需要透過上採樣將特徵圖變大
                    fuse_layer.append(
                        # 透過Sequential進行包裝
                        nn.Sequential(
                            # 構建卷積層
                            build_conv_layer(
                                self.conv_cfg,
                                # 不會改變channel深度
                                in_channels[j],
                                in_channels[i],
                                # 這裡會使用1x1的卷積核
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            # 構建標準化模塊
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            # 進行上採樣
                            nn.Upsample(
                                # 設定需要上採樣的倍率
                                scale_factor=2**(j - i),
                                # 根據upsample_cfg選擇差值模式
                                mode=self.upsample_cfg['mode'],
                                align_corners=self.
                                upsample_cfg['align_corners'])))
                elif j == i:
                    # 如果j跟i相等就不需要任合操作，直接使用原先特徵圖就行
                    fuse_layer.append(None)
                else:
                    # 構建下採樣的卷積
                    conv_downsamples = []
                    # 如果是需要透過下採樣就每次通過一個卷積就下採樣2倍，會一直通過卷積直到高寬符合
                    for k in range(i - j):
                        if k == i - j - 1:
                            # 如果到最後一層卷積就會到這裡，差別會是將channel調整到指定深度，以及沒有激活函數層
                            conv_downsamples.append(
                                # 透過Sequential將層結構進行包裝
                                nn.Sequential(
                                    # 卷積層結構
                                    build_conv_layer(
                                        self.conv_cfg,
                                        # 這裡會將channel深度調整到目標深度
                                        in_channels[j],
                                        in_channels[i],
                                        # 透過3x3的卷積核進行卷積
                                        kernel_size=3,
                                        # 步距設定成2進行下採樣
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    # 構建標準化層結構
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            # 其他就會到這裡
                            conv_downsamples.append(
                                # 透過Sequential將多層結構保存
                                nn.Sequential(
                                    # 構建卷積層結構
                                    build_conv_layer(
                                        self.conv_cfg,
                                        # 這裡channel深度不會發生變化
                                        in_channels[j],
                                        in_channels[j],
                                        # 使用3x3的卷積核
                                        kernel_size=3,
                                        # 步距設定成2進行下採樣
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    # 構建標準化層
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    # 構建激活函數層
                                    nn.ReLU(inplace=True)))
                    # 最後多層卷積下採樣用Sequential包裝
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            # 透過ModuleList包裝整個fuse_layer到fuse_layers當中
            fuse_layers.append(nn.ModuleList(fuse_layer))

        # 最後回傳fuse_layers用ModuleList包裝
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        # 進行HRNet的stage模塊向前傳遞
        # x = 特徵圖，list[tensor]，tensor shape [batch_size, channel, height, width]，list長度就會是stage分支數量
        if self.num_branches == 1:
            # 如果分支數量只有一個就會到這裡，直接將x[0]通過branches層結構
            return [self.branches[0](x[0])]

        # 遍歷分支數量
        for i in range(self.num_branches):
            # 將每個分支的特徵圖通過對應的branches層結構
            x[i] = self.branches[i](x[i])

        # 這裡就會是將不同高寬以及channel深度的特徵圖進行融合
        x_fuse = []
        # 遍歷融合的層結構
        for i in range(len(self.fuse_layers)):
            # 將y歸0
            y = 0
            # 遍歷分支數量
            for j in range(self.num_branches):
                if i == j:
                    # 如果當前的i與j相同就直接相加
                    y += x[j]
                else:
                    # 透過指定的層結構將特徵圖進行調整
                    y += self.fuse_layers[i][j](x[j])
            # 將y通過激活函數後放到x_fuse當中
            x_fuse.append(self.relu(y))
        # 最後回傳融合過的特徵圖，list[tensor]，list長度是該stage的分支數量，tensor shape [batch_size, channel, height, width]
        return x_fuse


@BACKBONES.register_module()
class HRNet(nn.Module):
    """HRNet backbone.

    `High-Resolution Representations for Labeling Pixels and Regions
    <https://arxiv.org/abs/1904.04514>`__

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Default: 3.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.

    Example:
        >>> from mmpose.models import HRNet
        >>> import torch
        >>> extra = dict(
        >>>     stage1=dict(
        >>>         num_modules=1,
        >>>         num_branches=1,
        >>>         block='BOTTLENECK',
        >>>         num_blocks=(4, ),
        >>>         num_channels=(64, )),
        >>>     stage2=dict(
        >>>         num_modules=1,
        >>>         num_branches=2,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4),
        >>>         num_channels=(32, 64)),
        >>>     stage3=dict(
        >>>         num_modules=4,
        >>>         num_branches=3,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4),
        >>>         num_channels=(32, 64, 128)),
        >>>     stage4=dict(
        >>>         num_modules=3,
        >>>         num_branches=4,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4, 4),
        >>>         num_channels=(32, 64, 128, 256)))
        >>> self = HRNet(extra, in_channels=1)
        >>> self.eval()
        >>> inputs = torch.rand(1, 1, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 32, 8, 8)
    """

    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False,
                 frozen_stages=-1):
        """ HRNet的初始化函數
        Args:
            extra: 每個stage的詳細模型配置參數
            in_channels: 輸入的channel深度
            conv_cfg: 卷積層設定資料
            norm_cfg: 標準化層設定資料
            norm_eval: 是否需要凍結標準差以及均值
            with_cp: 是否使用checkpoint
            zero_init_residual: 在最後的標準化層結構是否需要初始化成0
            frozen_stages: 需要凍結的層結構，如果是-1就全部都不會凍結
        """
        # Protect mutable default arguments
        # 深度拷貝一份標準化層結構設定資料
        norm_cfg = copy.deepcopy(norm_cfg)
        # 繼承自nn.Module，對繼承對象進行初始化
        super().__init__()
        # 保存傳入的參數
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.frozen_stages = frozen_stages

        # stem net
        # 主幹網路構建
        # 構建兩個標準化層結構且輸入的channel深度都會是2，這裡會獲取到兩個標準化實例對象以及個別的名稱
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        # 構建第一個卷積層結構
        self.conv1 = build_conv_layer(
            # 將卷積層設定資料傳入，如果是None就會默認使用Conv2d
            self.conv_cfg,
            # 將輸入的channel深度傳入
            in_channels,
            # 輸出的channel深度
            64,
            kernel_size=3,
            # 會進行2倍下採樣
            stride=2,
            padding=1,
            # 不使用偏置
            bias=False)

        # 將第一層標準化層結構透過add_module添加到模型當中，這裡之後就需要透過指定名稱進行呼叫
        self.add_module(self.norm1_name, norm1)
        # 構建第二個卷積層結構
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            # 這裡的輸入channel深度與輸出channel深度相同
            64,
            64,
            kernel_size=3,
            # 會進行2倍下採樣
            stride=2,
            padding=1,
            # 將偏置關閉
            bias=False)

        # 透過add_module將第二個標準化層結構添加到模型當中
        self.add_module(self.norm2_name, norm2)
        # 構建激活函數，這裡使用的會是ReLU激活函數
        self.relu = nn.ReLU(inplace=True)

        # 獲取上採樣的設定，如果extra當中沒有指定的upsample參數就會使用默認
        self.upsample_cfg = self.extra.get('upsample', {
            'mode': 'nearest',
            'align_corners': None
        })

        # stage 1
        # 獲取stage1的層結構設定資料
        self.stage1_cfg = self.extra['stage1']
        # 獲取stage1的基礎channel深度
        num_channels = self.stage1_cfg['num_channels'][0]
        # 獲取使用的block名稱，這裡會是BASIC或是BOTTLENECK兩種其中之一
        block_type = self.stage1_cfg['block']
        # 獲取在stage1總共需要堆疊多少層block
        num_blocks = self.stage1_cfg['num_blocks'][0]

        # 透過指定的block_type獲取對應的block類
        block = self.blocks_dict[block_type]
        # 獲取在block當中會將channel擴大多少倍，就可以獲取最終從stage1輸出的channel深度
        stage1_out_channels = num_channels * get_expansion(block)
        # 構建stage1的實例對象
        self.layer1 = self._make_layer(block, 64, stage1_out_channels, num_blocks)

        # stage 2
        # 獲取stage2的設定資料
        self.stage2_cfg = self.extra['stage2']
        # 獲取stage2的基礎channel深度
        num_channels = self.stage2_cfg['num_channels']
        # 獲取stage2使用的block名稱
        block_type = self.stage2_cfg['block']

        # 根據block名稱獲取對應的類
        block = self.blocks_dict[block_type]
        # 獲取輸出的channel深度，這裡產生list型態的資料，長度會是與num_channels相同
        num_channels = [channel * get_expansion(block) for channel in num_channels]
        # 構建過渡層結構
        self.transition1 = self._make_transition_layer([stage1_out_channels], num_channels)
        # 構建stage2的層結構
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        # stage 3
        # 獲取stage3的配置
        self.stage3_cfg = self.extra['stage3']
        # 獲取stage3的基礎channel深度
        num_channels = self.stage3_cfg['num_channels']
        # 獲取使用的block名稱
        block_type = self.stage3_cfg['block']

        # 根據名稱獲取block的類
        block = self.blocks_dict[block_type]
        # 將當前stage的基礎channel乘上expansion獲取最終輸出的channel深度
        num_channels = [channel * get_expansion(block) for channel in num_channels]
        # 構建過渡層結構
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        # 構建stage3多層結構
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        # stage 4
        # 獲取stage4的配置資料
        self.stage4_cfg = self.extra['stage4']
        # 獲取stage4當中每個分支的基礎channel深度
        num_channels = self.stage4_cfg['num_channels']
        # 獲取使用的block名稱
        block_type = self.stage4_cfg['block']

        # 透過名稱獲取block類
        block = self.blocks_dict[block_type]
        # 透過基礎的channel深度乘上expansion獲取最後的深度
        num_channels = [channel * get_expansion(block) for channel in num_channels]
        # 構建過渡結構
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)

        # 構建stage4的層結構
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multiscale_output=self.stage4_cfg.get('multiscale_output', False))

        # 將設定的層結構進行凍結
        self._freeze_stages()

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """ 構建過渡層結構
        Args:
            num_channels_pre_layer: 上層輸出的channel深度，這裡會是list，list長度表示輸出的特徵圖數量
            num_channels_cur_layer: 當前輸出的channel深度，這裡會是list，list長度表示輸出的特徵圖數量
        """
        """Make transition layer."""
        # 獲取當前層的特徵圖數量
        num_branches_cur = len(num_channels_cur_layer)
        # 獲取上層特徵圖數量
        num_branches_pre = len(num_channels_pre_layer)

        # 保存多層結構的list
        transition_layers = []
        # 遍歷當前層的數量
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                # 如果當前的i小於上層的特徵圖數量時就會到這裡
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    # 如果當前層的channel深度與上層的channel深度無法對齊就會到這裡
                    transition_layers.append(
                        # 使用Sequential將多層結構包裝
                        nn.Sequential(
                            # 構建卷積層
                            build_conv_layer(
                                self.conv_cfg,
                                # 將上層的channel深度調整到與當前層的channel深度相同
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                # 這裡使用的是3x3的卷積核
                                kernel_size=3,
                                # 步距為1
                                stride=1,
                                padding=1,
                                bias=False),
                            # 構建標準化層結構
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            # 設定激活函數
                            nn.ReLU(inplace=True)))
                else:
                    # 如果channel深度相同就不需要卷積層來調整channel深度
                    transition_layers.append(None)
            else:
                # 如果當前i超過上層的特徵圖數量就會到這裡
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    # 這裡的輸入channel深度會是最後上層的channel深度
                    in_channels = num_channels_pre_layer[-1]
                    # 輸出的channel深度會根據j決定
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        # 將多層結構放到Sequential當中
                        nn.Sequential(
                            # 構建卷積層
                            build_conv_layer(
                                self.conv_cfg,
                                # 將channel進行調整
                                in_channels,
                                out_channels,
                                # 這裡使用的是3x3的卷積核
                                kernel_size=3,
                                # 步距設定成2，表示會進行下採樣
                                stride=2,
                                padding=1,
                                bias=False),
                            # 構建表準化層
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            # 構建激活函數層
                            nn.ReLU(inplace=True)))
                # 最終保存到transition_layers當中
                transition_layers.append(nn.Sequential(*conv_downsamples))

        # 將transition_layers用ModuleList進行包裝
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """ 構建stage層結構
        Args:
            block: stage當中需要重複堆疊的模塊
            in_channels: 輸入的channel深度
            out_channels: 輸出的channel深度
            blocks: 總共需要堆疊多少層的block
            stride: 步距
        """
        # 先將downsample設定成None，預設是不會需要使用到downsample
        downsample = None
        if stride != 1 or in_channels != out_channels:
            # 如果會將高寬進行調整或是channel深度會調整時就會需要downsample，將殘差結構上的特徵圖進行調整
            downsample = nn.Sequential(
                # 這裡會使用卷積加上標準化將殘差邊上的特徵圖進行調整
                build_conv_layer(
                    self.conv_cfg,
                    # 會將channel維度調整到out_channels
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, out_channels)[1])

        layers = list()
        # 構建stage當中的第一個block模塊
        layers.append(
            block(
                # 將channel深度調整到輸出的channel深度
                in_channels,
                out_channels,
                # 步距使用的是指定的步距
                stride=stride,
                # 這裡會將downsample方式傳入
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        # 構建剩下的block模塊
        for _ in range(1, blocks):
            layers.append(
                block(
                    # 這裡的channel深度不會發生變化
                    out_channels,
                    out_channels,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        # 最後透過Sequential包裝
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        """ 構建stage層結構
        Args:
            layer_config: 層結構的設定資料
            in_channels: 輸入的channel深度
            multiscale_output: 是否使用多尺度輸出
        """
        """Make stage."""
        # 該stage整體需要重複堆疊多少次
        num_modules = layer_config['num_modules']
        # 該stage總共有多少分支
        num_branches = layer_config['num_branches']
        # 獲取個別分支需要堆疊block的數量
        num_blocks = layer_config['num_blocks']
        # 獲取個別分支的channel深度
        num_channels = layer_config['num_channels']
        # 根據layer_config當中指定的block名稱獲取對應的類
        block = self.blocks_dict[layer_config['block']]

        # 保存模型的list
        hr_modules = []
        # 遍歷需要整個stage堆疊的次數
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                # 如果沒有設定多尺度輸出且當前遍歷到最後一層就會將reset_multiscale_output設定成False
                reset_multiscale_output = False
            else:
                # 其他情況就會設定成True
                reset_multiscale_output = True

            # 構建層結構
            hr_modules.append(
                # 構建HRModule層結構
                HRModule(
                    # 總共有多少分支
                    num_branches,
                    # 使用的block類
                    block,
                    # 個別分支需要堆疊多少層block
                    num_blocks,
                    # 個別分支輸入的channel深度
                    in_channels,
                    # 個別分支輸出的channel深度
                    num_channels,
                    # reset_multiscale_output設定
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    upsample_cfg=self.upsample_cfg))

            # 更新輸入的channel深度，這裡會是最後一層的in_channels資訊
            in_channels = hr_modules[-1].in_channels

        # 最後將多層結構用Sequential包裝，並且回傳最後的channel深度
        return nn.Sequential(*hr_modules), in_channels

    def _freeze_stages(self):
        # 凍結指定的層結構
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            # 如果frozen_stages大於等於0就會到這裡
            # 將norm1以及norm2設定成驗證模式
            self.norm1.eval()
            self.norm2.eval()

            # 遍歷前兩層卷積以及標準化層
            for m in [self.conv1, self.norm1, self.conv2, self.norm2]:
                for param in m.parameters():
                    # 將當中的權重值都變成不計算反向傳遞值
                    param.requires_grad = False

        # 遍歷指定需要凍結的層結構
        for i in range(1, self.frozen_stages + 1):
            if i == 1:
                # 如果i=1就會到這裡
                m = getattr(self, 'layer1')
            else:
                # 其他就會到這裡
                m = getattr(self, f'stage{i}')

            # 將獲取到的層結構設定成驗證模式
            m.eval()
            for param in m.parameters():
                # 並且將當中的反向傳遞計算關閉
                param.requires_grad = False

            if i < 4:
                # 如果i<4就會到這裡
                m = getattr(self, f'transition{i}')
                # 將過渡層結構設定成驗證模式
                m.eval()
                for param in m.parameters():
                    # 將反向傳遞關閉
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # HRNet的forward函數
        # x = 圖像資料，tensor shape [batch_size, channel, height, width]

        # 進行第一層卷積
        x = self.conv1(x)
        # 進行第一層標準化
        x = self.norm1(x)
        # 進行激活函數
        x = self.relu(x)
        # 進行第二層卷積
        x = self.conv2(x)
        # 進行第二層標準化
        x = self.norm2(x)
        # 進行激活函數
        x = self.relu(x)
        # 通過layer1模塊
        x = self.layer1(x)

        # 保存輸出的特徵圖
        x_list = []
        # 遍歷stage2的分支數量
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                # 如果中間過渡層結構不是None就會到這裡，將通過過渡層結構的特徵圖保留到x_list當中
                x_list.append(self.transition1[i](x))
            else:
                # 如果不需要通過過渡層結構就直接放到x_list當中
                x_list.append(x)
        # 通過stage2層結構
        y_list = self.stage2(x_list)

        # 將x_list清空
        x_list = []
        # 遍歷stage3的分支數量
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                # 如果當前的index有過渡層結構
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                # 沒有的話就直接放到x_list
                x_list.append(y_list[i])
        # 將特徵圖通過stage3層結構
        y_list = self.stage3(x_list)

        # 將x_list清空
        x_list = []
        # 遍歷stage4當中的分支結構
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        # 將特徵圖通過stage4層結構
        y_list = self.stage4(x_list)

        # 最終y_list shape = [batch_size, channel, height, width]
        return y_list

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
