# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_upsample_layer, constant_init,
                      normal_init)

from mmpose.models.builder import build_loss
from ..backbones.resnet import BasicBlock
from ..builder import HEADS


@HEADS.register_module()
class AEHigherResolutionHead(nn.Module):
    """Associative embedding with higher resolution head. paper ref: Bowen
    Cheng et al. "HigherHRNet: Scale-Aware Representation Learning for Bottom-
    Up Human Pose Estimation".

    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joints
        tag_per_joint (bool): If tag_per_joint is True,
            the dimension of tags equals to num_joints,
            else the dimension of tags is 1. Default: True
        extra (dict): Configs for extra conv layers. Default: None
        num_deconv_layers (int): Number of deconv layers.
            num_deconv_layers should >= 0. Note that 0 means
            no deconv layers.
        num_deconv_filters (list|tuple): Number of filters.
            If num_deconv_layers > 0, the length of
        num_deconv_kernels (list|tuple): Kernel sizes.
        cat_output (list[bool]): Option to concat outputs.
        with_ae_loss (list[bool]): Option to use ae loss.
        loss_keypoint (dict): Config for loss. Default: None.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 tag_per_joint=True,
                 extra=None,
                 num_deconv_layers=1,
                 num_deconv_filters=(32, ),
                 num_deconv_kernels=(4, ),
                 num_basic_blocks=4,
                 cat_output=None,
                 with_ae_loss=None,
                 loss_keypoint=None):
        """ 具有更高分辨率頭的關聯嵌入
        Args:
            in_channels: 輸入的channel深度
            num_joints: 最後的分類數量
            tag_per_joint: 如果是True表示tag的維度與num_points相同，如果是False的話tag就是1
            extra: 層結構設定資料
            num_deconv_layers: 轉置卷積的數量
            num_deconv_filters: 過濾器數量
            num_deconv_kernels: 卷積核大小
            num_basic_blocks: basic_block數量
            cat_output: 是否需要將結果進行concat
            with_ae_loss: 是否使用ae損失
            loss_keypoint: 使用的loss計算方式
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()

        # 根據指定的loss計算方式
        self.loss = build_loss(loss_keypoint)
        # 如果tag_per_joint是True就dim_tag就會是num_joints否則就會是1
        dim_tag = num_joints if tag_per_joint else 1

        # 保存傳入參數
        self.num_deconvs = num_deconv_layers
        self.cat_output = cat_output

        # 最終輸出的channel深度保存
        final_layer_output_channels = []

        if with_ae_loss[0]:
            # 如果第一層有使用ae損失就會到這裡，輸出的channel深度會是num_joints加上dim_tag
            out_channels = num_joints + dim_tag
        else:
            # 其他就會是num_joints
            out_channels = num_joints

        # 將輸出的channel深度進行保存
        final_layer_output_channels.append(out_channels)
        # 遍歷deconv的層數
        for i in range(num_deconv_layers):
            if with_ae_loss[i + 1]:
                # 如果i+1的地方有使用ae損失就會到這裡，輸出的channel深度就會是num_joints加上dim_tag
                out_channels = num_joints + dim_tag
            else:
                # 否則輸出channel就會是num_joints
                out_channels = num_joints
            # 保存輸出的channel深度
            final_layer_output_channels.append(out_channels)

        # 構建deconv的輸出channel深度
        deconv_layer_output_channels = []
        for i in range(num_deconv_layers):
            # 根據是否使用ae損失會有不同的輸出channel深度
            if with_ae_loss[i]:
                out_channels = num_joints + dim_tag
            else:
                out_channels = num_joints
            deconv_layer_output_channels.append(out_channels)

        # 構建最後的層結構
        self.final_layers = self._make_final_layers(
            in_channels, final_layer_output_channels, extra, num_deconv_layers,
            num_deconv_filters)
        # 構建deconv的層結構
        self.deconv_layers = self._make_deconv_layers(
            in_channels, deconv_layer_output_channels, num_deconv_layers,
            num_deconv_filters, num_deconv_kernels, num_basic_blocks,
            cat_output)

    @staticmethod
    def _make_final_layers(in_channels, final_layer_output_channels, extra,
                           num_deconv_layers, num_deconv_filters):
        """ 構建最後的層結構
        Args:
            in_channels: 輸入的channel深度
            final_layer_output_channels: 最終每層輸出的channel深度
            extra: 層結構設定參數
            num_deconv_layers: 使用deconv的層數
            num_deconv_filters: 過濾器數量
        """
        """Make final layers."""
        if extra is not None and 'final_conv_kernel' in extra:
            # 如果extra當中有final_conv_kernel就會進來
            # 檢查final_conv_kernel需要是1或是3
            assert extra['final_conv_kernel'] in [1, 3]
            if extra['final_conv_kernel'] == 3:
                # 如果卷積核是3就會需要1的padding
                padding = 1
            else:
                # 如果卷積核是1就不需要padding
                padding = 0
            # 將kernel_size提取出來
            kernel_size = extra['final_conv_kernel']
        else:
            # 如果沒有設定就默認為卷積核為1x1
            kernel_size = 1
            padding = 0

        # 最終層的保存list
        final_layers = list()
        final_layers.append(
            # 構建卷積層
            build_conv_layer(
                cfg=dict(type='Conv2d'),
                # 將channel深度調整到指定深度
                in_channels=in_channels,
                out_channels=final_layer_output_channels[0],
                kernel_size=kernel_size,
                stride=1,
                padding=padding))

        # 構建剩下的deconv層結構
        for i in range(num_deconv_layers):
            # 獲取輸入的channel深度
            in_channels = num_deconv_filters[i]
            final_layers.append(
                # 構建卷積層結構
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    # 將channel調整到指定深度
                    in_channels=in_channels,
                    out_channels=final_layer_output_channels[i + 1],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))

        # 通過ModuleList進行包裝
        return nn.ModuleList(final_layers)

    def _make_deconv_layers(self, in_channels, deconv_layer_output_channels,
                            num_deconv_layers, num_deconv_filters,
                            num_deconv_kernels, num_basic_blocks, cat_output):
        """ 構建deconv層結構
        Args:
            in_channels: 輸入的channel深度
            deconv_layer_output_channels: deconv的輸出channel深度
            num_deconv_layers: deconv的層數
            num_deconv_filters: deconv的過濾數
            num_deconv_kernels: deconv的卷積核
            num_basic_blocks: 使用basic_block數量
            cat_output: 是否需要將輸出進行concat
        """
        """Make deconv layers."""
        # 保存deconv層結構的list
        deconv_layers = []
        # 遍歷總共需要多少層deconv層結構
        for i in range(num_deconv_layers):
            if cat_output[i]:
                # 如果該層需要concat就會使得輸入channel加深，所以這裡需要更新輸入channel深度
                in_channels += deconv_layer_output_channels[i]

            # 獲取輸出的channel深度
            planes = num_deconv_filters[i]
            # 獲取deconv的設定資料
            deconv_kernel, padding, output_padding = self._get_deconv_cfg(num_deconv_kernels[i])

            # 存放層結構的list
            layers = list()
            layers.append(
                # 透過Sequential將多層結構封裝
                nn.Sequential(
                    # 構建上採樣層結構
                    build_upsample_layer(
                        # 這裡使用的是deconv進行上採樣，這裡會是官方實現的轉置卷積
                        dict(type='deconv'),
                        # 將channel調整到指定的深度
                        in_channels=in_channels,
                        out_channels=planes,
                        kernel_size=deconv_kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False),
                    # 構建標準化層
                    nn.BatchNorm2d(planes, momentum=0.1),
                    nn.ReLU(inplace=True)))
            # 構建剩下需要堆疊的basic_blocks
            for _ in range(num_basic_blocks):
                # 構建BasicBlock並且用Sequential保存，這裡使用的是resnet當中的BasicBlock
                layers.append(nn.Sequential(BasicBlock(planes, planes), ))
            deconv_layers.append(nn.Sequential(*layers))
            # 更新輸入的channel深度
            in_channels = planes

        # 最後透過ModuleList包裝
        return nn.ModuleList(deconv_layers)

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        # 根據給定的卷積核大小，返回需要的padding值
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            # 如果都不在這四個當中就會報錯
            raise ValueError(f'Not supported num_kernels ({deconv_kernel}).')

        # 將需要的padding值進行回傳
        return deconv_kernel, padding, output_padding

    def get_loss(self, outputs, targets, masks, joints):
        """Calculate bottom-up keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - num_outputs: O
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            outputs (list(torch.Tensor[N,K,H,W])): Multi-scale output heatmaps.
            targets (List(torch.Tensor[N,K,H,W])): Multi-scale target heatmaps.
            masks (List(torch.Tensor[N,H,W])): Masks of multi-scale target
                heatmaps
            joints (List(torch.Tensor[N,M,K,2])): Joints of multi-scale target
                heatmaps for ae loss
        """
        """ 計算bottom-up的關節點損失
        Args:
            outputs: 網路預測的結果，list[tensor]，tensor shape [batch_size, channel, height, width]
            targets: 標註的熱力圖，list[tensor]，tensor shape [batch_size, num_joints, height, width]
            masks: 標註哪些地方不需要計算loss，list[tensor]，tensor shape [batch_size, height, width]
            joints: 標註關節點座標，list[tensor]，tensor shape [batch_size, max_people, num_joints, 2]
        """

        # 構建losses字典
        losses = dict()

        # 計算損失
        heatmaps_losses, push_losses, pull_losses = self.loss(outputs, targets, masks, joints)

        # 將batch的損失進行合併
        for idx in range(len(targets)):
            if heatmaps_losses[idx] is not None:
                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                if 'heatmap_loss' not in losses:
                    losses['heatmap_loss'] = heatmaps_loss
                else:
                    losses['heatmap_loss'] += heatmaps_loss
            if push_losses[idx] is not None:
                push_loss = push_losses[idx].mean(dim=0)
                if 'push_loss' not in losses:
                    losses['push_loss'] = push_loss
                else:
                    losses['push_loss'] += push_loss
            if pull_losses[idx] is not None:
                pull_loss = pull_losses[idx].mean(dim=0)
                if 'pull_loss' not in losses:
                    losses['pull_loss'] = pull_loss
                else:
                    losses['pull_loss'] += pull_loss

        # 回傳損失字典
        return losses

    def forward(self, x):
        """Forward function."""
        # 解碼頭forward部分，x shape [batch_size, channel, height, width]
        if isinstance(x, list):
            # 如果x是list型態，就需要將其提取出來
            x = x[0]

        # 最終輸出的保存list
        final_outputs = []
        # 將傳入的特徵圖通過第一層final_layers層結構，將channel調整
        y = self.final_layers[0](x)
        # 將y保存到final_outputs當中
        final_outputs.append(y)

        # 遍歷deconv的層數
        for i in range(self.num_deconvs):
            if self.cat_output[i]:
                # 如果該層需要進行concat就會進行concat，會在channel維度上進行拼接
                x = torch.cat((x, y), 1)

            # 透過轉置卷積進行2被上採樣，同時調整channel深度
            x = self.deconv_layers[i](x)
            # 透過final_layers將x進行通道調整
            y = self.final_layers[i + 1](x)
            # 將結果保存到final_outputs當中
            final_outputs.append(y)

        # 最後將結果回傳，list[tensor]，tensor shape [batch_size, channel, height, width]，list長度會是2
        # channel深度會是[34, 17]
        return final_outputs

    def init_weights(self):
        """Initialize model weights."""
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for _, m in self.final_layers.named_modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
