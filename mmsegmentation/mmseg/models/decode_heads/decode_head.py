# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, auto_fp16, force_fp32

from mmseg.core import build_pixel_sampler
from mmseg.ops import resize
from ..builder import build_loss
from ..losses import accuracy


class BaseDecodeHead(BaseModule, metaclass=ABCMeta):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255.
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 init_cfg=dict(
                     type='Normal', std=0.01, override=dict(name='conv_seg'))):
        """
        :param in_channels: 輸入的channel
        :param channels: 輸出的channel，這裡還不會是最後分類類別數的數量
        :param num_classes: 最後輸出的channel，也就是最後的分類類別數
        :param dropout_ratio: dropout的概率值
        :param conv_cfg: 卷積設定資料
        :param norm_cfg: 標準化層結構的設定
        :param act_cfg: 激活函數的設定
        :param in_index: 輸入特徵圖的index(?)
        :param input_transform: 如果有要將多層的特徵層一起放到解碼頭就會設定融合的方式
        :param loss_decode: 損失計算的設定
        :param ignore_index: 在分割時哪個值在計算損失時不會納入，就是物體邊緣交界處
        :param sampler:
        :param align_corners:
        :param init_cfg: 初始化config
        """
        # 已看過

        # 將init_cfg傳到繼承當中，BaseModule為MMCV當中最基底的Module
        super(BaseDecodeHead, self).__init__(init_cfg)
        # _init_inputs = 檢查in_channels與in_index與input_transform之間有沒有錯誤，同時會將值進行保存
        self._init_inputs(in_channels, in_index, input_transform)
        # 保存其他的變數
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        # 構建損失計算方式
        if isinstance(loss_decode, dict):
            # 如果是dict就直接構建
            # self.loss_decode = 損失函數的實例對象
            self.loss_decode = build_loss(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            # 如果是list就遍歷list進行構建
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(loss))
        else:
            # 如果loss_decode有其他型態這裡就直接報錯
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        # sampler相關的設定，這裡我們先跳過
        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

        # 最後會再通過self.conv_seg將channel深度調整到最終深度，也就是channel=num_classes，這裡會用卷積核大小為1的做channel調整
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            # 如果有設定dropout_ratio就會有Dropout層
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            # 否則就會是None
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """
        # 已看過
        # 該函數主要是在確認[in_channels,l in_index, input_transform]是否正確同時保存值

        # 如果有要將多層的特徵層一起放到解碼頭就會設定融合的方式
        if input_transform is not None:
            # 這裡只提供兩種方式，如果使用其他種就會在這裡報錯
            assert input_transform in ['resize_concat', 'multiple_select']
        # 將融合方式保存下來
        self.input_transform = input_transform
        # 輸入到解碼頭的特徵層的index
        self.in_index = in_index
        if input_transform is not None:
            # 如果是要融合多個特徵層的話，in_channels以及in_index都是要list或是tuple型態
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            # 同時in_channels與in_index是一組的，所以長度比需要一樣
            assert len(in_channels) == len(in_index)
            # 這裡會有兩種融合模式，一種是透過拼接(需要將高寬調整成一樣)，另一種是透過multiple_select(不知道是什麼)
            if input_transform == 'resize_concat':
                # 透過concat的最後通道數就會是所有特徵層channel的總和
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            # 如果只有將一個特徵層放到預測頭那麼in_channels與in_index都必須要是int格式
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            # 保留in_channels
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        # 已看過
        # inputs = 多層特徵層，有不同的維度以及尺度的特徵圖

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            # 如果都沒有設定任何特徵圖融合方式，就會使用指定的特徵圖的層數
            inputs = inputs[self.in_index]

        return inputs

    @auto_fp16()
    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        """
        :param inputs: list類型，多層特徵層，有不大小以及深度的特徵層 
        :param img_metas: list類型，圖像的詳細資料
        :param gt_semantic_seg: tensor類型，堆疊了標註好的圖像的tensor 
        :param train_cfg: 訓練的一些配置，這裡預設會是空
        :return: 
        """
        # 已看過
        # 使用forward函數進行向前傳播獲取最後的輸出結果
        # seg_logits shape = [batch_size, channel=num_classes, height, width]
        seg_logits = self.forward(inputs)
        # 將最終輸出與標註圖進行損失計算
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

    def cls_seg(self, feat):
        """Classify each pixel."""
        # 已看過
        # 最終輸出將channel調整到與num_classes相同
        if self.dropout is not None:
            # 通過dropout層
            feat = self.dropout(feat)
        # output shape = [batch_size, channel=num_classes, height, width]
        output = self.conv_seg(feat)
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        # 已看過，主要是用來計算segmentation損失使用的
        # seg_logit = 透過模型計算出來的結果
        # seg_label = 人工標記的結果

        # 構建一個損失的字典
        loss = dict()
        # 透過resize將seg_logit的高寬調整到與標註的圖像大小相同
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logit, seg_label)
        else:
            seg_weight = None
        # 將標註的tensor從第一個維度展平
        # seg_label shape = [batch_size, height * width]
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            # 如果當前的損失計算不是用list包裝起來，就在最外層加上list，主要是有時候需要多種損失計算，這樣才可以通用
            losses_decode = [self.loss_decode]
        else:
            # 如果已經是用list包裝起來就直接拿過去用就可以
            losses_decode = self.loss_decode
        # 遍歷所有的損失計算方式
        for loss_decode in losses_decode:
            # 兩個操作都一樣只是開新的與相加的差別而已
            if loss_decode.loss_name not in loss:
                # 如果該損失計算沒有在loss字典當中就需要在字典當中添加
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                # 如果該損失計算已經有在loss字典當中就直接相加
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logit,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        # 透過accuracy進行計算正確率
        loss['acc_seg'] = accuracy(
            seg_logit, seg_label, ignore_index=self.ignore_index)
        return loss
