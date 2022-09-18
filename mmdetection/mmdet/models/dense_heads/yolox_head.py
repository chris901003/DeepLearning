# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class YOLOXHead(BaseDenseHead, BBoxTestMixin):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 strides=[8, 16, 32],
                 use_depthwise=False,
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='IoULoss',
                     mode='square',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=5.0),
                 loss_obj=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):
        """ 構建YOLOX的分類頭初始化函數
        Args:
            num_classes: 分類類別數
            in_channels: 輸入channel深度
            feat_channels: 特徵圖的channel深度
            stacked_convs: 堆疊的卷積層數量
            strides: 每個特徵圖對於原圖的下採樣倍數
            use_depthwise: 是否使用dw卷積
            dcn_on_last_conv: 在最後一個卷積是否使用dcn卷積
            conv_bias: 在卷積中是否使用偏置
            conv_cfg: 卷積層設定
            norm_cfg: 標準化層設定
            act_cfg: 激活函數設定
            loss_cls: 分類類別損失設定
            loss_bbox: 標註匡損失設定
            loss_obj: 是否有物體損失設定
            loss_l1: l1損失設定
            train_cfg: train當中的特別參數
            test_cfg: test當中的特別參數
            init_cfg: 初始化方式
        """

        # 繼承自BaseDenseHead, BBoxTestMixin，對繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)
        # 將傳入資料進行保存
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        # 檢查指定的conv_bias需要是auto或是bool參數
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        # 預先將use_sigmoid_cls設定成True
        self.use_sigmoid_cls = True

        # 保存傳入資料
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # 構建類別分類損失
        self.loss_cls = build_loss(loss_cls)
        # 構建預測匡損失
        self.loss_bbox = build_loss(loss_bbox)
        # 構建預測是否有物體損失
        self.loss_obj = build_loss(loss_obj)

        # 預先將use_l1設定成False
        self.use_l1 = False  # This flag will be modified by hooks.
        # 構建l1損失計算
        self.loss_l1 = build_loss(loss_l1)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        # 保存訓練以及測試的專用參數
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        # 先將sampling設定成False
        self.sampling = False
        if self.train_cfg:
            # 如果train_cfg不是None就會到這裡
            # 構建正負樣本匹配實例化對象
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            # 構建sampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.fp16_enabled = False
        # 構建最終分類卷積層結構
        self._init_layers()

    def _init_layers(self):
        # 構建多層分類卷積層
        self.multi_level_cls_convs = nn.ModuleList()
        # 構建多層回歸預測卷積層
        self.multi_level_reg_convs = nn.ModuleList()
        # 構建多層分類類別卷積層
        self.multi_level_conv_cls = nn.ModuleList()
        # 構建多層回歸預測卷積層
        self.multi_level_conv_reg = nn.ModuleList()
        # 構建多層預測是否為正樣本卷積層
        self.multi_level_conv_obj = nn.ModuleList()
        # 遍歷總共有多少個特徵圖輸出
        for _ in self.strides:
            # 添加分類頭的卷積
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            # 添加回歸頭的卷積
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            # 構建最終分類卷積以及回歸卷積以及預測正負樣本卷積
            conv_cls, conv_reg, conv_obj = self._build_predictor()
            # 添加到對應位置
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)

    def _build_stacked_convs(self):
        """Initialize conv layers of a single level head."""
        # 構建堆疊卷積層
        # 這裡會根據是否使用dw卷積獲取卷積類
        conv = DepthwiseSeparableConvModule if self.use_depthwise else ConvModule
        # 多層卷積保存空間
        stacked_convs = []
        # 遍歷指定堆疊層數
        for i in range(self.stacked_convs):
            # 獲取輸入channel
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                # 如果有指定在最後一層要使用dcn卷積就會將卷積設定成dcn
                conv_cfg = dict(type='DCNv2')
            else:
                # 否則就會使用傳入的卷積方式
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                # 使用指定的卷積類
                conv(
                    # 調整channel到feat_channels
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        # 用Sequntial將多層卷積打包
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        # 構建最終分類頭
        # 構建類別分類卷積，將channel變成分類類別數
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        # 構建回歸卷積，將channel變成4，分別是(中心點x偏移, 中心點y偏移, 高, 寬)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        # 構建判斷是否為正樣本，將channel變成1，只分類成是否為正樣本
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        # 回傳實例化對象
        return conv_cls, conv_reg, conv_obj

    def init_weights(self):
        super(YOLOXHead, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls,
                                      self.multi_level_conv_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg,
                       conv_obj):
        """Forward feature of a single scale level."""
        # 進行一次forward

        # 進行分類方面的兩次卷積
        cls_feat = cls_convs(x)
        # 進行回歸方面的兩次卷積
        reg_feat = reg_convs(x)

        # 透過分類類別卷積將channel深度調整到分類類別數
        cls_score = conv_cls(cls_feat)
        # 透過回歸卷積將channel深度調整到4
        bbox_pred = conv_reg(reg_feat)
        # 通過判斷是否為正樣本將channel深度調整到1
        objectness = conv_obj(reg_feat)

        # 回傳卷積後結果
        # cls_scored shape = [batch_size, num_classes, height, width]
        # bbox_pred shape = [batch_size, 4, height, width]
        # objectness shape = [batch_size, 1]
        return cls_score, bbox_pred, objectness

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        # 進行分類頭向前傳遞，這裡將所需資料打包傳入到multi_apply當中
        return multi_apply(self.forward_single, feats,
                           self.multi_level_cls_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_conv_cls,
                           self.multi_level_conv_reg,
                           self.multi_level_conv_obj)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   objectnesses,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        """ 獲取真正標註的匡
        Args:
            cls_scores: 分類類別置信度分數
            bbox_preds: 預測匡
            objectnesses: 正樣本預測概率
            img_metas: 圖像詳細資訊
            cfg: test的預處理設定
            rescale: 是否需要重新縮放
            with_nms: 是否使用nms
        """
        # 檢查batch_size是否相同
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        # 如果有特別設定cfg就會更新使用的cfg否則就使用原始的cfg
        cfg = self.test_cfg if cfg is None else cfg
        # 獲取圖像經過多少倍的縮放
        scale_factors = np.array(
            [img_meta['scale_factor'] for img_meta in img_metas])

        # 獲取batch_size大小
        num_imgs = len(img_metas)
        # 獲取特徵圖大小
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        # 構建priors網格，這裡會將步距填上
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        # 將類別置信度分數攤平 [bs, height, width, num_cls] -> [bs, height * width, num_cls]
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        # 將回歸匡預測攤平 [bs, height, width, 4] -> [bs, height * width, 4]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # 將正樣本預測攤平 [bs, height, width, 1]  -> [bs, height * width, 1]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        # 將不同尺度的特徵圖資料融合
        # 對於類別以及正樣本需要通過sigmoid [bs, total_pixel, channel]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        flatten_priors = torch.cat(mlvl_priors)

        # 將預測匡縮放到原圖上
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        if rescale:
            # 將座標位置進行縮放
            flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(scale_factors).unsqueeze(1)

        # 結果的列表
        result_list = []
        # 遍歷所有圖像
        for img_id in range(len(img_metas)):
            # 將該圖像的所有資料提取出來
            cls_scores = flatten_cls_scores[img_id]
            score_factor = flatten_objectness[img_id]
            bboxes = flatten_bboxes[img_id]

            # 將通過nms後的結果放到result_list當中
            result_list.append(self._bboxes_nms(cls_scores, bboxes, score_factor, cfg))

        return result_list

    def _bbox_decode(self, priors, bbox_preds):
        """ 對於預測匡進行解碼
        Args:
            priors: 不同尺度特徵圖的座標點對應到原圖的座標點，tensor shape [sum(height * width), 4]
            bbox_preds: 預測回歸匡資料，tensor shape [bs, sum(height * width), 4]
        """
        # 對於每個座標點先將偏移量放大回原圖後再加上原圖原始座標，這樣就直接會是在原圖上的預測中心點座標
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        # 先對預測的高寬以e為底數預測值為指數進行運算，最後乘上放大倍率
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        # 獲取[xmin, ymin, xmax, ymax]資訊
        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        # 將結果在最後維度進行堆疊
        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        # decoded_bboxes shape [bs, sum(height * width), 4]，最後的4就會是[xmin, ymin, xmax, ymax]
        return decoded_bboxes

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg):
        """ 進行nms非極大值抑制
        Args:
            cls_scores: 分類類別置信度，tensor [sum(height * width), num_cls]
            bboxes: 回歸匡，tensor shape [sum(height * width), 4]
            score_factor: 置信度倍率參數，這裡會是預測是否為正樣本的概率，tensor shape [sum(height * width)]
            cfg: nms的相關參數
        """
        # 對於每個像素點獲取最大置信度的類別
        max_scores, labels = torch.max(cls_scores, 1)
        # 獲取哪些匡是合法的，這裡會將類別置信度乘上正樣本置信度，如果超過閾值表示該點預測的為正樣本
        valid_mask = score_factor * max_scores >= cfg.score_thr

        # 將非合法匡的部分過濾掉
        # 回歸匡
        bboxes = bboxes[valid_mask]
        # 置信度分數
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        # 預測類別
        labels = labels[valid_mask]

        if labels.numel() == 0:
            # 如果沒有標註訊息就直接回傳
            return bboxes, labels
        else:
            # 進行nms非極大值抑制
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            # 將結果回傳
            return dets, labels[keep]

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        """ yolox的分類頭損失計算
        Args:
            cls_scores: 預測結果，每個座標點的分類類別概率，list[tensor]，tensor shape [batch_size, num_classes, height, width]
            bbox_preds: 回歸匡預測，每個座標點的回歸預測，list[tensor]，tensor shape [batch_size, 4, height, width]
            objectnesses: 預測每個座標點是否為正樣本，list[tensor]，tensor shape [batch_size, 1, height, width]
            gt_bboxes: 標註匡資料，list[tensor]，tensor shape [num_object, 4]，list長度會是batch_size
            gt_labels: 標註匡對應上的類別，list[tensor]，tensor shape [num_object]，list長度會是batch_size
            img_metas: 一個batch圖像的詳細資訊
            gt_bboxes_ignore: 忽略掉的標註匡資料
        """
        # 獲取一個batch當中有多少張圖像
        num_imgs = len(img_metas)
        # 獲取不同尺度的特徵圖的高寬資料
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        # 為了多種尺度的特徵圖需要先處理
        # mlvl_priors，list[tensor]，list長度就會是不同尺度的特徵圖，tensor shape [feat_height * feat_height, 2或是4]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        # 將cls_pred進行通道處理，這裡list長度就會是特徵圖的尺度數量
        # [bs, num_cls, height, width] -> [bs, height, width, num_cls] -> [bs, height * width, num_cls]
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        # 將bbox_pred進行通道處理，這裡的list長度就會是特徵圖的尺度數量
        # [bs, 4, height, width] -> [bs, height, width, 4] -> [bs, height * width, 4]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # 將objectness進行處理，這裡的list長度就會是特徵圖的尺度數量
        # [bs, 1, height, width] -> [bs, height, width, 1] -> [bs, height * width, 1]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        # 將list部分進行拼接，這裡會在第二個維度進行拼接
        # [bs, sum(height * width), channel]
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        # 將剛才構建的網格進行concat，shape [sum(height * width), 4]
        flatten_priors = torch.cat(mlvl_priors)
        # 對於預測回歸匡進行解碼
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        # pos_masks = 標註那些預測匡是正樣本，正樣本部分會是True，tensor shape [num_pred]
        # cls_targets = 正樣本預測匡所需預測出的類別置信度分數，tensor shape [num_pos, num_cls]
        # obj_targets = 每個預測匡應預測出是否為正樣本的概率，tensor shape [num_pred, 1]
        # bbox_targets = 正樣本預測匡應預測出的匡選區域，tensor shape [num_pos, 4]
        # l1_targets = 正樣本的l1值
        # num_fg_imgs = 每張圖像有多少正樣本，list[int]，list長度會是batch_size
        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels)

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        # 一個batch當中圖像總共有多少正樣本，這裡會是tensor格式
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        # 多gpu時才會有作用，否則就會是num_pos
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        # 將一個batch的資料在第一個維度上進行concat
        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            # 如果有使用l1損失就會到這裡
            l1_targets = torch.cat(l1_targets, 0)

        # 計算回歸匡損失，這裡會提取出正樣本的部分計算損失
        loss_bbox = self.loss_bbox(flatten_bboxes.view(-1, 4)[pos_masks], bbox_targets) / num_total_samples
        # 計算正樣本判斷損失，這裡所有預測匡都會進行計算損失
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1), obj_targets) / num_total_samples
        # 計算分類損失，這裡只會計算正樣本的分類損失
        loss_cls = self.loss_cls(flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
                                 cls_targets) / num_total_samples

        # 最後將損失包裝成字典
        loss_dict = dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj)

        if self.use_l1:
            # 如果有使用l1損失就會進行計算後添加到損失字典當中
            loss_l1 = self.loss_l1(flatten_bbox_preds.view(-1, 4)[pos_masks], l1_targets) / num_total_samples
            loss_dict.update(loss_l1=loss_l1)

        # 回傳損失字典
        return loss_dict

    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """
        """ 計算一張圖像的結果
        Args:
            cls_preds: 分類類別的預測，tensor shape [sum(height, width), num_cls]
            objectness: 是否為正樣本，tensor shape [sum(height, width), 1]
            priors: 特徵圖映射回原圖的縮放倍率以及對應座標，tensor shape [sum(height, width), 4]
            decoded_bboxes: 通過映射回原圖後的預測匡，tensor shape [sum(height, width), 4]
            gt_bboxes: 標註匡，tensor shape [num_object, 4]
            gt_labels: 標註匡對應類別，tensor shape [num_object]
        """

        # 獲取sum(height, width)資訊，也就是總共有多少預測匡
        num_priors = priors.size(0)
        # 獲取標註匡有多少個
        num_gts = gt_labels.size(0)
        # 將gt_bboxes轉成與decoded_bboxes相同型態
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        # No target
        if num_gts == 0:
            # 如果沒有任何標註匡就會到這裡，構建一堆0資料
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            return foreground_mask, cls_target, obj_target, bbox_target, l1_target, 0

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        # 這裡會構建priors的偏移量[網格點中心點位置]，offset_priors shape [sum(height * width), 4]
        offset_priors = torch.cat([priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        # 透過assigner的assign匹配正負樣本，這裡會將類別預測以及正樣本概率通過sigmoid後相成獲取最後的置信度，也就是進行SimOTA
        # 這裡會將結果打包成AssignResult的實例化對象
        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

        # 進行正樣本匹配資料處理，回傳SamplingResult實例對象
        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        # 獲取哪些預測匡是正樣本
        pos_inds = sampling_result.pos_inds
        # 獲取總共有多少正樣本
        num_pos_per_img = pos_inds.size(0)

        # 獲取正樣本的iou值
        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        # 獲取要計算分類損失時需要的one-hot格式的標註訊息，shape [num_pos, num_cls]，這裡的值還會再乘上iou
        cls_target = F.one_hot(sampling_result.pos_gt_labels, self.num_classes) * pos_ious.unsqueeze(-1)
        # 構建判斷是否為正樣本的tensor，shape [sum(height * width), 1]，這裡初始化為0
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        # 將正樣本的部分設定成1其他保持為0
        obj_target[pos_inds] = 1
        # 獲取正樣本應對應到的真實匡的[xmin, ymin, xmax, ymax]資訊
        bbox_target = sampling_result.pos_gt_bboxes
        # 獲取l1應該要的值，這裡會是shape [num_pos, 4]
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            # 如果需要計算l1損失就會到這裡
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        # 構建一個tensor shape [sum(height * width)]且初始化為0
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        # 將正樣本的部分設定成1其他保持0
        foreground_mask[pos_inds] = 1
        # 回傳參數
        return foreground_mask, cls_target, obj_target, bbox_target, l1_target, num_pos_per_img

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target
