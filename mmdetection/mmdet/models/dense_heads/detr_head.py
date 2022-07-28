# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding
from mmcv.runner import force_fp32

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead


@HEADS.register_module()
class DETRHead(AnchorFreeHead):
    """Implements the DETR transformer head.

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        """ 已看過，這裡是DETR的預測匡頭
        Args:
            num_classes: 分類類別數
            in_channels: 輸入的channel深度
            num_query: 總共會產生出多少個預測匡
            num_reg_fcs: 在FFN當中會用到多少的全連接層
            transformer: transformer的設定
            sync_cls_avg_factor: 多gpu會用到的東西
            positional_encoding: 位置編碼
            loss_cls: 分類類別的損失設定
            loss_bbox: 預測匡的損失設定
            loss_iou: iou的損失設定
            train_cfg: 訓練用的config檔案
            test_cfg: 測試時用的config檔案
            init_cfg: 初始化的方法
        """
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        # 這裡繼承自AnchorFreeHead，將初始化設定傳入
        super(AnchorFreeHead, self).__init__(init_cfg)
        # 將背景的分類權重設定成0
        self.bg_cls_weight = 0
        # 如果是單gpu，sync_cls_avg_factor會是False
        self.sync_cls_avg_factor = sync_cls_avg_factor
        # 從loss_cls當中找到類別的損失權重，如果沒有在loss_cls當中就會先是None
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DETRHead):
            # 傳入的class_weight需要是float型態
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            # 從loss_cls當中獲取背景類別的權重，如果沒有特別設定就會跟普通的class_weight相同
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            # 檢查bg_cls_weight需要是float型態
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            # 構建一個全為1且長度是類別數加一，之後再乘上類別的損失權重
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            # 將最後一個地方的權重設定成背景的權重
            class_weight[num_classes] = bg_cls_weight
            # 將class_weight放到loss_cls當中保存
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                # 將bg_cls_weight從loss_cls當中去除
                loss_cls.pop('bg_cls_weight')
            # 保存bg_cls_weight
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            # 有設定train_cfg就會到這裡
            # 當我們有設定train_cfg當中就一定要有assigner參數，沒有的話就回直接報錯
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            # 將assigner提取出來，裡面就會有預測匡與真實匡的配對方式(在DETR中就會是匈牙利匹配)
            assigner = train_cfg['assigner']
            # 在計算分類損失時的權重要與計算cost時的權重相同，如果不相同這裡就直接報錯
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            # 在計算預測匡損失的權重要與計算cost時的權重要相同，否則這裡就會直接報錯
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            # 在計算iou損失的權重要與計算cost時的權重相同，否則在這裡就會直接報錯
            assert loss_iou['loss_weight'] == assigner['iou_cost']['weight'], \
                'The regression iou weight for loss and matcher should be' \
                'exactly the same.'
            # 構建指派的實例化對象(在DETR當中就會是匈牙利匹配)
            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            # 實例化sampler，目前不知道作用
            self.sampler = build_sampler(sampler_cfg, context=self)
        # 進行一系列保存值
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        # 構建loss實例化對象
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            # 如果使用的是二分類，這裡輸出的分類數就會直接是num_classes
            self.cls_out_channels = num_classes
        else:
            # 其他的就會有一個背景，所以會是num_classes加上1
            self.cls_out_channels = num_classes + 1
        # 從transformer獲取激活函數的類型，如果沒有就會使用ReLU作為激活函數
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        # 構建激活函數實例對象
        self.activate = build_activation_layer(self.act_cfg)
        # 構建位置編碼實例對象
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        # 構建生成預測匡的transformer，裏面包含了encoder與decoder
        self.transformer = build_transformer(transformer)
        # 獲取一個特徵點會用多少維度的向量進行表示
        self.embed_dims = self.transformer.embed_dims
        # positional_encoding當中需要有num_feats參數
        assert 'num_feats' in positional_encoding
        # 獲取位置編碼的channel深度
        num_feats = positional_encoding['num_feats']
        # 檢查channel是否合法
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        # 構建對初始輸入到transformer head的數據進行調整
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        # 已看過，對原始輸入到transformer head的數據進行調整

        # 將輸入的資料透過conv調整通道深度，調整到embed_dims深度
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)
        # 透過全連接層將channel深度從embed_dims調整到分類類別數
        self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        # 構建FFN全連接層
        self.reg_ffn = FFN(
            self.embed_dims,
            self.embed_dims,
            self.num_reg_fcs,
            self.act_cfg,
            dropout=0.0,
            add_residual=False)
        # 構建預測匡的全連接層，將channel深度從embed_dims調整到4
        self.fc_reg = Linear(self.embed_dims, 4)
        # query的位置編碼，這裡用的就是可學習的
        self.query_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is DETRHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, feats, img_metas):
        """Forward function.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        # 已看過，DETRHead的forward函數
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single, feats, img_metas_list)

    def forward_single(self, x, img_metas):
        """"Forward function for a single feature level.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        # 已看過，DETR的解碼頭

        # 獲取batch_size
        batch_size = x.size(0)
        # 獲取最一開始輸入到網路當中時圖像的高寬
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        # 構建一個masks且shape [batch_size, input_img_height, input_img_width]且全為1
        masks = x.new_ones((batch_size, input_img_h, input_img_w))
        # 遍歷一個batch的圖像數量
        for img_id in range(batch_size):
            # 獲取圖像經過一系列處理後的圖像大小，與上面的input_img會有不同，因為上面的為了變成一個batch需要相同高寬
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            # 將該地方的masks變成0
            masks[img_id, :img_h, :img_w] = 0

        # 通過一個卷積層將channel調整到embed_dim深度
        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        # 透過差值方式將mask的大小調整到與x相同
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        # position encoding，進行位置編碼，shape [batch_size, embed_dim, height, width]
        pos_embed = self.positional_encoding(masks)  # [bs, embed_dim, h, w]
        # outs_dec: [nb_dec, bs, num_query, embed_dim]
        # 將資料輸入到transformer當中進行encoder與decoder操作
        # outs_dec shape = [num_layers, batch_size, num_queries, channel=embed_dim]
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)

        # all_cls_scores = [num_layers, batch_size, num_queries, channel=num_classes]
        all_cls_scores = self.fc_cls(outs_dec)
        # all_bbox_preds = [num_layers, batch_size, num_queries, channel=4]
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores_list,
             all_bbox_preds_list,
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # NOTE defaultly only the outputs from the last feature scale is used.
        # 已看過，計算損失的地方
        # all_cls_scores_list = 預測匡對應上的預測類別
        #       list[tensor] tensor shape [num_layers, batch_size, num_queries, channel=num_classes]
        # all_bbox_preds_list = 預測匡
        #       list[tensor] tensor shape [num_layers, batch_size, num_queries, channel=4]
        # gt_bboxes_list = 一個圖像的標註匡
        #       list[tensor] tensor shape [num_bboxes, 4]，list長度就會是batch_size
        # gt_labels_list = 一個圖像標註匡對應上的類別
        #       list[tensor] tensor shape [num_bboxes]，list長度就會是batch_size
        # img_metas = 圖像的相關資料

        # 將資料取出來
        all_cls_scores = all_cls_scores_list[-1]
        all_bbox_preds = all_bbox_preds_list[-1]
        assert gt_bboxes_ignore is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        # 獲取decoder的輸出層數量
        num_dec_layers = len(all_cls_scores)
        # 將標註訊息複製num_layers次
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        # 多複製幾分img_metas
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        # 計算loss，回傳的都會是list[tensor]，list長度就會是decoder的輸出層數，每一個輸出都會計算出loss值
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss from the last decoder layer，最後輸出的loss會在最後面
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]
        # loss from other decoder layers，其他輔助訓練的會在前面
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # 已看過，計算單層輸出的loss值

        # 獲取batch_size
        num_imgs = cls_scores.size(0)
        # 將第一個維度的部分變成list，也就是list的長度會是batch_size
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        # 這裡的變數代表的意思可以到下面get_targets函數查看
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # 將labels_list的內容在第一個維度拼接，shape [num_pred * batch_size]
        labels = torch.cat(labels_list, 0)
        # 將label_weights的內容在第一個維度拼接，shape [num_pred * batch_size]
        label_weights = torch.cat(label_weights_list, 0)
        # 將bbox_targets的內容在第一個維度拼接，shape [num_pred * batch_size, 4]
        bbox_targets = torch.cat(bbox_targets_list, 0)
        # 將bbox_weights的內容在第一個維度拼接，shape [num_pred * batch_size, 4]
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss，計算分類損失
        # cls_scores shape [batch_size, num_pred, cls_out_channels] -> [batch_size * num_pred, cls_out_channels]
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        # 這裡會有一個權重
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        # 權重至少會是1
        cls_avg_factor = max(cls_avg_factor, 1)

        # 計算分類類別損失，這裡使用的是交叉熵
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        # 獲取正樣本數量
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        # factors是用來將相對位置的預測匡變成絕對位置的
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            # 獲取圖像原始大小
            img_h, img_w, _ = img_meta['img_shape']
            # factor shape [num_pred, 4]
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        # factors shape [num_pred * batch_size, 4]
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        # bbox_preds shape [batch_size, num_pred, 4] -> [batch_size * num_pred, 4]
        bbox_preds = bbox_preds.reshape(-1, 4)
        # 將[center_x, center_y, w, h]轉成[xmin, ymin, xmax, ymax]並且變成絕對位置
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        # 計算iou損失
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        # 計算l1損失
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        # 回傳類別以及l1損失以及iou
        return loss_cls, loss_bbox, loss_iou

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        # 已看過
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        # labels_list = list[tensor] tensor shape [num_pred]，list長度就會是batch_size，裡面存的就會是該預測匡應該要預測的類別
        # label_weights_list = list[tensor]，該類別佔的比重shape與labels_list相同
        # bbox_targets_list = list[tensor] tensor shape [num_pred, 4]，list長度是batch_size，裡面存的就會是該預測匡應該要的數值
        # bbox_weight_list = list[tensor]，該預測匡佔的比重，shape與bbox_targets_list相同，如果是背景就會是0
        # pos_inds_list = list[tensor] tensor shape [num_gt]，list長度是batch_size，紀錄哪些index的預測匡是正樣本
        # neg_inds_list = list[tensor]，紀錄哪些index的預測匡是負樣本，shape與pos_inds_list相同
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_bboxes_list, gt_labels_list, img_metas, gt_bboxes_ignore_list)
        # 計算總共的正樣本
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        # 計算總共的負樣本
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        # 將結果回傳
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_bboxes,
                           gt_labels,
                           img_meta,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            img_meta (dict): Meta information for one image.
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # 已看過
        # 這裡就會是對於一張照片的預測內容

        # 獲取總共有多少個預測匡
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler，進行預測匡與真實匡配對，回傳會是AssignResult實例對象
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, img_meta,
                                             gt_bboxes_ignore)
        # 將剛才的回傳送進去，會傳的是一個SamplingResult的實例化對象
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # 獲取正樣本在預測匡中的index
        pos_inds = sampling_result.pos_inds
        # 獲取副樣本在預測匡中的index
        neg_inds = sampling_result.neg_inds

        # label targets
        # 構建labels且shape[num_preds]預先全部設定成num_classes表示沒有目標
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        # 將有對應上gt的預測匡的index位置變成應該要是的分類類別的index
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # 設定label的權重，shape [num_preds]且全為1
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        # 構建bbox_target且shape [num_pred, 4]且全為0
        bbox_targets = torch.zeros_like(bbox_pred)
        # 構建bbox_weights且shape [num_pred, 4]且全為0
        bbox_weights = torch.zeros_like(bbox_pred)
        # 將有對應到gt的預測匡的index部分權重改成1
        bbox_weights[pos_inds] = 1.0
        # 獲取圖像的高寬
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        # 構建factor縮放比例，shape [1, 4]
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        # 將gt標註匡從絕對座標改成相對座標
        pos_gt_bboxes_normalized = sampling_result.pos_gt_bboxes / factor
        # 將[xmin, ymin, xmax, ymax]轉成[center_x, center_y, w, h]格式
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        # 保存到bbox_targets當中
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        # 將資料回傳
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # 已看過，將backbone輸出的資料進行預測
        # x = 從backbone出來的特徵圖，tuple[tensor]，tensor shape [batch_size, channel, height, width]
        # img_metas = 圖像的相關資訊
        # gt_bboxes = 圖像標註的標註匡，list[tensor]，list長度會是batch_size，tensor shape [num_gt_box, 4]
        # gt_labels = 圖像標註的標註匡的類別，list[tensor]，list長度會是batch_size，tensor shape [num_gt_box]
        # gt_bboxes_ignore = 因某些不合法的關係被過濾掉的標註匡
        # proposal_cfg = proposal_cfg用的

        # proposal_cfg必須要是None
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        # 進行向前傳遞
        # out = tuple[list[tensor]]，out[0]=預測類別且tensor shape[num_layers, batch_size, num_queries, num_classes]
        # out[1]=預測匡資訊且tensor shape[num_layers, batch_size, num_queries, 4]
        outs = self(x, img_metas)
        if gt_labels is None:
            # 將標註匡加進去，如果沒有標註匡類別
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            # 如果有給該圖像正確的標註匡類別以及標註匡就加進去
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        # loss就是計算出來的損失值，losses=dict，就會是key對上value，value就會是loss值
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        # 將損失內容進行回傳
        return losses

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores_list,
                   all_bbox_preds_list,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        # NOTE defaultly only using outputs from the last feature level,
        # and only the outputs from the last decoder layer is used.
        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)

        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_query, 4].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 5], \
                    where the first 4 columns are bounding box positions \
                    (tl_x, tl_y, br_x, br_y) and the 5-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            det_bboxes /= det_bboxes.new_tensor(scale_factor)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
        return results_list

    def forward_onnx(self, feats, img_metas):
        """Forward function for exporting to ONNX.

        Over-write `forward` because: `masks` is directly created with
        zero (valid position tag) and has the same spatial size as `x`.
        Thus the construction of `masks` is different from that in `forward`.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.

                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        num_levels = len(feats)
        img_metas_list = [img_metas for _ in range(num_levels)]
        return multi_apply(self.forward_single_onnx, feats, img_metas_list)

    def forward_single_onnx(self, x, img_metas):
        """"Forward function for a single feature level with ONNX exportation.

        Args:
            x (Tensor): Input feature from backbone's single stage, shape
                [bs, c, h, w].
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format (cx, cy, w, h).
                Shape [nb_dec, bs, num_query, 4].
        """
        # Note `img_shape` is not dynamically traceable to ONNX,
        # since the related augmentation was done with numpy under
        # CPU. Thus `masks` is directly created with zeros (valid tag)
        # and the same spatial shape as `x`.
        # The difference between torch and exported ONNX model may be
        # ignored, since the same performance is achieved (e.g.
        # 40.1 vs 40.1 for DETR)
        batch_size = x.size(0)
        h, w = x.size()[-2:]
        masks = x.new_zeros((batch_size, h, w))  # [B,h,w]

        x = self.input_proj(x)
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks.unsqueeze(1), size=x.shape[-2:]).to(torch.bool).squeeze(1)
        pos_embed = self.positional_encoding(masks)
        outs_dec, _ = self.transformer(x, masks, self.query_embedding.weight,
                                       pos_embed)

        all_cls_scores = self.fc_cls(outs_dec)
        all_bbox_preds = self.fc_reg(self.activate(
            self.reg_ffn(outs_dec))).sigmoid()
        return all_cls_scores, all_bbox_preds

    def onnx_export(self, all_cls_scores_list, all_bbox_preds_list, img_metas):
        """Transform network outputs into bbox predictions, with ONNX
        exportation.

        Args:
            all_cls_scores_list (list[Tensor]): Classification outputs
                for each feature level. Each is a 4D-tensor with shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds_list (list[Tensor]): Sigmoid regression
                outputs for each feature level. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            img_metas (list[dict]): Meta information of each image.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        assert len(img_metas) == 1, \
            'Only support one input image while in exporting to ONNX'

        cls_scores = all_cls_scores_list[-1][-1]
        bbox_preds = all_bbox_preds_list[-1][-1]

        # Note `img_shape` is not dynamically traceable to ONNX,
        # here `img_shape_for_onnx` (padded shape of image tensor)
        # is used.
        img_shape = img_metas[0]['img_shape_for_onnx']
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        batch_size = cls_scores.size(0)
        # `batch_index_offset` is used for the gather of concatenated tensor
        batch_index_offset = torch.arange(batch_size).to(
            cls_scores.device) * max_per_img
        batch_index_offset = batch_index_offset.unsqueeze(1).expand(
            batch_size, max_per_img)

        # supports dynamical batch inference
        if self.loss_cls.use_sigmoid:
            cls_scores = cls_scores.sigmoid()
            scores, indexes = cls_scores.view(batch_size, -1).topk(
                max_per_img, dim=1)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
        else:
            scores, det_labels = F.softmax(
                cls_scores, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img, dim=1)
            bbox_index = (bbox_index + batch_index_offset).view(-1)
            bbox_preds = bbox_preds.view(-1, 4)[bbox_index]
            det_labels = det_labels.view(-1)[bbox_index]
            bbox_preds = bbox_preds.view(batch_size, -1, 4)
            det_labels = det_labels.view(batch_size, -1)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_preds)
        # use `img_shape_tensor` for dynamically exporting to ONNX
        img_shape_tensor = img_shape.flip(0).repeat(2)  # [w,h,w,h]
        img_shape_tensor = img_shape_tensor.unsqueeze(0).unsqueeze(0).expand(
            batch_size, det_bboxes.size(1), 4)
        det_bboxes = det_bboxes * img_shape_tensor
        # dynamically clip bboxes
        x1, y1, x2, y2 = det_bboxes.split((1, 1, 1, 1), dim=-1)
        from mmdet.core.export import dynamic_clip_for_onnx
        x1, y1, x2, y2 = dynamic_clip_for_onnx(x1, y1, x2, y2, img_shape)
        det_bboxes = torch.cat([x1, y1, x2, y2], dim=-1)
        det_bboxes = torch.cat((det_bboxes, scores.unsqueeze(-1)), -1)

        return det_bboxes, det_labels
