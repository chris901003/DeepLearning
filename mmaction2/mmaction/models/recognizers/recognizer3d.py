# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        # 已看過，定義訓練時每次調用執行的計算
        # imgs = 一個batch的影像資料，tensor shape [batch_size, num_clips, channel, clip_len, height, width]
        # labels = 一個batch的標註訊息，tensor shape [batch_size, labels]
        # kwargs = 其他參數，通常為空

        # 這裡會檢查我們一定需要有分類頭
        assert self.with_cls_head
        # 將imgs的通道進行調整，[batch_size * num_clips, channel, clip_len, height, width]，這裡的融合可以看成是batch_size變大
        # 反正不同的片段本身也不會有特徵融合的操作，所以可以看成獨立的batch資料
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        # 構建損失需要用的dict空間
        losses = dict()

        # 進行特徵提取，x shape = [batch_size * num_clips, channel, clip_len, height, width]
        x = self.extract_feat(imgs)
        if self.with_neck:
            # 如果有neck結構就會到這裡
            x, loss_aux = self.neck(x, labels.squeeze())
            losses.update(loss_aux)

        # 進入分類頭，cls_score shape = [batch_size * num_clips, num_classes]
        cls_score = self.cls_head(x)
        # 將labels如果有維度值是1的地方壓縮並且放到gt_labels上
        gt_labels = labels.squeeze()
        # 計算損失值，將預測結果以及標註訊息放進去
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        # 已看過，進行測試的向前傳遞
        # imgs = 圖像資料，tensor shape [batch_size, num_crop * num_clip, channel, clip_len, height, width]

        # 獲取當前batch_size
        batches = imgs.shape[0]
        # 獲取片段數量，這裡我們將不同的剪裁也視為不同的片段
        num_segs = imgs.shape[1]
        # 將imgs進行通道調整 [batch_size * num_crop * num_clip, channel, clip_len, height, width]，將batch維度進行融合
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.max_testing_views is not None:
            # 如果有設定max_testing_views就會到這裡
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                x = self.extract_feat(batch_imgs)
                if self.with_neck:
                    x, _ = self.neck(x)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            # 沒有設定max_testing_views就會到這裡
            # 進行特徵提取，feat shape = [batch_size * num_crop * num_clips, channel, clip_len, height, width]
            feat = self.extract_feat(imgs)
            if self.with_neck:
                # 如果有neck模塊就會到這裡
                feat, _ = self.neck(feat)

        if self.feature_extraction:
            # 如果有需要進行feature_extraction就會到這裡
            feat_dim = len(feat[0].size()) if isinstance(feat, tuple) else len(
                feat.size())
            assert feat_dim in [
                5, 2
            ], ('Got feature of unknown architecture, '
                'only 3D-CNN-like ([N, in_channels, T, H, W]), and '
                'transformer-like ([N, in_channels]) features are supported.')
            if feat_dim == 5:  # 3D-CNN architecture
                # perform spatio-temporal pooling
                avg_pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(feat, tuple):
                    feat = [avg_pool(x) for x in feat]
                    # concat them
                    feat = torch.cat(feat, axis=1)
                else:
                    feat = avg_pool(feat)
                # squeeze dimensions
                feat = feat.reshape((batches, num_segs, -1))
                # temporal average pooling
                feat = feat.mean(axis=1)
            return feat

        # should have cls_head if not extracting features
        # 檢查是否有分類頭，如果沒有分類頭就會報錯
        assert self.with_cls_head
        # 進行分類，cls_score shape = [batch_size * num_crop * num_clips, num_classes]
        cls_score = self.cls_head(feat)
        # 經過average_clip處理，cls_score shape = [batch_size, num_classes]
        cls_score = self.average_clip(cls_score, num_segs)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x, _ = self.neck(x)

        outs = self.cls_head(x)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)
