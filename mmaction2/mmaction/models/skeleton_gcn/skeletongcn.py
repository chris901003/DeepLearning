# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import RECOGNIZERS
from .base import BaseGCN


@RECOGNIZERS.register_module()
class SkeletonGCN(BaseGCN):
    """Spatial temporal graph convolutional networks."""

    def forward_train(self, skeletons, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        # 進行骨骼關節點的GCN預測部分，這裡是會計算損失值
        # 檢查需要有分類頭
        assert self.with_cls_head
        # 構建損失保存字典
        losses = dict()

        # 進行特徵提取，x shape [batch_size * people, channel, frames, num_node]
        x = self.extract_feat(skeletons)
        # 將提取結果透過分類頭將channel調整到分類數量，output shape [batch_size, num_classes]
        output = self.cls_head(x)
        gt_labels = labels.squeeze(-1)
        # 透過cls_head當中的loss計算損失，當中會有topk的損失以及向後傳遞的損失值
        loss = self.cls_head.loss(output, gt_labels)
        # 更新losses字典
        losses.update(loss)

        # 回傳損失值
        return losses

    def forward_test(self, skeletons):
        """Defines the computation performed at every call when evaluation and
        testing."""
        x = self.extract_feat(skeletons)
        assert self.with_cls_head
        output = self.cls_head(x)

        return output.data.cpu().numpy()
