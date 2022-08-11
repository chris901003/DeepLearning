# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule

from mmocr.models.builder import DECODERS


@DECODERS.register_module()
class BaseDecoder(BaseModule):
    """Base decoder class for text recognition."""

    def __init__(self, init_cfg=None, **kwargs):
        super().__init__(init_cfg=init_cfg)

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        # 已看過，decoder部分的forward函數
        self.train_mode = train_mode
        if train_mode:
            # 如果當前是訓練模式就會到這裡，return shape = [batch_size, seq_len, num_classes]
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        # 在測試模式下就不會將target_dict資料傳入進去，return shape = [batch_size, seq_len, num_classes]
        return self.forward_test(feat, out_enc, img_metas)
