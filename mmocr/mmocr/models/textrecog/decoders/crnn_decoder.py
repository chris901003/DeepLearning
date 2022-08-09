# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import Sequential

from mmocr.models.builder import DECODERS
from mmocr.models.textrecog.layers import BidirectionalLSTM
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class CRNNDecoder(BaseDecoder):
    """Decoder for CRNN.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        rnn_flag (bool): Use RNN or CNN as the decoder.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels=None,
                 num_classes=None,
                 rnn_flag=False,
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        """ 已看過，CRNN的decoder初始化部分
        Args:
            in_channels: 輸入的channel深度
            num_classes: 最後分類的類別數量
            rnn_flag: 在decoder當中要使用rnn或是cnn，如果是True表示在decoder當中要用rnn
            init_cfg: 初始化方式設定
        """
        # 繼承自BaseDecoder，將繼承對象進行初始化
        super().__init__(init_cfg=init_cfg)
        # 保存最後分類類別數量
        self.num_classes = num_classes
        # 保存要用rnn或是cnn
        self.rnn_flag = rnn_flag

        if rnn_flag:
            # 如果使用的是rnn就會到這裡
            self.decoder = Sequential(
                # 這裡會使用到雙向的LSTM循環神經網路，且傳入的資料是(nIN, nHidden, nOut)
                BidirectionalLSTM(in_channels, 256, 256),
                BidirectionalLSTM(256, 256, num_classes))
        else:
            # 如果使用的是cnn就會到這裡，就直接將輸入的channel調整到最後分類類別數量
            self.decoder = nn.Conv2d(
                in_channels, num_classes, kernel_size=1, stride=1)

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        """
        Args:
            feat: 透過backbone提取出來的特徵圖資訊，tensor shape [batch_size, channel, height=1, width]
            out_enc: 從encoder輸出的資料，在CRNN當中不會用到encoder層結構，所以這裡會是None
            targets_dict: 標註相關的dict，詳細內容到encoder_decode_recognizer.py當中找
            img_metas: 一個batch當中圖像的詳細資訊
        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.
        """
        # 已看過，CRNN的訓練模式下的decoder的forward部分
        # CRNN當中最終的height會是1，這裡會檢查傳入的特徵圖是否符合
        assert feat.size(2) == 1, 'feature height must be 1'
        if self.rnn_flag:
            # 如果是用rnn進行decode就會到這裡
            # 先將高度維度進行壓縮，這裡高度會是1所以不會有問題
            x = feat.squeeze(2)  # [N, C, W]
            # 將通道進行重新排列，將寬度部分放到最前面
            x = x.permute(2, 0, 1)  # [W, N, C]
            # 進入decoder的BLSTM層結構
            x = self.decoder(x)  # [W, N, C]
            # 最終調整通道順序為 [batch_size, width, channel]
            outputs = x.permute(1, 0, 2).contiguous()
        else:
            # 如果是用cnn進行decode就會到這裡
            # 直接透過卷積將channel調整到最終份類數量
            x = self.decoder(feat)
            # 將通道進行調整 [batch_size, channel, height, width] -> [batch_size, width, channel, height]
            x = x.permute(0, 3, 1, 2).contiguous()
            # 將shape提取出來
            n, w, c, h = x.size()
            # 最後將height部分壓縮，shape [batch_size, width, channel]
            outputs = x.view(n, w, c * h)
        # outputs shape = [batch_size, width, channel]
        return outputs

    def forward_test(self, feat, out_enc, img_metas):
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, H, 1, W)`.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.
        """
        return self.forward_train(feat, out_enc, None, img_metas)
