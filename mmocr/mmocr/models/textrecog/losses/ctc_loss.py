# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn

from mmocr.models.builder import LOSSES


@LOSSES.register_module()
class CTCLoss(nn.Module):
    """Implementation of loss module for CTC-loss based text recognition.

    Args:
        flatten (bool): If True, use flattened targets, else padded targets.
        blank (int): Blank label. Default 0.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
        zero_infinity (bool): Whether to zero infinite losses and
            the associated gradients. Default: False.
            Infinite losses mainly occur when the inputs
            are too short to be aligned to the targets.
    """

    def __init__(self,
                 flatten=True,
                 blank=0,
                 reduction='mean',
                 zero_infinity=False,
                 **kwargs):
        """ 已看過，構建CTC損失計算初始化方式
        Args:
            flatten: 如果是True就是用flatten後的target，如果是False就是用padding後的target
            blank: 空白的index
            reduction: 透過哪種方式將損失合併
            zero_infinity: 是否當損失值變成無窮時歸為0且將反向傳遞也變成0，這裡預設會是關閉，造成損失無窮的原因會是傳入的字串長度過短
            kwargs: 其他參數，這裡會有ignore_index資訊
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        # 檢查傳入的資料是否符合規定
        assert isinstance(flatten, bool)
        assert isinstance(blank, int)
        assert isinstance(reduction, str)
        assert isinstance(zero_infinity, bool)

        # 將傳入的資料進行保存
        self.flatten = flatten
        self.blank = blank
        # 構建torch官方實現的CTC損失，這裡會需要指定blank存放的index
        self.ctc_loss = nn.CTCLoss(
            blank=blank, reduction=reduction, zero_infinity=zero_infinity)

    def forward(self, outputs, targets_dict, img_metas=None):
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            targets_dict (dict): A dict with 3 keys ``target_lengths``,
                ``flatten_targets`` and ``targets``.

                - | ``target_lengths`` (Tensor): A tensor of shape :math:`(N)`.
                    Each item is the length of a word.

                - | ``flatten_targets`` (Tensor): Used if ``self.flatten=True``
                    (default). A tensor of shape
                    (sum(targets_dict['target_lengths'])). Each item is the
                    index of a character.

                - | ``targets`` (Tensor): Used if ``self.flatten=False``. A
                    tensor of :math:`(N, T)`. Empty slots are padded with
                    ``self.blank``.

            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            dict: The loss dict with key ``loss_ctc``.
        """
        # 已看過，計算CTC損失值
        # outputs = 預測結果，tensor shape [batch_size, width, channel]
        # targets_dict = 標註訊息相關內容
        # img_metas = 圖像的詳細資訊

        valid_ratios = None
        if img_metas is not None:
            # 獲取img_metas當中的valid_ratios資訊
            valid_ratios = [
                # 如果當中沒有valid_ratio資訊就默認為1.0
                img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
            ]

        # 將預測的結果透過log_softmax，也就是log(softmax(tensor))的操作，這裡是對channel部分進行softmax
        outputs = torch.log_softmax(outputs, dim=2)
        # 獲取batch_size以及序列的長度
        bsz, seq_len = outputs.size(0), outputs.size(1)
        # 進行通道調整 [batch_size, width, channel] -> [width, batch_size, channel]
        outputs_for_loss = outputs.permute(1, 0, 2).contiguous()  # T * N * C

        if self.flatten:
            # 如果有設定將標註訊息全部攤平就會到這裡，記錄下攤平後的結果
            targets = targets_dict['flatten_targets']
        else:
            targets = torch.full(
                size=(bsz, seq_len), fill_value=self.blank, dtype=torch.long)
            for idx, tensor in enumerate(targets_dict['targets']):
                valid_len = min(tensor.size(0), seq_len)
                targets[idx, :valid_len] = tensor[:valid_len]

        # 獲取每張圖像的標註文字長度
        target_lengths = targets_dict['target_lengths']
        # 將標註的文字長度控制在最大序列長度以下
        target_lengths = torch.clamp(target_lengths, min=1, max=seq_len).long()

        # 構建input_lengths，這裡的長度都會是一樣的就是width
        input_lengths = torch.full(
            size=(bsz, ), fill_value=seq_len, dtype=torch.long)
        if not self.flatten and valid_ratios is not None:
            # 如果沒有攤平就會到這裡
            input_lengths = [
                math.ceil(valid_ratio * seq_len)
                for valid_ratio in valid_ratios
            ]
            input_lengths = torch.Tensor(input_lengths).long()

        # 進行CTC的損失計算
        # 對於pytorch官方的CTC有問題可以到:https://zhuanlan.zhihu.com/p/67415439
        loss_ctc = self.ctc_loss(outputs_for_loss, targets, input_lengths,
                                 target_lengths)

        # 將損失放到dict當中
        losses = dict(loss_ctc=loss_ctc)

        # 回傳損失的dict
        return losses
