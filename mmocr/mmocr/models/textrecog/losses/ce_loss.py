# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmocr.models.builder import LOSSES


@LOSSES.register_module()
class CELoss(nn.Module):
    """Implementation of loss module for encoder-decoder based text recognition
    method with CrossEntropy loss.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
        ignore_first_char (bool): Whether to ignore the first token in target (
            usually the start token). If ``True``, the last token of the output
            sequence will also be removed to be aligned with the target length.
    """

    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 ignore_first_char=False):
        """ 已看過，主要是提供encoder-decoder的文字判讀的損失計算，且計算方式使用CrossEntropy
        Args:
            ignore_index: 需要忽略掉不計算loss的index
            reduction: 統合loss的方式
            ignore_first_char: 是否需要忽略第一個文字
        """
        # 繼承自nn.Module，對繼承對象進行初始化
        super().__init__()
        # ignore_index需要是int格式
        assert isinstance(ignore_index, int)
        # reduction需要是string
        assert isinstance(reduction, str)
        # reduction方式有3種
        assert reduction in ['none', 'mean', 'sum']
        # ignore_first_char會是bool型態
        assert isinstance(ignore_first_char, bool)

        # 構建交叉熵的實例對象
        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction)
        # 保存ignore_first_char資訊
        self.ignore_first_char = ignore_first_char

    def format(self, outputs, targets_dict):
        """ 已看過，對於傳入的outputs與targets_dict進行格式化
        Args:
            outputs: 預測出來的結果，tensor shape = [batch_size, seq_len, channel=num_classes]
            targets_dict: 標註訊息的dict，裡面會有gt訊息
        """
        # 將padded過後的標註訊息提取出來
        targets = targets_dict['padded_targets']
        if self.ignore_first_char:
            # 如果需要將第一個文字進行忽略就會到這裡
            # 將targets的第一個文字忽略
            targets = targets[:, 1:].contiguous()
            # 將預測的最後一個忽略
            outputs = outputs[:, :-1, :]

        # 調整通道順序，shape [batch_size, seq_len, channel=num_classes] -> [batch_size, channel=num_classes, seq_len]
        outputs = outputs.permute(0, 2, 1).contiguous()

        # 將結果回傳
        return outputs, targets

    def forward(self, outputs, targets_dict, img_metas=None):
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            targets_dict (dict): A dict with a key ``padded_targets``, which is
                a tensor of shape :math:`(N, T)`. Each element is the index of
                a character.
            img_metas (None): Unused.

        Returns:
            dict: A loss dict with the key ``loss_ce``.
        """
        # 已看過，計算CrossEntropy損失
        # outputs = 預測出來的結果，tensor shape = [batch_size, seq_len, channel=num_classes]
        # targets_dict = 標註訊息的dict，裡面會有gt訊息
        # img_metas = 每個圖像的詳細資料，這裡不會用到

        # 對傳入的outputs與targets_dict進行格式化
        # outputs shape = [batch_size * (seq_len - 1), channel=num_classes]
        # targets shape = [batch_size * (seq_len - 1)]
        outputs, targets = self.format(outputs, targets_dict)

        # 計算交叉熵的損失，這裡傳入的outputs與targets需要符合pytorch官方的通道排列
        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        # 將損失值放到dict當中
        losses = dict(loss_ce=loss_ce)

        # 最後回傳損失值
        return losses


@LOSSES.register_module()
class SARLoss(CELoss):
    """Implementation of loss module in `SAR.

    <https://arxiv.org/abs/1811.00751>`_.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ("none", "mean", "sum").

    Warning:
        SARLoss assumes that the first input token is always `<SOS>`.
    """

    def __init__(self, ignore_index=-1, reduction='mean', **kwargs):
        super().__init__(ignore_index, reduction)

    def format(self, outputs, targets_dict):
        targets = targets_dict['padded_targets']
        # targets[0, :], [start_idx, idx1, idx2, ..., end_idx, pad_idx...]
        # outputs[0, :, 0], [idx1, idx2, ..., end_idx, ...]

        # ignore first index of target in loss calculation
        targets = targets[:, 1:].contiguous()
        # ignore last index of outputs to be in same seq_len with targets
        outputs = outputs[:, :-1, :].permute(0, 2, 1).contiguous()

        return outputs, targets


@LOSSES.register_module()
class TFLoss(CELoss):
    """Implementation of loss module for transformer.

    Args:
        ignore_index (int, optional): The character index to be ignored in
            loss computation.
        reduction (str): Type of reduction to apply to the output,
            should be one of the following: ("none", "mean", "sum").
        flatten (bool): Whether to flatten the vectors for loss computation.

    Warning:
        TFLoss assumes that the first input token is always `<SOS>`.
    """

    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 flatten=True,
                 **kwargs):
        """ 已看過，構建transformer的損失計算方式
        Args:
            ignore_index: 需要忽略掉不計算loss的index
            reduction: 統合loss的方式
            flatten: 是否進行攤平
        """
        # 繼承自CELoss，對繼承對象進行初始化
        super().__init__(ignore_index, reduction)
        assert isinstance(flatten, bool)

        # 保存flatten資料
        self.flatten = flatten

    def format(self, outputs, targets_dict):
        """ 已看過，transformer專用的交叉熵loss計算的格式化函數
        Args:
            outputs: 預測的資料，tensor shape = [batch_size, seq_len, channel=num_classes]
            targets_dict: 標註訊息資料，會是dict格式
        """
        # 會將最後一個時間點的序列刪除，outputs shape = [batch_size, seq_len - 1, channel=num_classes]
        outputs = outputs[:, :-1, :].contiguous()
        # 將targets_dict當中padded_targets資訊取出來，targets shape = [batch_size, seq_len]
        targets = targets_dict['padded_targets']
        # 將targets當中的第一個序列移除，targets shape = [batch_size, seq_len - 1]
        targets = targets[:, 1:].contiguous()
        if self.flatten:
            # 如果需要進行攤平就會到這裡
            # outputs shape = [batch_size, seq_len - 1, num_classes] -> [batch_size * (seq_len - 1), num_classes]
            outputs = outputs.view(-1, outputs.size(-1))
            # targets shape = [batch_size, seq_len - 1] -> [batch_size * (seq_len - 1)]
            targets = targets.view(-1)
        else:
            # 如果不需要攤平就會調整通道順序
            # outputs shape = [batch_size, seq_len, num_classes] -> [batch_size, num_classes, seq_len]
            outputs = outputs.permute(0, 2, 1).contiguous()

        # 將格式化好的outputs與targets進行回傳
        return outputs, targets
