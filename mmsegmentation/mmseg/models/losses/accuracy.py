# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


def accuracy(pred, target, topk=1, thresh=None, ignore_index=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        target (torch.Tensor): The target of each prediction, shape (N, , ...)
        ignore_index (int | None): The label index to be ignored. Default: None
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """

    """
    :param pred: 預測出來的結果，shape [batch_size, channel, height, width] 
    :param target: 標註的圖像，shape [batch_size, height, width]
    :param topk: 對於每個像素點我們取前k大的當作結果，只要其中一個是正確的該相素點就判定為預測正確
    :param thresh: 閾值，當預測置信度低於該閾值都視為沒有預測到
    :param ignore_index: 計算時忽略掉的index
    :return: 
    """
    # 已看過，用來計算正確率的函數
    # topk需要是int或是tuple型態
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        # 如果topk是int型態就轉成tuple，並且記錄下只會返回單一值
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        # 進入這裡表示batch_size為0，就會直接返回0
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        # 依據是單一值或是多個值會有list或是int的返回值
        return accu[0] if return_single else accu
    # 預測的shape會比標註的shape多出channel維度，在標註當中就直接是結果所以不會有channel維度
    assert pred.ndim == target.ndim + 1
    # batch_size需要一樣大
    assert pred.size(0) == target.size(0)
    # 如果前k大的k大於channel就會報錯
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    # pred_value, pred_label shape = [batch_size, maxk, height, width]
    pred_value, pred_label = pred.topk(maxk, dim=1)
    # transpose to shape (maxk, N, ...)，shape [maxk, batch_size, height, width]
    pred_label = pred_label.transpose(0, 1)
    # target shape [batch_size, height, width] -> [1, batch_size, height, width] -> [maxk, batch_size, height, width]
    # correct shape [maxk, batch_size, height, width]，相同的地方會是True否則就會是False
    correct = pred_label.eq(target.unsqueeze(0).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        # 如果有設定閾值，就會將預測值低於閾值的地方改成False
        correct = correct & (pred_value > thresh).t()
    if ignore_index is not None:
        # 將ignore的數字忽略掉
        correct = correct[:, target != ignore_index]
    res = []
    # eps = 一個很小的值，主要是避免除以0會報錯
    eps = torch.finfo(torch.float32).eps
    # 遍歷topk
    for k in topk:
        # Avoid causing ZeroDivisionError when all pixels
        # of an image are ignored
        # 下面就是計算正確率
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True) + eps
        if ignore_index is not None:
            total_num = target[target != ignore_index].numel() + eps
        else:
            total_num = target.numel() + eps
        res.append(correct_k.mul_(100.0 / total_num))
    return res[0] if return_single else res


class Accuracy(nn.Module):
    """Accuracy calculation module."""

    def __init__(self, topk=(1, ), thresh=None, ignore_index=None):
        """Module to calculate the accuracy.

        Args:
            topk (tuple, optional): The criterion used to calculate the
                accuracy. Defaults to (1,).
            thresh (float, optional): If not None, predictions with scores
                under this threshold are considered incorrect. Default to None.
        """
        super().__init__()
        self.topk = topk
        self.thresh = thresh
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            tuple[float]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk, self.thresh,
                        self.ignore_index)
