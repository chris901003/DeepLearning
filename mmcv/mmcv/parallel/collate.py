# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from .data_container import DataContainer


def collate(batch: Sequence, samples_per_gpu: int = 1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """
    # 已看過，這個函數會作為dataloader的collect_fn

    # batch = list[dict]，list的長度就會是batch_size，裏面的dict資料就會是每張圖像的詳細資訊(在第一次進入到collate時的傳入內容)
    if not isinstance(batch, Sequence):
        # 檢查傳入的batch是否合法
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        # 構建stacked的存放位置
        stacked = []
        if batch[0].cpu_only:
            # 這裡會進行遍歷，我們會以一個gpu的batch_size為單位
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    # 將一個gpu的batch_size丟入到stacked當中
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            # stacked = list[list[dict]]
            # 最後回傳的是DataContainer實例對象
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            # 可以進行堆疊的資料會進入到這裡
            # 遍歷batch當中的資料
            for i in range(0, len(batch), samples_per_gpu):
                # 檢查batch當中的data需要是tensor這樣才可以進行堆疊
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    # 如果有設定pad_dims就會進來
                    # ndim = 獲取batch當中訓練圖像的ndim，如果是2d圖像就會是3(channel, height, width)
                    ndim = batch[i].dim()
                    # 如果pad_dims大於ndim就會報錯，pad_dims表示padding的維度有多少個，在2d圖中就是高寬兩個
                    assert ndim > batch[i].pad_dims
                    # 構建出max_shape列表，長度就是pad_dims且預設值為0
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    # 將max_shape進行給值，max_shape的值就會是圖像的高寬
                    for dim in range(1, batch[i].pad_dims + 1):
                        # 這裡用-dim是因為最前面會是channel
                        # 第一次取-1就是取width，第二次取-2就是height(資料為2d圖像時)
                        max_shape[dim - 1] = batch[i].size(-dim)
                    # 遍歷一個gpu的batch_size
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            # 一個batch當中除了可以padding部分以外其他的深度必須要相同，所以會在這裡進行檢查
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            # 將max_shape變成整個batch當中最大的高寬，也就是最後所有的圖像會透過padding變成max_shape的高寬
                            # 所以需要先找到高寬的最大值
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    # padded_samples = padding完成的圖像tensor會紀錄在這裡
                    padded_samples = []
                    # 遍歷一個gpu的batch_size
                    for sample in batch[i:i + samples_per_gpu]:
                        # 構建出一個pad列表，長度為pad_dims的兩倍且預設都為0
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        # 透過最大的高寬減去當前圖像的高寬就會知道需要填充多少，存到pad當中
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            # 將圖像進行padding，讓每個tensor最終的高寬會相同，這樣才可以進行堆疊
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    # 透過default_collate將padded_samples傳入
                    # 出來的就會是堆疊好的tensor，shape [batch_size, channel, height, width]
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        # 如果組成的方式是Mapping就會到這裡來
        return {
            # 遍歷batch[0]當中的內容同時也遍歷batch，也就是會將一個照片的某個資訊先堆疊起來(Ex:第一次先對img_metas堆疊)
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)
