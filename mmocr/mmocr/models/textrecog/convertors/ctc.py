# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn.functional as F

import mmocr.utils as utils
from mmocr.models.builder import CONVERTORS
from .base import BaseConvertor


@CONVERTORS.register_module()
class CTCConvertor(BaseConvertor):
    """Convert between text, index and tensor for CTC loss-based pipeline.

    Args:
        dict_type (str): Type of dict, should be either 'DICT36' or 'DICT90'.
        dict_file (None|str): Character dict file path. If not none, the file
            is of higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, the list
            is of higher priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        lower (bool): If True, convert original string to lower case.
    """

    def __init__(self,
                 dict_type='DICT90',
                 dict_file=None,
                 dict_list=None,
                 with_unknown=True,
                 lower=False,
                 **kwargs):
        """ 已看過，CTC初始化部分
        Args:
            dict_type: 使用的資料集格式，這裡只會有DICT90或是DICT36兩種選項
            dict_file: 對單一英文字的資料集格式，這裡的優先級會大於dict_type
            dict_list: 文字的list，這裡的優先級會大於dict_file
            with_unknown: 是否會有unknown這個值
            lower: 將原始的字串變成小寫格式
        """
        # 繼承自BaseConverter，對繼承對象進行初始化
        super().__init__(dict_type, dict_file, dict_list)
        # 檢查with_unknown以及lower的型態
        assert isinstance(with_unknown, bool)
        assert isinstance(lower, bool)

        # 保存傳入資料
        self.with_unknown = with_unknown
        self.lower = lower
        # 主要是將<BLK>與<UKN>標籤放到轉換表當中
        self.update_dict()

    def update_dict(self):
        # CTC-blank
        # 已看過，將BLK以及UKN標籤放入到轉換表當中

        # 構建blank的token
        blank_token = '<BLK>'
        # 保存blank_idx的index
        self.blank_idx = 0
        # 將<BLK>標籤放到index對應到char的第0個index
        self.idx2char.insert(0, blank_token)

        # unknown，設定unknown標籤
        self.unknown_idx = None
        if self.with_unknown:
            # 如果有需要unknown標籤就會到這裡，將<UKN>標籤放到index轉char的最後一個index上
            self.idx2char.append('<UKN>')
            # 更新unknown在的index值
            self.unknown_idx = len(self.idx2char) - 1

        # update char2idx，重新構建char轉到index的dict
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def str2tensor(self, strings):
        """Convert text-string to ctc-loss input tensor.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            dict (str: tensor | list[tensor]):
                tensors (list[tensor]): [torch.Tensor([1,2,3,3,4]),
                    torch.Tensor([5,4,6,3,7])].
                flatten_targets (tensor): torch.Tensor([1,2,3,3,4,5,4,6,3,7]).
                target_lengths (tensor): torch.IntTensot([5,5]).
        """
        # 已看過，將標註的字串轉成對應的index後再轉成tensor格式
        # string = list[str]，list長度就會是batch_size且資料會是str，str當中就是該圖像對應的標註訊息
        # 檢查string當中是否為list且當中資料是否為str
        assert utils.is_type_list(strings, str)

        # 最後轉成tensor格式的標註訊息
        tensors = []
        # 透過str2idx將strings資料轉成對應的index
        # list[list[int]]，第一個list長度就會是batch_size，第二個會是對應圖像的文字長度，int就會是對應上的index
        indexes = self.str2idx(strings)
        # 遍歷整個batch的index資訊
        for index in indexes:
            # 將index資訊轉成tensor格式
            tensor = torch.IntTensor(index)
            # 將tensor資料保存下來
            tensors.append(tensor)
        # target_lengths = 在計算CTC損失時需要的資料，需要知道每張圖像的標註文字長度
        target_lengths = torch.IntTensor([len(t) for t in tensors])
        # 將標註訊息進行展平，也就是shape [一個batch的圖像中總共有多少文字]
        flatten_target = torch.cat(tensors)

        # 將原始標註的tensors以及攤平後的標註tensors以及每一個有多少長度文字打包成dict進行回傳
        return {
            'targets': tensors,
            'flatten_targets': flatten_target,
            'target_lengths': target_lengths
        }

    def tensor2idx(self, output, img_metas, topk=1, return_topk=False):
        """Convert model output tensor to index-list.
        Args:
            output (tensor): The model outputs with size: N * T * C.
            img_metas (list[dict]): Each dict contains one image info.
            topk (int): The highest k classes to be returned.
            return_topk (bool): Whether to return topk or just top1.
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                [0.9,0.9,0.98,0.97,0.96]]
                (
                    indexes_topk (list[list[list[int]->len=topk]]):
                    scores_topk (list[list[list[float]->len=topk]])
                ).
        """
        # 已看過，在測試時將 預測結果的tensor轉成對應的index
        # output = 預測的輸出，tensor shape [batch_size=1, width, channel]
        # img_metas = 當前batch圖像的詳細資訊
        # topk = 前k大的可能進行回傳
        # return_topk = 是否將topk都進行回傳，如果是False就只會回傳最大的

        # 檢查傳入的img_metas是否是dict格式
        assert utils.is_type_list(img_metas, dict)
        # img_metas的資料數量要與output的數量對齊
        assert len(img_metas) == output.size(0)
        # topk需要是int格式
        assert isinstance(topk, int)
        assert topk >= 1

        # 獲取valid_ratios資訊
        valid_ratios = [
            img_meta.get('valid_ratio', 1.0) for img_meta in img_metas
        ]

        # 獲取batch大小
        batch_size = output.size(0)
        # 將預測出來的結果通過softmax，這裡是在channel維度上面進行softmax
        output = F.softmax(output, dim=2)
        # 將tensor轉到cpu上同時取消反向傳遞的計算
        output = output.cpu().detach()
        # 獲取每個序列時間點topk的值以及對應的index
        # batch_topk_value與batch_topk_idx的shape = [batch_size=1, width, channel=topk]
        batch_topk_value, batch_topk_idx = output.topk(topk, dim=2)
        # 獲取top1的資訊，batch_max_idx shape = [batch_size=1, width]
        batch_max_idx = batch_topk_idx[:, :, 0]
        # 記錄下topk的資訊
        scores_topk, indexes_topk = [], []
        # 置信度分數以及index的保存
        scores, indexes = [], []
        # 獲取序列長度資訊
        feat_len = output.size(1)
        # 遍歷一個batch的預測
        for b in range(batch_size):
            # 獲取valid_ratio資訊
            valid_ratio = valid_ratios[b]
            # 根據valid_ratio會決定最終decode的序列長度，因為有些部分是透過padding變長的，這樣padding部分就會需要忽略掉
            decode_len = min(feat_len, math.ceil(feat_len * valid_ratio))
            # 獲取該圖像的預測當中每個序列上置信度最大的index
            pred = batch_max_idx[b, :]
            # 保存每個序列時間點上選擇的index
            select_idx = []
            prev_idx = self.blank_idx
            # 遍歷我們需要解碼的長度，這裡就會是執行將blank忽略同時如果有相同連續的index就只會取一個的操作
            for t in range(decode_len):
                # 獲取該序列點上最大置信度的index
                tmp_value = pred[t].item()
                if tmp_value not in (prev_idx, self.blank_idx):
                    # 如果選擇的index不是blank_idx也不是prev_idx就會到這裡，直接保存到select_idx當中
                    select_idx.append(t)
                # 將prev_idx變成當前的index
                prev_idx = tmp_value
            # 將select_idx轉成LongTensor格式
            select_idx = torch.LongTensor(select_idx)
            # 將select_idx套用到batch_topk_value與batch_topk_idx，這樣就會是過濾掉連續相同index與空白的部分
            topk_value = torch.index_select(batch_topk_value[b, :, :], 0,
                                            select_idx)  # valid_seqlen * topk
            topk_idx = torch.index_select(batch_topk_idx[b, :, :], 0,
                                          select_idx)
            # 這裡將資料轉成list[list]，第一個list就會是最後的預測文字長度，第二個list長度會是topk
            topk_idx_list, topk_value_list = topk_idx.numpy().tolist(
            ), topk_value.numpy().tolist()
            # 將topk_idx_list與topk_value_list進行保存
            indexes_topk.append(topk_idx_list)
            scores_topk.append(topk_value_list)
            # 這裡會將top1的資料另外保存
            indexes.append([x[0] for x in topk_idx_list])
            scores.append([x[0] for x in topk_value_list])

        if return_topk:
            # 如果需要topk的全部資料就會回傳這些
            return indexes_topk, scores_topk

        # 否則我們只會回傳top1的資料
        return indexes, scores
