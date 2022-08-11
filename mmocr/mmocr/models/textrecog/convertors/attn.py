# Copyright (c) OpenMMLab. All rights reserved.
import torch

import mmocr.utils as utils
from mmocr.models.builder import CONVERTORS
from .base import BaseConvertor


@CONVERTORS.register_module()
class AttnConvertor(BaseConvertor):
    """Convert between text, index and tensor for encoder-decoder based
    pipeline.

    Args:
        dict_type (str): Type of dict, should be one of {'DICT36', 'DICT90'}.
        dict_file (None|str): Character dict file path. If not none,
            higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, higher
            priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        max_seq_len (int): Maximum sequence length of label.
        lower (bool): If True, convert original string to lower case.
        start_end_same (bool): Whether use the same index for
            start and end token or not. Default: True.
    """

    def __init__(self,
                 dict_type='DICT90',
                 dict_file=None,
                 dict_list=None,
                 with_unknown=True,
                 max_seq_len=40,
                 lower=False,
                 start_end_same=True,
                 **kwargs):
        """ 已看過，將index與文字與tensor格式之間轉換，這裡是針對encoder-decoder設計
        Args:
            dict_type: 資料集的文字集
            dict_file: 透過file獲取文字集
            dict_list: 透過很多的file獲取文字集
            with_unknown: 是否包含unknown這個index
            max_seq_len: 最大序列長度
            lower: 是否需要將標註文字轉成小寫
            start_end_same: start與end的index是否相同
        """
        # 繼承自BaseConverter，對繼承對象進行初始化
        super().__init__(dict_type, dict_file, dict_list)
        # 檢查傳入的資料型態是否正確
        assert isinstance(with_unknown, bool)
        assert isinstance(max_seq_len, int)
        assert isinstance(lower, bool)

        # 保存傳入的資料
        self.with_unknown = with_unknown
        self.max_seq_len = max_seq_len
        self.lower = lower
        self.start_end_same = start_end_same

        # 將對應關係更新，添加上一些對應關係
        self.update_dict()

    def update_dict(self):
        # 已看過，添加一些對應關係

        # 開始與結束的標籤名稱
        start_end_token = '<BOS/EOS>'
        # unknown的標籤名稱
        unknown_token = '<UKN>'
        # padding的標籤名稱
        padding_token = '<PAD>'

        # unknown
        # 將unknown的index預設為None
        self.unknown_idx = None
        if self.with_unknown:
            # 如果有需要unknown的index就會到這裡將unknown添加上去，這裡會加在最後面
            self.idx2char.append(unknown_token)
            self.unknown_idx = len(self.idx2char) - 1

        # BOS/EOS
        # 將開始與結束標籤往最後的地方添加
        self.idx2char.append(start_end_token)
        self.start_idx = len(self.idx2char) - 1
        if not self.start_end_same:
            # 如果start與end不是同一個index就會到這裡，再將token往最後面加
            self.idx2char.append(start_end_token)
        self.end_idx = len(self.idx2char) - 1

        # padding
        # padding部分添加上去
        self.idx2char.append(padding_token)
        self.padding_idx = len(self.idx2char) - 1

        # update char2idx
        # 最後一次更新char對應到idx的資料
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def str2tensor(self, strings):
        """
        Convert text-string into tensor.
        Args:
            strings (list[str]): ['hello', 'world']
        Returns:
            dict (str: Tensor | list[tensor]):
                tensors (list[Tensor]): [torch.Tensor([1,2,3,3,4]),
                                                    torch.Tensor([5,4,6,3,7])]
                padded_targets (Tensor(bsz * max_seq_len))
        """
        # 已看過，將標註的str轉成tensor格式，將文字映射到對應的index之後再轉成tensor格式
        # strings = list[str]，list長度就會是batch_size

        # 檢查傳入的strings是否為list且當中的資料是str格式
        assert utils.is_type_list(strings, str)

        # 最終要回傳的資料保存的地方
        tensors, padded_targets = [], []
        # 將strings的內容轉成index，indexes = list[list[int]]，第一個list長度會是batch_size，第二個list會是該圖像的字串長度
        indexes = self.str2idx(strings)
        # 遍歷一個batch的標註資料
        for index in indexes:
            # 將index的資料轉成tensor格式
            tensor = torch.LongTensor(index)
            # 添加到tensors當中保存
            tensors.append(tensor)
            # target tensor for loss
            # src_target = 長度會是len(index)+2且全為0
            src_target = torch.LongTensor(tensor.size(0) + 2).fill_(0)
            # 在最後的部分更改成end的index
            src_target[-1] = self.end_idx
            # 在最前的部分更改成start的index
            src_target[0] = self.start_idx
            # 中間部分將tensor資料放入
            src_target[1:-1] = tensor
            # 構建長度為max_seq_len且全為padding的index的tensor
            padded_target = (torch.ones(self.max_seq_len) *
                             self.padding_idx).long()
            # 獲取當前標註的文字長度
            char_num = src_target.size(0)
            if char_num > self.max_seq_len:
                # 如果當前標註的文字長度大於最大序列長度，我們就會取前max_seq_len放到padded_target當中
                padded_target = src_target[:self.max_seq_len]
            else:
                # 如果max_seq_len比當前標註文字長度長就會將前src_target變成當前標註字串
                padded_target[:char_num] = src_target
            # 最後保存
            padded_targets.append(padded_target)
        # 最後在第0個維度進行stack，padded_targets = tensor shape [batch_size, len=max_seq_len]
        padded_targets = torch.stack(padded_targets, 0).long()

        # 將最後結果回傳
        return {'targets': tensors, 'padded_targets': padded_targets}

    def tensor2idx(self, outputs, img_metas=None):
        """
        Convert output tensor to text-index
        Args:
            outputs (tensor): model outputs with size: N * T * C
            img_metas (list[dict]): Each dict contains one image info.
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]]
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                                         [0.9,0.9,0.98,0.97,0.96]]
        """
        # 已看過，將預測結果的tensor格式轉回成index格式
        # outputs = tensor shape [batch_size, seq_len, num_classes]
        # img_metas = 圖像的詳細資訊

        # 獲取當前的batch大小
        batch_size = outputs.size(0)
        # 獲取當前需被忽略的index值
        ignore_indexes = [self.padding_idx]
        # 最終結果保存的地方
        indexes, scores = [], []
        # 遍歷整個batch的資料
        for idx in range(batch_size):
            # 獲取當前圖像的序列資訊，seq shape = [seq_len, num_classes]
            seq = outputs[idx, :, :]
            # 將num_classes通道進行softmax概率化
            seq = seq.softmax(dim=-1)
            # 獲取置信度分數最大的分類類別以及置信度分數，max_value與max_idx的shape = [seq_len]
            max_value, max_idx = torch.max(seq, -1)
            # 一張圖像資料的保存地方
            str_index, str_score = [], []
            # 將tensor轉成ndarray同時將資料轉好cpu上
            output_index = max_idx.cpu().detach().numpy().tolist()
            output_score = max_value.cpu().detach().numpy().tolist()
            # 開始遍歷整個序列的長度
            for char_index, char_score in zip(output_index, output_score):
                if char_index in ignore_indexes:
                    # 如果獲取的index是需要忽略的index就直接continue
                    continue
                if char_index == self.end_idx:
                    # 如果獲取的是結束的index就直接break跳出
                    break
                # 否則就將預測的結果放到str_index與str_score當中
                str_index.append(char_index)
                str_score.append(char_score)

            # 最後完整的句子資訊就會放到indexes與scores當中
            indexes.append(str_index)
            scores.append(str_score)

        # 回傳
        return indexes, scores
