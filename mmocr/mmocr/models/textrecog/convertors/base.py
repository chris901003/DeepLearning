# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import CONVERTORS
from mmocr.utils import list_from_file


@CONVERTORS.register_module()
class BaseConvertor:
    """Convert between text, index and tensor for text recognize pipeline.

    Args:
        dict_type (str): Type of dict, options are 'DICT36', 'DICT37', 'DICT90'
            and 'DICT91'.
        dict_file (None|str): Character dict file path. If not none,
            the dict_file is of higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, the list
            is of higher priority than dict_type, but lower than dict_file.
    """
    # 這裡預設start與end與padding的index都為0
    start_idx = end_idx = padding_idx = 0
    # unknown的index預設為None，表示不使用unknown
    unknown_idx = None
    # 是否將標註轉成小寫英文，預設為False
    lower = False

    dicts = dict(
        DICT36=tuple('0123456789abcdefghijklmnopqrstuvwxyz'),
        DICT90=tuple('0123456789abcdefghijklmnopqrstuvwxyz'
                     'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()'
                     '*+,-./:;<=>?@[\\]_`~'),
        # With space character
        DICT37=tuple('0123456789abcdefghijklmnopqrstuvwxyz '),
        DICT91=tuple('0123456789abcdefghijklmnopqrstuvwxyz'
                     'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()'
                     '*+,-./:;<=>?@[\\]_`~ '))

    def __init__(self, dict_type='DICT90', dict_file=None, dict_list=None):
        """ 已看過，label轉換的基底初始化部分
        Args:
            dict_type: 字典的型態，會有DICT[36, 37, 90, 91]四種可能
            dict_file: 字符字典文件路徑，如果有設定的話優先序會大於dict_type
            dict_list: 文字的list，如果有設定的話優先順序會最大
        """
        # 檢查傳入的dict_file與dict_list是否為指定的數據格式
        assert dict_file is None or isinstance(dict_file, str)
        assert dict_list is None or isinstance(dict_list, list)
        # 構建index轉到char的list
        self.idx2char = []
        if dict_file is not None:
            # 如果有設定dict_file就會到這裡
            for line_num, line in enumerate(list_from_file(dict_file)):
                line = line.strip('\r\n')
                if len(line) > 1:
                    raise ValueError('Expect each line has 0 or 1 character, '
                                     f'got {len(line)} characters '
                                     f'at line {line_num + 1}')
                if line != '':
                    self.idx2char.append(line)
        elif dict_list is not None:
            # 如果有設定dict_list就會到這裡
            self.idx2char = list(dict_list)
        else:
            # 其他情況，就是只有傳入dict_type部分
            if dict_type in self.dicts:
                # 如果dict_type有在支援的選項當中就會到這裡
                # 獲取在self.dicts當中的資料，self.dicts在最上面有定義好
                self.idx2char = list(self.dicts[dict_type])
            else:
                # 如果沒有在合法的選項當中就會到這裡報錯
                raise NotImplementedError(f'Dict type {dict_type} is not '
                                          'supported')

        # 如果當中有相同文字但是對應到多個index在這裡就會報錯，也就是如果有重複的value就會報錯
        assert len(set(self.idx2char)) == len(self.idx2char), \
            'Invalid dictionary: Has duplicated characters.'

        # 構建反向關係，這裡會是從文字轉成index
        self.char2idx = {char: idx for idx, char in enumerate(self.idx2char)}

    def num_classes(self):
        """Number of output classes."""
        # 已看過，透過idx2char獲取總共有多少分類
        return len(self.idx2char)

    def str2idx(self, strings):
        """Convert strings to indexes.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        """
        # 已看過，將str轉成對應的index
        # strings = 標註訊息，list[str]，list長度會是batch_size，str會是每張圖像對應的標註文字

        # 檢查傳入的strings是否為list格式
        assert isinstance(strings, list)

        # 保存最後結果的地方
        indexes = []
        # 遍歷一個batch的標註資料
        for string in strings:
            if self.lower:
                # 如果有需要將標註資料全部變成小寫就會進行轉換
                string = string.lower()
            # 當前的str對應上的index
            index = []
            # 遍歷一個string當中的個別字母
            for char in string:
                # 透過char2idx的字典獲取對應的index，如果沒有在字典當中的字母就會返回unknown對應上去的index
                char_idx = self.char2idx.get(char, self.unknown_idx)
                if char_idx is None:
                    # 如果char_idx最終的結果是None就會到這裡進行報錯表示有問題
                    raise Exception(f'Chararcter: {char} not in dict,'
                                    f' please check gt_label and use'
                                    f' custom dict file,'
                                    f' or set "with_unknown=True"')
                # 將結果保存
                index.append(char_idx)
            # 將一張圖像的標註資料保存
            indexes.append(index)

        # 返回映射後的結果，list[list[int]]，第一個list長度就會是batch_size，第二個會是對應圖像的文字長度，int就會是對應上的index
        return indexes

    def str2tensor(self, strings):
        """Convert text-string to input tensor.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            tensors (list[torch.Tensor]): [torch.Tensor([1,2,3,3,4]),
                torch.Tensor([5,4,6,3,7])].
        """
        raise NotImplementedError

    def idx2str(self, indexes):
        """Convert indexes to text strings.

        Args:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        Returns:
            strings (list[str]): ['hello', 'world'].
        """
        # 已看過，將index轉成str
        # indexes = 需要轉換的index，list[list]，第一個list長度會是batch_size，第二個list會是該圖像的文字長度

        # 檢查傳入的indexes需要是list型態
        assert isinstance(indexes, list)

        # 保存最後轉換的str的空間
        strings = []
        # 遍歷一整個batch的index
        for index in indexes:
            # 從index轉到char
            string = [self.idx2char[i] for i in index]
            # 添加上去
            strings.append(''.join(string))

        # 最後回傳
        return strings

    def tensor2idx(self, output):
        """Convert model output tensor to character indexes and scores.
        Args:
            output (tensor): The model outputs with size: N * T * C
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                [0.9,0.9,0.98,0.97,0.96]].
        """
        raise NotImplementedError
