# Copyright (c) OpenMMLab. All rights reserved.
import json
import warnings

from mmocr.datasets.builder import PARSERS
from mmocr.utils import StringStrip


@PARSERS.register_module()
class LineStrParser:
    """Parse string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in result dict.
        keys_idx (list[int]): Value index in sub-string list
            for each key above.
        separator (str): Separator to separate string to list of sub-string.
    """

    def __init__(self,
                 keys=['filename', 'text'],
                 keys_idx=[0, 1],
                 separator=' ',
                 **kwargs):
        """ 已看過，將annotation文件當中的每一行轉成dcit格式
        Args:
            keys: 在dict當中的key值
            keys_idx: 上面每個鍵的子字符串列表中的值索引
            separator: 將一個string透過separator分成多個子字串
        """
        # 檢查傳入的資料是否合法
        assert isinstance(keys, list)
        assert isinstance(keys_idx, list)
        assert isinstance(separator, str)
        assert len(keys) > 0
        assert len(keys) == len(keys_idx)
        # 保存傳入資料
        self.keys = keys
        self.keys_idx = keys_idx
        self.separator = separator
        # 構建StringStrip實例對象，這裡會是將字串前後進行調整
        self.strip_cls = StringStrip(**kwargs)

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        line_str = data_ret[map_index]
        line_str = self.strip_cls(line_str)
        if len(line_str.split(' ')) > 2:
            msg = 'More than two blank spaces were detected. '
            msg += 'Please use LineJsonParser to handle '
            msg += 'annotations with blanks. '
            msg += 'Check Doc '
            msg += 'https://mmocr.readthedocs.io/en/latest/'
            msg += 'tutorials/blank_recog.html '
            msg += 'for details.'
            warnings.warn(msg)
        line_str = line_str.split(self.separator)
        if len(line_str) <= max(self.keys_idx):
            raise Exception(
                f'key index: {max(self.keys_idx)} out of range: {line_str}')

        line_info = {}
        for i, key in enumerate(self.keys):
            line_info[key] = line_str[self.keys_idx[i]]
        return line_info


@PARSERS.register_module()
class LineJsonParser:
    """Parse json-string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in both json-string and result dict.
    """

    def __init__(self, keys=[]):
        assert isinstance(keys, list)
        assert len(keys) > 0
        self.keys = keys

    def get_item(self, data_ret, index):
        """ 已看過，獲取指定index的圖像資料
        Args:
            data_ret: LmdbAnnFileBackend的實例對象，會是由lmdb的標註文件所組成的
            index: 指定的圖像index
        """
        # 這裡的index是隨機選取的透過取mod後可以找到在data_ret當中的哪個index
        map_index = index % len(data_ret)
        # 獲取該index的json字串資料，這裡會是dict的樣式不過型態是str
        json_str = data_ret[map_index]
        # 將樣式為dict的str變成dict
        line_json_obj = json.loads(json_str)
        # 最終回傳的結果
        line_info = {}
        # 這裡會遍歷我們指定需要提取出來的key
        for key in self.keys:
            if key not in line_json_obj:
                # 如果line_json_obj當中沒有找到指定的key就會報錯
                raise Exception(f'key {key} not in line json {line_json_obj}')
            # 將需要提取出的資料保存
            line_info[key] = line_json_obj[key]

        # 最後回傳
        return line_info
