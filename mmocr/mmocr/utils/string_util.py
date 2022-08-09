# Copyright (c) OpenMMLab. All rights reserved.
class StringStrip:
    """Removing the leading and/or the trailing characters based on the string
    argument passed.

    Args:
        strip (bool): Whether remove characters from both left and right of
            the string. Default: True.
        strip_pos (str): Which position for removing, can be one of
            ('both', 'left', 'right'), Default: 'both'.
        strip_str (str|None): A string specifying the set of characters
            to be removed from the left and right part of the string.
            If None, all leading and trailing whitespaces
            are removed from the string. Default: None.
    """

    def __init__(self, strip=True, strip_pos='both', strip_str=None):
        """ 已看過，將字串的前導以及後綴部分進行刪除
        Args:
            strip: 是否移除字串左右部分多餘的空白
            strip_pos: 指定左邊或是右邊移除
            strip_str: 指定刪除左邊以及右邊的指定字串
        """
        # 檢查傳入資料是否合法
        assert isinstance(strip, bool)
        assert strip_pos in ('both', 'left', 'right')
        assert strip_str is None or isinstance(strip_str, str)

        # 將傳入資料進行保存
        self.strip = strip
        self.strip_pos = strip_pos
        self.strip_str = strip_str

    def __call__(self, in_str):

        if not self.strip:
            return in_str

        if self.strip_pos == 'left':
            return in_str.lstrip(self.strip_str)
        elif self.strip_pos == 'right':
            return in_str.rstrip(self.strip_str)
        else:
            return in_str.strip(self.strip_str)
