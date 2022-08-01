# Copyright (c) OpenMMLab. All rights reserved.


def is_3dlist(x):
    """check x is 3d-list([[[1], []]]) or 2d empty list([[], []]) or 1d empty
    list([]).

    Notice:
        The reason that it contains 1d or 2d empty list is because
        some arguments from gt annotation file or model prediction
        may be empty, but usually, it should be 3d-list.
    """
    if not isinstance(x, list):
        return False
    if len(x) == 0:
        return True
    for sub_x in x:
        if not is_2dlist(sub_x):
            return False

    return True


def is_2dlist(x):
    """check x is 2d-list([[1], []]) or 1d empty list([]).

    Notice:
        The reason that it contains 1d empty list is because
        some arguments from gt annotation file or model prediction
        may be empty, but usually, it should be 2d-list.
    """
    # 已看過，檢查是否為list[list]格式
    if not isinstance(x, list):
        # 如果一開始就不是list就會回傳False
        return False
    if len(x) == 0:
        # 如果當中為空就回傳True
        return True

    # 檢查第一層list當中資料是否全為list，如果是就會回傳True，否則就會是False
    return all(isinstance(item, list) for item in x)


def is_type_list(x, type):

    if not isinstance(x, list):
        return False

    return all(isinstance(item, type) for item in x)


def is_none_or_type(x, type):

    return isinstance(x, type) or x is None


def equal_len(*argv):
    assert len(argv) > 0

    num_arg = len(argv[0])
    for arg in argv:
        if len(arg) != num_arg:
            return False
    return True


def valid_boundary(x, with_score=True):
    # 已看過，檢查boundary資訊是否正確

    # 獲取長度
    num = len(x)
    if num < 8:
        # 如果長度小於8就是不合法
        return False
    if num % 2 == 0 and (not with_score):
        # 如果剛好是偶數with_score就會需要是False，因為這樣才會剛好湊齊(x, y)
        return True
    if num % 2 == 1 and with_score:
        # 如果是奇數with_score就會需要是True，這樣扣除置信度就會是偶數
        return True

    # 其他就是False
    return False
