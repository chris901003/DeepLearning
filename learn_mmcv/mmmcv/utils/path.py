import os
import os.path as osp
from pathlib import Path

from .misc import is_str


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    """
    :param filename: 檔案路徑位置
    :param msg_tmpl: 如果檔案不存在要報錯的訊息
    :return:
    """
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))
