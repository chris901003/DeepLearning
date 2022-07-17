# Copyright (c) OpenMMLab. All rights reserved.
import ast
import copy
import os
import os.path as osp
import platform
import shutil
import sys
import tempfile
import types
import uuid
import warnings
from argparse import Action, ArgumentParser
from collections import abc
from importlib import import_module
from pathlib import Path

from addict import Dict
from yapf.yapflib.yapf_api import FormatCode

from .misc import import_modules_from_strings
from .path import check_file_exist

if platform.system() == 'Windows':
    import regex as re  # type: ignore
else:
    import re  # type: ignore

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
DEPRECATION_KEY = '_deprecation_'
RESERVED_KEYS = ['filename', 'text', 'pretty_text']


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


def add_args(parser, cfg, prefix=''):
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument('--' + prefix + k)
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, action='store_true')
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + '.')
        elif isinstance(v, abc.Iterable):
            parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+')
        else:
            print(f'cannot parse key {prefix + k} of type {type(v)}')
    return parser


class Config:
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    @staticmethod
    def _validate_py_syntax(filename):
        """
        :param filename: 臨時文件地址
        """
        # 已看過

        # 讀取檔案
        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            content = f.read()
        try:
            # 透過ast檢查該檔案內容是否符合python格式
            ast.parse(content)
        except SyntaxError as e:
            # 當該檔案不符合python格式就會報錯
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    @staticmethod
    def _substitute_predefined_vars(filename, temp_config_name):
        """
        :param filename: config檔案位置
        :param temp_config_name: 臨時文件位置
        """
        # 已看過

        # file_dirname = config文件的資料夾位置
        file_dirname = osp.dirname(filename)
        # file_basename = config檔案名稱
        file_basename = osp.basename(filename)
        # file_basename_no_extension = config檔案名稱，這裡不會有副檔名
        file_basename_no_extension = osp.splitext(file_basename)[0]
        # file_extname = config檔案的副檔名
        file_extname = osp.splitext(filename)[1]

        # 將以上內容全部放到support_templates當中，用dict型態管理
        support_templates = dict(
            fileDirname=file_dirname,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        # 開啟config文件
        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            # config_file = config文件當中內容
            config_file = f.read()
        # 遍歷支援的{{}}替代
        for key, value in support_templates.items():
            # regexp = 正則表達式，等等會使用這個當作過濾器，可以過濾出我們要的位置進行替換
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            # 這裡為了要更好的兼容windows系統，所以會將windows中的\\轉換成/
            value = value.replace('\\', '/')
            # 透過re這個模組將config_file當中符合當前regexp的位置變成value
            # 也就是說如果在撰寫config時可以有更多的彈性，透過{{}}可以避免在不同資料夾歸類方式下會造成錯誤
            config_file = re.sub(regexp, value, config_file)
        # 將最終結果寫入到臨時文件當中
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _pre_substitute_base_vars(filename, temp_config_name):
        """
        :param filename: 臨時文件位置
        :param temp_config_name: 臨時文件位置
        :param return: 隨機生成的字串與原字串的映射關係
        """
        # 已看過

        # 將文件當中value為_base_.xxx部分進行轉換，這個會出現在繼承base.py的變數的config文件當中
        """Substitute base variable placehoders to string, so that parsing
        would work."""
        # 開啟臨時文件
        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            config_file = f.read()
        # base_var_dict = 最終會回傳的內容，存放映射關係
        base_var_dict = {}
        # BASE_KEY = '_base_'
        # regexp = 正則表達式，主要是用來過濾_base_用的
        regexp = r'\{\{\s*' + BASE_KEY + r'\.([\w\.]+)\s*\}\}'
        # base_vars = 將config_file當中符合regexp的部分拿出來，並且會是一個list
        # 將config_file當中value是_base_.xxx的xxx部分拿出來
        base_vars = set(re.findall(regexp, config_file))
        # 遍歷base_vars
        for base_var in base_vars:
            # 給定一個隨機字串
            randstr = f'_{base_var}_{uuid.uuid4().hex.lower()[:6]}'
            # 將隨機生成的字串與原始字串進行映射，並且存在base_var_dict當中
            base_var_dict[randstr] = base_var
            # 根據當前遍歷到的base_var構建正則表達
            regexp = r'\{\{\s*' + BASE_KEY + r'\.' + base_var + r'\s*\}\}'
            # 將config_file當中符合表達的部份轉換成剛剛隨機生成的字串
            config_file = re.sub(regexp, f'"{randstr}"', config_file)
        # 最終將轉換完成的結果寫入到臨時文件當中
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)
        # 透過函數在臨時文件當中不會出現value中有_base_.xxx的內容
        # return = 隨機生成的字串與原字串的映射關係
        return base_var_dict

    @staticmethod
    def _substitute_base_vars(cfg, base_var_dict, base_cfg):
        """
        :param cfg: config文件內容
        :param base_var_dict: 繼承config文件內容隨機字串與原始字串的映射關係
        :param base_cfg: 繼承config文件內容
        :param return: 將最後替換完成的config文件返回
        """
        """Substitute variable strings to their actual values."""
        # 已看過

        # 深度拷貝一份config文件
        cfg = copy.deepcopy(cfg)

        if isinstance(cfg, dict):
            # 正常來說透過python格式的config文件會是走這裡
            # 遍歷所有config文件內的內容
            for k, v in cfg.items():
                if isinstance(v, str) and v in base_var_dict:
                    # 當value是str且有在隨機字串當中
                    # 通常都是用這裡
                    new_v = base_cfg
                    # base_var_dict[v] = _base_.xxx
                    for new_k in base_var_dict[v].split('.'):
                        # 第一次會先定位到繼承自哪個config文件，第二次就會是找到對應的名稱
                        new_v = new_v[new_k]
                    # 最後更新到cfg當中
                    cfg[k] = new_v
                elif isinstance(v, (list, tuple, dict)):
                    # 這裡就透過遞迴調用
                    cfg[k] = Config._substitute_base_vars(
                        v, base_var_dict, base_cfg)
        elif isinstance(cfg, tuple):
            # 這部分比較少會用到，這裡就先不去細看
            cfg = tuple(
                Config._substitute_base_vars(c, base_var_dict, base_cfg)
                for c in cfg)
        elif isinstance(cfg, list):
            # 這部分比較少會用到，這裡就先不去細看
            cfg = [
                Config._substitute_base_vars(c, base_var_dict, base_cfg)
                for c in cfg
            ]
        elif isinstance(cfg, str) and cfg in base_var_dict:
            # 這部分比較少會用到，這裡就先不去細看
            new_v = base_cfg
            for new_k in base_var_dict[cfg].split('.'):
                new_v = new_v[new_k]
            cfg = new_v

        # return = 將最後替換好的config文件進行返回
        return cfg

    @staticmethod
    def _file2dict(filename, use_predefined_variables=True):
        """
        :param filename: config檔案名稱
        :param use_predefined_variables: 是否將config當中的{{}}進行處理
        """
        # 已看過

        # 將傳入的config檔案名稱轉換成絕對路徑，這樣就可以透過絕對路徑進行讀取檔案
        filename = osp.abspath(osp.expanduser(filename))
        # 檢查該config文件是否存在
        check_file_exist(filename)
        # 取出config文件的副檔名
        fileExtname = osp.splitext(filename)[1]
        # 這裡mmsegmentation可以支持多種文檔進行配置，當檔案類型都不在以下幾種就會報錯
        if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
            raise OSError('Only py/yml/yaml/json type are supported now!')

        # 透過tempfile構建臨時資料夾，這種臨時資料夾會在關閉時同時被刪除
        # 這裡使用with as方式可以確保最終臨時資料夾會被關閉且刪除
        # 這裡會需要使用到臨時文件是因為在原始文件當中會有{{}}，而我們會需要將其中進行轉換同時也不能去更改原始內容
        # 另一個原因是可以減少緩存的使用，雖然好像都不構成一定要這麼做的理由
        with tempfile.TemporaryDirectory() as temp_config_dir:
            # temp_config_file = 臨時文件的文件實例對象
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)
            if platform.system() == 'Windows':
                temp_config_file.close()
            # temp_config_name = 臨時文件的檔案名稱
            temp_config_name = osp.basename(temp_config_file.name)
            if use_predefined_variables:
                # 將config文件當中{{}}部分進行替換
                # 這裡只會換[fileDirname, fileBasename, fileBasenameNoExtension, fileExtname]，其他的保持不變
                # 這裡預設會是True
                Config._substitute_predefined_vars(filename,
                                                   temp_config_file.name)
            else:
                # 如果確定config當中沒有需要進行轉換的{{}}就可以將use_predefined_variables設定為False
                # 這裡就直接拷貝一份一樣的config文件到臨時文件當中
                shutil.copyfile(filename, temp_config_file.name)

            # 將文件當中value為_base_.xxx部分進行轉換，會將原先_base_.xxx轉換成一個隨機字串，同時對應關係存在base_var_dict當中
            # 官方描述為從基類引入變量
            # base_var_dict = 隨機生成的字串與原字串的映射關係
            base_var_dict = Config._pre_substitute_base_vars(
                temp_config_file.name, temp_config_file.name)

            if filename.endswith('.py'):
                # 當config文件為python格式時，會走這裡進去
                # temp_module_name = 臨時文件檔案名稱，這裡會去除副檔名
                temp_module_name = osp.splitext(temp_config_name)[0]
                # 將臨時文件放到系統路徑當中，這樣之後才可以動態import進來
                sys.path.insert(0, temp_config_dir)
                # 檢查臨時文件是否符合python格式
                Config._validate_py_syntax(filename)
                # 透過import_module動態將臨時文件import進來
                mod = import_module(temp_module_name)
                # 將剛剛加入到系統路徑的檔案彈出
                sys.path.pop(0)
                # cfg_dict = dict格式，當config裡有a=b那麼在mod當中就會被記錄成key=a且value=b的dict格式
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                    and not isinstance(value, types.ModuleType)
                    and not isinstance(value, types.FunctionType)
                }
                # 將剛才import近來的臨時文件取消import
                del sys.modules[temp_module_name]
            elif filename.endswith(('.yml', '.yaml', '.json')):
                # 當config文件是[yml, yaml, json]格式就會走這裡(這裡我們先不去看，因為大多數都還是用python格式導入)
                import mmcv
                cfg_dict = mmcv.load(temp_config_file.name)
            # 將臨時文件關閉，透過手動關閉系統會自動將臨時文件刪除
            temp_config_file.close()

        # check deprecation information
        # deprecated = 已棄用，表是這個config文件已經被棄用或是即將要被棄用，最好是替換其他方法
        if DEPRECATION_KEY in cfg_dict:
            deprecation_info = cfg_dict.pop(DEPRECATION_KEY)
            # 警告該config在未來會被棄用
            warning_msg = f'The config file {filename} will be deprecated ' \
                'in the future.'
            # 提示可以選擇的替換config
            if 'expected' in deprecation_info:
                warning_msg += f' Please use {deprecation_info["expected"]} ' \
                    'instead.'
            # 可以參考哪些網站進行替換
            if 'reference' in deprecation_info:
                warning_msg += ' More information can be found at ' \
                    f'{deprecation_info["reference"]}'
            # 顯示警告標語
            warnings.warn(warning_msg, DeprecationWarning)

        # cfg_text = 前半部分是config檔案路徑，後半部分是config文件的配置內容，前後部分會用換行隔開，每個設定也會用換行隔開
        cfg_text = filename + '\n'
        # 讀取config文件，並且將內容寫入到cfg_text當中
        with open(filename, encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            # 當config文件當中有_base_關鍵字表示要繼承其他config文件，在這裡會處理這種情況
            # cfg_dir = config文件所處資料夾位置
            cfg_dir = osp.dirname(filename)
            # 將cfg_dict當中key為_base_部分的value拿出來，同時會將cfg_dict的_base_去除
            base_filename = cfg_dict.pop(BASE_KEY)
            # 因為如果只有繼承一個config文件會是string格式，所以這裡會轉換成list[str]格式
            # 裡面都是config文件路徑
            # base_filename = ['./config_a.py', './config_b.py', ..., './config_xx.py']
            base_filename = base_filename if isinstance(
                base_filename, list) else [base_filename]

            # 創建兩個空間，一個是要放dict另一個是str的
            cfg_dict_list = list()
            cfg_text_list = list()
            # 遍歷所有要繼承的config文件
            for f in base_filename:
                # 這裡透過遞歸方式去解析要繼承的config文件，傳入的是config文件絕對位置
                # _cfg_dict = config文件的dict格式
                # _cfg_text = Config文件的string格式
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f))
                # 將結果放入到紀錄當中
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            # 繼承的config會先在base_cfg_dict混合，之後才會跟當前config混合
            base_cfg_dict = dict()
            # 遍歷繼承的config文件內容
            for c in cfg_dict_list:
                # 檢查在繼承config文件當中有沒有重複的key
                duplicate_keys = base_cfg_dict.keys() & c.keys()
                # 當繼承的config文件當中有重複的key時這裡就會報錯
                if len(duplicate_keys) > 0:
                    raise KeyError('Duplicate key is not allowed among bases. '
                                   f'Duplicate keys: {duplicate_keys}')
                # 更新繼承config文件當中的dict
                base_cfg_dict.update(c)
            # base_cfg_dict = 所有繼承的config文件的配置內容

            # Substitute base variables from strings to their actual values
            # 處理config文件當中value有_base_.xxx的部分
            # 將原始config內容以及繼承config文件內容以及隨機字串與原字串映射關係傳入
            # cfg_dict = 替換完成的config dict格式
            cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict,
                                                    base_cfg_dict)

            # 將當前的config文件配置與繼承的config文件配置進行融合，這裡如果遇到相同的key會優先使用當前的config內容
            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            # 將最後的結果給cfg_dict
            cfg_dict = base_cfg_dict

            # merge cfg_text
            # 將最後的結果也要保存到cfg_text當中
            cfg_text_list.append(cfg_text)
            cfg_text = '\n'.join(cfg_text_list)

        # cfg_dict = config文件的dict格式
        # cfg_text = config文件全部轉成str格式
        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b, allow_list_keys=False):
        """
        :param a: 當前config文件配置內容
        :param b: 繼承config文件配置內容
        :param allow_list_keys:
        :return:
        """
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Default: False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]
        """
        # 已看過

        # 深拷貝一份b
        b = b.copy()
        # 遍歷a所有的內容
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                # 這裡預設是不會進來的
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f'Index {k} exceeds the length of list {b}')
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict):
                # 檢查b裡面是否有相同的key且在a沒有指定說要拋棄繼承該key的value
                if k in b and not v.pop(DELETE_KEY, False):
                    # 這裡默認只會支援dict格式
                    allowed_types = (dict, list) if allow_list_keys else dict
                    if not isinstance(b[k], allowed_types):
                        # 當不是dict格式就會直接報錯
                        raise TypeError(
                            f'{k}={v} in child config cannot inherit from '
                            f'base because {k} is a dict in the child config '
                            f'but is of type {type(b[k])} in base config. '
                            f'You may set `{DELETE_KEY}=True` to ignore the '
                            f'base config.')
                    # 通過遞迴調用繼續合併
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    # 如果b中沒有該key或是a要直接拋棄繼承中的key的value就會直接到這裡
                    b[k] = ConfigDict(v)
            else:
                # 直接給到b
                b[k] = v
        # 回傳合併好的config_dict
        return b

    @staticmethod
    def fromfile(filename,
                 use_predefined_variables=True,
                 import_custom_modules=True):
        """
        :param filename: config檔案名稱
        :param use_predefined_variables: 預設為True，這個是用來在解析config文件時可以將{{}}轉換成正確的資料
        :param import_custom_modules: 預設為True，目前暫時不確定是什麼
        :return:
        """
        # 已看過

        # 如果filename是Path格式就將filename轉換成string格式
        if isinstance(filename, Path):
            filename = str(filename)
        # cfg_dict = config文件當中的配置，這裡會是dict格式
        # cfg_text = 前半部分會是config檔案位置，後半部分就會是config內容，中間都會以換行符號進行分隔
        cfg_dict, cfg_text = Config._file2dict(filename,
                                               use_predefined_variables)
        # 檢查cfg_dict當中有沒有custom_imports的key值，這裡通常都不會有，所以我們先跳過
        if import_custom_modules and cfg_dict.get('custom_imports', None):
            import_modules_from_strings(**cfg_dict['custom_imports'])
        # 最後實例化Config實例對象
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    @staticmethod
    def fromstring(cfg_str, file_format):
        """Generate config from config str.

        Args:
            cfg_str (str): Config str.
            file_format (str): Config file format corresponding to the
               config str. Only py/yml/yaml/json type are supported now!

        Returns:
            :obj:`Config`: Config obj.
        """
        if file_format not in ['.py', '.json', '.yaml', '.yml']:
            raise OSError('Only py/yml/yaml/json type are supported now!')
        if file_format != '.py' and 'dict(' in cfg_str:
            # check if users specify a wrong suffix for python
            warnings.warn(
                'Please check "file_format", the file format may be .py')
        with tempfile.NamedTemporaryFile(
                'w', encoding='utf-8', suffix=file_format,
                delete=False) as temp_file:
            temp_file.write(cfg_str)
            # on windows, previous implementation cause error
            # see PR 1077 for details
        cfg = Config.fromfile(temp_file.name)
        os.remove(temp_file.name)
        return cfg

    @staticmethod
    def auto_argparser(description=None):
        """Generate argparser from config file automatically (experimental)"""
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument('config', help='config file path')
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument('config', help='config file path')
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        """
        :param cfg_dict: config文件的dict格式
        :param cfg_text: config文件的str格式
        :param filename: config文件位置
        """
        # 已看過

        # 如果沒有傳入cfg_dict就創建一個空的字典，通常不會沒有傳入
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            # 有傳入的話也必須要是dict格式
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        # 遍歷cfg_dict裡所有的key
        for key in cfg_dict:
            # key不可以是保留字
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        if isinstance(filename, Path):
            # filename轉換成str格式
            filename = str(filename)

        # 將cfg_dict從python原生的dict變成更好用的add_dict
        super().__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        # 保存filename
        super().__setattr__('_filename', filename)
        # 保存cfg_text
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename) as f:
                text = f.read()
        else:
            text = ''
        super().__setattr__('_text', text)

    @property
    def filename(self):
        # 回傳filename
        return self._filename

    @property
    def text(self):
        # 回傳config內容的str型態
        return self._text

    @property
    def pretty_text(self):

        # 回傳config內容但是有經過排版
        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),'
                    for v_ in v).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                attr_str = _indent(attr_str, indent) + ']'
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style='pep8',
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True)
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    def __repr__(self):
        return f'Config (path: {self.filename}): {self._cfg_dict.__repr__()}'

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self):
        return (self._cfg_dict, self._filename, self._text)

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)

        return other

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __setstate__(self, state):
        _cfg_dict, _filename, _text = state
        super().__setattr__('_cfg_dict', _cfg_dict)
        super().__setattr__('_filename', _filename)
        super().__setattr__('_text', _text)

    def dump(self, file=None):
        """
        :param file: work_dir加上config文件名稱，所形成的檔案路徑
        """

        """Dumps config into a file or returns a string representation of the
        config.

        If a file argument is given, saves the config to that file using the
        format defined by the file argument extension.

        Otherwise, returns a string representing the config. The formatting of
        this returned string is defined by the extension of `self.filename`. If
        `self.filename` is not defined, returns a string representation of a
         dict (lowercased and using ' for strings).

        Examples:
            >>> cfg_dict = dict(item1=[1, 2], item2=dict(a=0),
            ...     item3=True, item4='test')
            >>> cfg = Config(cfg_dict=cfg_dict)
            >>> dump_file = "a.py"
            >>> cfg.dump(dump_file)

        Args:
            file (str, optional): Path of the output file where the config
                will be dumped. Defaults to None.
        """
        # 已看過
        import mmcv
        # cfg_dict = config的dict格式，也就是訓練的所有參數的dict格式
        cfg_dict = super().__getattribute__('_cfg_dict').to_dict()
        if file is None:
            if self.filename is None or self.filename.endswith('.py'):
                return self.pretty_text
            else:
                file_format = self.filename.split('.')[-1]
                return mmcv.dump(cfg_dict, file_format=file_format)
        elif file.endswith('.py'):
            # 預設會是走這裡
            # 會將config文件的所有配置內容寫入到log當中
            with open(file, 'w', encoding='utf-8') as f:
                f.write(self.pretty_text)
        else:
            file_format = file.split('.')[-1]
            return mmcv.dump(cfg_dict, file=file, file_format=file_format)

    def merge_from_dict(self, options, allow_list_keys=True):
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))

            >>> # Merge list element
            >>> cfg = Config(dict(pipeline=[
            ...     dict(type='LoadImage'), dict(type='LoadAnnotations')]))
            >>> options = dict(pipeline={'0': dict(type='SelfLoadImage')})
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(pipeline=[
            ...     dict(type='SelfLoadImage'), dict(type='LoadAnnotations')])

        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in ``options`` and will replace the element of the
              corresponding index in the config if the config is a list.
              Default: True.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super().__getattribute__('_cfg_dict')
        super().__setattr__(
            '_cfg_dict',
            Config._merge_a_into_b(
                option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys))


class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val == 'None':
            return None
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple: The expanded list or tuple from the string.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)
