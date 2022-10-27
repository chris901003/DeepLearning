import os
import json
import sys
import pickle
import logging
from importlib import import_module
import numpy as np
from SpecialTopic.ST.utils import get_cls_from_dict


def parser_cfg(file_path):
    # 目前支援讀取檔案類型，如果有缺少的可以再繼續新增
    # [json, txt, pickle, py, npy]
    assert os.path.exists(file_path), f'指定的檔案 {file_path} 不存在無法讀取'
    if os.path.splitext(file_path)[1] == '.json':
        with open(file_path, 'r') as f:
            results = json.load(f)
    elif os.path.splitext(file_path)[1] == '.txt':
        with open(file_path, 'r') as f:
            results = f.readlines()
    elif os.path.splitext(file_path)[1] == '.pickle':
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
    elif os.path.splitext(file_path)[1] == '.py':
        base_path = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        sys.path.insert(0, base_path)
        results = import_module(file_name)
        sys.path.pop(0)
    elif os.path.splitext(file_path)[1] == '.npy':
        results = np.load(file_path)
    else:
        raise NotImplementedError('目前不支援該檔案類型')
    return results


def list_of_list(datas):
    return any(isinstance(data, list) for data in datas)


def create_logger(log_config, log_name=None):
    support_logger_level = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    support_handler = {
        'FileHandler': logging.FileHandler,
        'StreamHandler': logging.StreamHandler
    }
    log_link = log_config.get('log_link', None)
    assert log_link is not None, '需要提供log link到的模塊對象名稱'
    if log_name is None:
        log_name = log_link
    else:
        log_name = log_name + '.' + log_link
    logger = logging.getLogger(log_name)
    log_level = log_config.get('level', None)
    if log_level is not None:
        log_level = support_logger_level.get(log_level, None)
        assert log_level is not None, '尚未提供該設定level值'
        logger.setLevel(log_level)
    format_cfg = log_config.get('format', None)
    if format_cfg is not None:
        format_cfg = [f'%({format_name})s' for format_name in format_cfg]
        format_info = ' - '.join(format_cfg)
        formatter = logging.Formatter(format_info)
    else:
        formatter = None
    handler = log_config.get('handler', None)
    if handler is not None:
        for handler_info in handler:
            handler_name = handler_info.get('type', None)
            handler_cls = get_cls_from_dict(support_handler, handler_info)
            init_param = dict()
            if handler_name == 'FileHandler':
                filename = handler_info.get('save_path', None)
                assert filename is not None, '需提供保存log的路徑'
                init_param['filename'] = filename
                init_param['mode'] = 'w'
            handler_obj = handler_cls(**init_param)
            format_cfg = handler_info.get('format', None)
            if format_cfg is not None:
                format_cfg = [f'%({format_name})s' for format_name in format_cfg]
                formatter_custom = ' - '.join(format_cfg)
                handler_obj.setFormatter(formatter_custom)
            elif formatter is not None:
                handler_obj.setFormatter(formatter)
            level_custom = handler_info.get('level', None)
            if level_custom is not None:
                level_custom = support_logger_level.get(level_custom, None)
                assert level_custom is not None
                handler_obj.setLevel(level_custom)
            logger.addHandler(handler_obj)
    sub_log_cfg = log_config.get('sub_log', None)
    sub_log = dict()
    if sub_log_cfg is not None:
        for sub_log_info in sub_log_cfg:
            log_link_name = sub_log_info.get('log_link', None)
            assert log_link_name is not None, '需要提供log link到的模塊對象名稱'
            sub_log_obj = create_logger(sub_log_info, log_name=log_name)
            sub_log[log_link_name] = sub_log_obj
    log_data = dict(logger=logger, sub_log=sub_log)
    return log_data
