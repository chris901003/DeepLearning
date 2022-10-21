import os
import json
import sys
import pickle
from importlib import import_module
import numpy as np


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
