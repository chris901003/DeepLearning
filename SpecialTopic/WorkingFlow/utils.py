import os
import json


def parser_cfg(json_file_path):
    assert os.path.exists(json_file_path), f'指定的config文件 {json_file_path} 不存在'
    with open(json_file_path) as f:
        config = json.load(f)
    return config


def list_of_list(datas):
    return any(isinstance(data, list) for data in datas)
