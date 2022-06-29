import os
import yaml
# 也是一種字典格式，只是有更簡單的操作，可以用簡單的操作新增資料
from easydict import EasyDict as edict


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """
    def __init__(self, cfg_dict=None, config_file=None):
        """
        :param cfg_dict: 預設是None
        :param config_file: 預設是None
        """
        # 已看過
        if cfg_dict is None:
            cfg_dict = {}

        # 如果初始化時就有傳入檔案就可以直接將檔案的資料放入到字典當中
        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read()))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        # 已看過
        with open(config_file, 'r') as fo:
            # self.update(yaml.load(fo.read()))
            # 將yaml裡面的資料放入到字典當中
            self.update(yaml.load(fo.read(), Loader=yaml.FullLoader))
    
    def merge_from_dict(self, config_dict):
        # 已看過
        # 傳入的是已經變成字典的資料，可以直接將兩個字典做融合
        self.update(config_dict)


def get_config(config_file=None):
    # 已看過
    # 從deepsortor.py實例化，且傳入的config_file=None
    return YamlParser(config_file=config_file)


if __name__ == "__main__":
    # 已看過
    # 這裡都只做測試用
    cfg = YamlParser(config_file="../configs/yolov3.yaml")
    cfg.merge_from_file("../configs/deep_sort.yaml")
