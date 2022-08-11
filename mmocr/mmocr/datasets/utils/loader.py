# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmocr.datasets.builder import LOADERS, build_parser
from .backend import (HardDiskAnnFileBackend, HTTPAnnFileBackend,
                      PetrelAnnFileBackend)


@LOADERS.register_module()
class AnnFileLoader:
    """Annotation file loader to load annotations from ann_file, and parse raw
    annotation to dict format with certain parser.

    Args:
        ann_file (str): Annotation file path.
        parser (dict): Dictionary to construct parser
            to parse original annotation infos.
        repeat (int|float): Repeated times of dataset.
        file_storage_backend (str): The storage backend type for annotation
            file. Options are "disk", "http" and "petrel". Default: "disk".
        file_format (str): The format of annotation file. Options are
            "txt" and "lmdb". Default: "txt".
    """

    _backends = {
        'disk': HardDiskAnnFileBackend,
        'petrel': PetrelAnnFileBackend,
        'http': HTTPAnnFileBackend
    }

    def __init__(self,
                 ann_file,
                 parser,
                 repeat=1,
                 file_storage_backend='disk',
                 file_format='txt',
                 **kwargs):
        """ 已看過，從標註檔案當中讀取標註資料並且將讀入的資料轉成dict格式儲存
        Args:
            ann_file: 標註資料檔案位置
            parser: 解析器相關參數
            repeat: dataset重複次數
            file_storage_backend: 標註資料檔案的存放設備
            file_format: 標註檔案的副檔名
        """
        # 檢查傳入的資料是否符合規定
        assert isinstance(ann_file, str)
        assert isinstance(repeat, (int, float))
        assert isinstance(parser, dict)
        assert repeat > 0
        assert file_storage_backend in ['disk', 'http', 'petrel']
        assert file_format in ['txt', 'lmdb']

        if file_format == 'lmdb' and parser['type'] == 'LineStrParser':
            raise ValueError('We only support using LineJsonParser '
                             'to parse lmdb file. Please use LineJsonParser '
                             'in the dataset config')
        # 構建解析器實例對象
        self.parser = build_parser(parser)
        # 保存repeat次數
        self.repeat = repeat
        # 構建backend實例對象
        self.ann_file_backend = self._backends[file_storage_backend](
            file_format, **kwargs)
        # 讀取ann_file當中資料
        self.ori_data_infos = self._load(ann_file)

    def __len__(self):
        return int(len(self.ori_data_infos) * self.repeat)

    def _load(self, ann_file):
        """Load annotation file."""
        # 已看過，讀取標註檔案資訊，這裡會根據標註檔案存放的設備呼叫不同的方式讀取

        return self.ann_file_backend(ann_file)

    def __getitem__(self, index):
        """Retrieve anno info of one instance with dict format."""
        # 已看過，會傳入需要獲取圖像資料的指定index
        return self.parser.get_item(self.ori_data_infos, index)

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < len(self):
            data = self[self._n]
            self._n += 1
            return data
        raise StopIteration

    def close(self):
        """For ann_file with lmdb format only."""
        self.ori_data_infos.close()


@LOADERS.register_module()
class HardDiskLoader(AnnFileLoader):
    """Load txt format annotation file from hard disks."""

    def __init__(self, ann_file, parser, repeat=1):
        warnings.warn(
            'HardDiskLoader is deprecated, please use '
            'AnnFileLoader instead.', UserWarning)
        super().__init__(
            ann_file,
            parser,
            repeat,
            file_storage_backend='disk',
            file_format='txt')


@LOADERS.register_module()
class LmdbLoader(AnnFileLoader):
    """Load lmdb format annotation file from hard disks."""

    def __init__(self, ann_file, parser, repeat=1):
        warnings.warn(
            'LmdbLoader is deprecated, please use '
            'AnnFileLoader instead.', UserWarning)
        super().__init__(
            ann_file,
            parser,
            repeat,
            file_storage_backend='disk',
            file_format='lmdb')
