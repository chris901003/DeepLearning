# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class VideoDataset(BaseDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 0.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
        """ 已看過，構建由影片組成的Dataset
        Args:
            ann_file: 標註文件的檔案位置
            pipeline: 影片處理的流水線
            start_index: 指定開始幀的index
            kwargs: 其他參數，例如路徑的root部分
        """
        # 繼承自BaseDataset，將繼承對象進行初始化
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)

    def load_annotations(self):
        """Load annotation file to get video information."""
        # 已看過，讀取標註文件資料來獲取每個影片的詳細資訊
        if self.ann_file.endswith('.json'):
            # 如果標註文件是json格式就會到這裡，使用json的方式讀取
            return self.load_json_annotations()

        # 最終回傳的list
        video_infos = []
        # 進行讀檔
        with open(self.ann_file, 'r') as fin:
            # 依照一行一行進行讀取
            for line in fin:
                # 透過strip將頭尾的空行以及換行符去除，之後透過split將每個空格分開
                line_split = line.strip().split()
                if self.multi_class:
                    # 如果一個影片有多個標註訊息就會到這裡
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    # 獲取影片名稱以及標註訊息
                    filename, label = line_split
                    # 將label從str變成int格式
                    label = int(label)
                if self.data_prefix is not None:
                    # 如果有給檔案路徑前綴部分就會添加上去
                    filename = osp.join(self.data_prefix, filename)
                # 將結果添加到video_infos當中，且是用dict格式
                video_infos.append(dict(filename=filename, label=label))
        # 最終將結果回傳
        return video_infos
