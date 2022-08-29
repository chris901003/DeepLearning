# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np

from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class PoseDataset(BaseDataset):
    """Pose dataset for action recognition.

    The dataset loads pose and apply specified transforms to return a
    dict containing pose information.

    The ann_file is a pickle file, the json file contains a list of
    annotations, the fields of an annotation include frame_dir(video_id),
    total_frames, label, kp, kpscore.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split used. Only applicable to UCF or
            HMDB. Allowed choiced are 'train1', 'test1', 'train2', 'test2',
            'train3', 'test3'. Default: None.
        valid_ratio (float | None): The valid_ratio for videos in KineticsPose.
            For a video with n frames, it is a valid training sample only if
            n * valid_ratio frames have human pose. None means not applicable
            (only applicable to Kinetics Pose). Default: None.
        box_thr (str | None): The threshold for human proposals. Only boxes
            with confidence score larger than `box_thr` is kept. None means
            not applicable (only applicable to Kinetics Pose [ours]). Allowed
            choices are '0.5', '0.6', '0.7', '0.8', '0.9'. Default: None.
        class_prob (dict | None): The per class sampling probability. If not
            None, it will override the class_prob calculated in
            BaseDataset.__init__(). Default: None.
        **kwargs: Keyword arguments for ``BaseDataset``.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 split=None,
                 valid_ratio=None,
                 box_thr=None,
                 class_prob=None,
                 **kwargs):
        """ 透過關節點判斷行為的資料集
        Args:
            ann_file: 標註文件檔案資料路徑
            pipeline: 資料處理流
            split: 是否有使用數據集拆分
            valid_ratio: 一段影片當中需要有多少比例的圖像有檢測到有效關節點
            box_thr: 人物檢測的閾值，要超過閾值才會認定為人物
            class_prob: 每個類別的抽樣概率
        """
        # 將modality設定成Pose
        modality = 'Pose'
        # split, applicable to ucf or hmdb
        # 保存split資訊
        self.split = split

        # 繼承自BaseDataset，將繼承對象初始化
        super().__init__(ann_file, pipeline, start_index=0, modality=modality, **kwargs)

        # box_thr, which should be a string
        # 將人物匡選置信度保存，如果用的數據集都已經將關節點提取完畢就不會用到這個參數
        self.box_thr = box_thr
        if self.box_thr is not None:
            # box_thr需要是以下幾種參數
            assert box_thr in ['0.5', '0.6', '0.7', '0.8', '0.9']

        # Thresholding Training Examples
        # 將valid_ratio保存
        self.valid_ratio = valid_ratio
        if self.valid_ratio is not None:
            # 如果有設定valid_ratio就會到這裡
            assert isinstance(self.valid_ratio, float)
            if self.box_thr is None:
                self.video_infos = self.video_infos = [
                    x for x in self.video_infos
                    if x['valid_frames'] / x['total_frames'] >= valid_ratio
                ]
            else:
                key = f'valid@{self.box_thr}'
                self.video_infos = [
                    x for x in self.video_infos
                    if x[key] / x['total_frames'] >= valid_ratio
                ]
                if self.box_thr != '0.5':
                    box_thr = float(self.box_thr)
                    for item in self.video_infos:
                        inds = [
                            i for i, score in enumerate(item['box_score'])
                            if score >= box_thr
                        ]
                        item['anno_inds'] = np.array(inds)

        if class_prob is not None:
            # 如果有設定class_prob就會到這裡
            self.class_prob = class_prob

        # 構建logger資訊
        logger = get_root_logger()
        logger.info(f'{len(self)} videos remain after valid thresholding')

    def load_annotations(self):
        """Load annotation file to get video information."""
        # 讀取標註訊息，這裡會需要檢查檔案需要是以.pkl做為結尾
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        # 透過mmcv將ann_file讀入
        # 這裡以ntu60_xsub_train為例
        # data = list[dict]，list長度會是訓練資料數量，dict當中會有每個訓練資料的詳細內容
        # dict = {
        #   'frame_dir': 這個我們不會用到，原始幀圖像的資料夾名稱，這裡已經提取完骨架所以不需要原始圖像
        #   'label': 分類類別
        #   'img_shape': 圖像高寬
        #   'original_shape': 原始圖像高寬
        #   'total_frames': 總共有多少幀
        #   'keypoint': 每幀的關節點資訊，ndarray shape [人數, 幀數, 關節點數量, 關節點(x, y)座標]
        #   'keypoint_score': 每個關節點預測的置信度，ndarray shape [人數, 幀數, 關節點數量]
        # }
        data = mmcv.load(self.ann_file)

        if self.split:
            # 如果有使用split就會到這裡
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            data = [x for x in data if x[identifier] in split[self.split]]

        # 遍歷data當中的每個dict
        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                # 將filename添加上前綴，不過這裡我們用不到
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                # 將frame_dir添加上前綴，不過這裡我們用不到
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])
        # 回傳data資訊
        return data
