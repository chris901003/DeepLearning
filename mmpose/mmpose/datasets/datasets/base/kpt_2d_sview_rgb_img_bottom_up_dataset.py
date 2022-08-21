# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod

import numpy as np
import xtcocotools
from torch.utils.data import Dataset
from xtcocotools.coco import COCO

from mmpose.datasets import DatasetInfo
from mmpose.datasets.pipelines import Compose


class Kpt2dSviewRgbImgBottomUpDataset(Dataset, metaclass=ABCMeta):
    """Base class for bottom-up datasets.

    All datasets should subclass it.
    All subclasses should overwrite:
        Methods:`_get_single`

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        coco_style (bool): Whether the annotation json is coco-style.
            Default: True
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 coco_style=True,
                 test_mode=False):
        """ 所有bottom-up的資料處理的底層
        Args:
            ann_file: 標註資料檔案路徑
            img_prefix: 圖像的root路徑
            data_cfg: data的config資料
            pipeline: 圖像處理流
            dataset_info: dataset的資訊
            coco_style: 檔案存放方式是否為coco
            test_mode: 是否為test模式
        """

        # 存放圖像詳細資訊的字典
        self.image_info = {}
        # 存放圖像對應的標註訊息字典
        self.ann_info = {}

        # 保存傳入資料
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.pipeline = pipeline
        self.test_mode = test_mode

        # bottom-up
        # 分別從data_cfg當中獲取base_size與base_sigma
        self.base_size = data_cfg['base_size']
        self.base_sigma = data_cfg['base_sigma']
        # 將int_sigma設定成False
        self.int_sigma = False

        # 將data_cfg當中的image_size保存到ann_info當中
        self.ann_info['image_size'] = np.array(data_cfg['image_size'])
        # 將data_cfg當中的heatmap_size保存到anno_info當中
        self.ann_info['heatmap_size'] = np.array(data_cfg['heatmap_size'])
        # 將data_cfg當中的num_joints保存到anno_info當中
        self.ann_info['num_joints'] = data_cfg['num_joints']
        # 將data_cfg當中的num_scales保存到anno_info當中
        self.ann_info['num_scales'] = data_cfg['num_scales']
        # 將data_cfg當中的scale_aware_sigma保存到anno_info當中
        self.ann_info['scale_aware_sigma'] = data_cfg['scale_aware_sigma']

        # 將data_cfg當中的inference_channel保存到anno_info當中
        self.ann_info['inference_channel'] = data_cfg['inference_channel']
        # 將data_cfg當中的dataset_channel保存到anno_info當中
        self.ann_info['dataset_channel'] = data_cfg['dataset_channel']

        # 獲取use_nms以及soft_nms以及oks_thr資訊
        self.use_nms = data_cfg.get('use_nms', False)
        self.soft_nms = data_cfg.get('soft_nms', True)
        self.oks_thr = data_cfg.get('oks_thr', 0.9)

        if dataset_info is None:
            # 如果沒有給定dataset_info就會報錯
            raise ValueError(
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.')

        # 將dataset_info變成DatasetInfo實例化對象
        dataset_info = DatasetInfo(dataset_info)

        # 核對anno_info與dataset_info當中指定的關節點數量是否相同
        assert self.ann_info['num_joints'] == dataset_info.keypoint_num
        # 將兩兩一組的對稱關節點index保存到anno_info當中
        self.ann_info['flip_pairs'] = dataset_info.flip_pairs
        # 將每個關節點的對稱關節點index保存到anno_info當中
        self.ann_info['flip_index'] = dataset_info.flip_index
        # 將哪些index為上半身存放到anno_info當中
        self.ann_info['upper_body_ids'] = dataset_info.upper_body_ids
        # 將哪些index為下半身存放到anno_info當中
        self.ann_info['lower_body_ids'] = dataset_info.lower_body_ids
        # 獲取每個關節點在計算損失值時的權重
        self.ann_info['joint_weights'] = dataset_info.joint_weights
        # 獲取哪兩個關節點index需要進行連線
        self.ann_info['skeleton'] = dataset_info.skeleton
        # 將sigmas資料儲存
        self.sigmas = dataset_info.sigmas
        # 保存當前使用的資料集名稱
        self.dataset_name = dataset_info.dataset_name

        if coco_style:
            # 如果使用的資料集是coco類型的就會到這裡
            # 構建COCO實例化對象，主要是提供api讀取資料
            self.coco = COCO(ann_file)
            if 'categories' in self.coco.dataset:
                # 如果coco.dataset當中有categories就會到這裡
                # 因為我們只需要人的資訊，所以cats當中只會有person
                cats = [
                    cat['name']
                    for cat in self.coco.loadCats(self.coco.getCatIds())
                ]
                # 分類類別會是背景加上cats
                self.classes = ['__background__'] + cats
                # 獲取分類類別數
                self.num_classes = len(self.classes)
                # 構建類別名稱對應到index的字典
                self._class_to_ind = dict(
                    zip(self.classes, range(self.num_classes)))
                # 構建類別名稱對應到coco當中的index的字典
                self._class_to_coco_ind = dict(
                    zip(cats, self.coco.getCatIds()))
                # 構建當前index對應到coco當中的index的字典
                self._coco_ind_to_class_ind = dict(
                    (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                    for cls in self.classes[1:])
            # 獲取資料集當中的所有圖片index
            self.img_ids = self.coco.getImgIds()
            if not test_mode:
                # 如果是訓練模式就會到這裡，將沒有目標的圖像過濾掉
                self.img_ids = [
                    img_id for img_id in self.img_ids if
                    len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
                ]
            # 獲取最後的圖像數量
            self.num_images = len(self.img_ids)
            # index對應到圖像名稱，圖像名稱對應到index
            self.id2name, self.name2id = self._get_mapping_id_name(self.coco.imgs)

        # 構建圖像處理流
        self.pipeline = Compose(self.pipeline)

    @staticmethod
    def _get_mapping_id_name(imgs):
        """
        Args:
            imgs (dict): dict of image info.

        Returns:
            tuple: Image name & id mapping dicts.

            - id2name (dict): Mapping image id to name.
            - name2id (dict): Mapping image name to id.
        """
        # 構建index對應到的圖像名稱
        id2name = {}
        # 構建圖像名稱對應到的index
        name2id = {}
        # 遍歷所有圖像
        for image_id, image in imgs.items():
            # 獲取檔案名稱
            file_name = image['file_name']
            # 保存index對應到圖像名稱
            id2name[image_id] = file_name
            # 保存名稱對應到index
            name2id[file_name] = image_id

        # 返回對應結果
        return id2name, name2id

    def _get_mask(self, anno, idx):
        """ 獲取mask資訊
        Args:
            anno: 標註訊息，list[dict]，list長度就是該張圖像有多少標註訊息，dict標註訊息的詳細資訊
            idx: 圖像的index
        """
        """Get ignore masks to mask out losses."""
        # 獲取coco工具實例化對象
        coco = self.coco
        # 獲取圖像詳細資訊，包括高寬資訊
        img_info = coco.loadImgs(self.img_ids[idx])[0]

        # 構建全為0的ndarray且shape = [height, width]
        m = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)

        # 遍歷該圖像的標註訊息
        for obj in anno:
            if 'segmentation' in obj:
                # 如果當中有segmentation就會到這裡
                if obj['iscrowd']:
                    # 如果有被標註為難檢測對象就會到這裡
                    rle = xtcocotools.mask.frPyObjects(obj['segmentation'],
                                                       img_info['height'],
                                                       img_info['width'])
                    m += xtcocotools.mask.decode(rle)
                elif obj['num_keypoints'] == 0:
                    # 如果沒有半個關節點就會到這裡
                    rles = xtcocotools.mask.frPyObjects(
                        obj['segmentation'], img_info['height'],
                        img_info['width'])
                    for rle in rles:
                        m += xtcocotools.mask.decode(rle)

        # 如果m小於0.5的地方會是True其他會是False
        return m < 0.5

    @abstractmethod
    def _get_single(self, idx):
        """Get anno for a single image."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, results, *args, **kwargs):
        """Evaluate keypoint results."""

    def prepare_train_img(self, idx):
        """Prepare image for training given the index."""
        # 根據傳入的idx獲取對應的圖像資料
        # 透過_get_single獲取對應的資料，這裡會提取出接下來pipeline需要使用到的資料
        results = copy.deepcopy(self._get_single(idx))
        # 將anno_info資訊放到results當中
        results['ann_info'] = self.ann_info
        # 將results放到圖像處理流當中
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Prepare image for testing given the index."""
        results = copy.deepcopy(self._get_single(idx))
        results['ann_info'] = self.ann_info
        return self.pipeline(results)

    def __len__(self):
        """Get dataset length."""
        return len(self.img_ids)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            # 如果是測試模式就會到這裡，透過prepare_test_img獲取圖像資訊
            return self.prepare_test_img(idx)

        # 如果是訓練模式就會到這裡，透過prepare_train_img獲取訓練圖像資訊
        return self.prepare_train_img(idx)
