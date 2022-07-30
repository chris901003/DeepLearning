# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset

import mmocr.utils as utils
from mmocr import digit_version
from mmocr.core.evaluation.hmean import eval_hmean


@DATASETS.register_module()
class IcdarDataset(CocoDataset):
    """Dataset for text detection while ann_file in coco format.

    Args:
        ann_file_backend (str): Storage backend for annotation file,
            should be one in ['disk', 'petrel', 'http']. Default to 'disk'.
    """
    CLASSES = ('text')

    def __init__(self,
                 ann_file,
                 pipeline,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 select_first_k=-1,
                 ann_file_backend='disk'):
        """ 已看過，icdar資料集的初始化方式
        Args:
            ann_file: 標註資訊的檔案位置
            pipeline: 圖像處理流
            classes: 預設為None
            data_root: 檔案的root路徑，預設為None
            img_prefix: 圖像資料夾路徑，預設為None
            seg_prefix: segmentation圖像資料夾路徑，預設為None
            proposal_file: proposal的檔案路徑，預設為None
            test_mode: 是否為測試模式，預設為False
            filter_empty_gt: 是否要過濾掉沒有gt的圖像，預設為True
            select_first_k: 預設為-1
            ann_file_backend: 標註檔案存放的設備，預設為disk
        """
        # select first k images for fast debugging.
        # select_first_k的設定是為了debugging用的
        self.select_first_k = select_first_k
        # 檢查backend是否是有支援的
        assert ann_file_backend in ['disk', 'petrel', 'http']
        # 記錄下ann_file_backend
        self.ann_file_backend = ann_file_backend

        # 繼承mm.det的CocoDataset，將傳入的參數放進去進行初始化
        # 在mmocr處理icdar時會提供json檔案且此json檔案資料內容排法與coco相同，所以這裡可以透過cocotools進行讀取
        super().__init__(ann_file, pipeline, classes, data_root, img_prefix,
                         seg_prefix, proposal_file, test_mode, filter_empty_gt)

        # Set dummy flags just to be compatible with MMDet
        # 設定一個沒用的flags只是為了可以用在MMDet上，也就是只是因為MMDet需要flags但我們這裡用不到
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        # 已看過，主要是使用cocotools讀取標註檔案，因為我們有準備icdar的json資料且資料排法與coco相同，所以可以使用
        # anno_file = 標註文件檔案位置
        if self.ann_file_backend == 'disk':
            # 標註文件存放設備如果是硬碟就會到這裡，使用COCO api將資料讀取出來
            self.coco = COCO(ann_file)
        else:
            # 其他存放設備會到這裡
            mmcv_version = digit_version(mmcv.__version__)
            if mmcv_version < digit_version('1.3.16'):
                raise Exception('Please update mmcv to 1.3.16 or higher '
                                'to enable "get_local_path" of "FileClient".')
            file_client = mmcv.FileClient(backend=self.ann_file_backend)
            with file_client.get_local_path(ann_file) as local_path:
                self.coco = COCO(local_path)
        # 獲取我們指定的類別與標註文件當中所有類別的index對應關係，這裡我們需要的類別是所有在標註文件的類別，所以不會有差別
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        # 獲取對應dict，key=在標註文件當中第index類，value=該類別在預測當中會是第index類
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        # 獲取訓練圖像的index，這裡的長度就會是訓練圖像的數量
        self.img_ids = self.coco.get_img_ids()
        # 最後需要保存下來的東西，每一個會是一張圖像的詳細資料
        data_infos = []

        # 計數用的
        count = 0
        # 遍歷所有需要訓練的圖像
        for i in self.img_ids:
            # 獲取該圖像的資料
            info = self.coco.load_imgs([i])[0]
            # 將圖像檔案名稱保留
            info['filename'] = info['file_name']
            # 添加到data_infos當中
            data_infos.append(info)
            # 計數器加一
            count = count + 1
            if count > self.select_first_k and self.select_first_k > 0:
                # 如果要方便進行debug可以設定加載select_first_k張圖像就可以，可以讓一個epoch訓練速度變快
                break
        # 回傳經過整理的圖像資料
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, masks_ignore, seg_map. "masks"  and
                "masks_ignore" are represented by polygon boundary
                point sequences.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ignore = []
        gt_masks_ann = []

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
                gt_masks_ignore.append(ann.get(
                    'segmentation', None))  # to float32 for latter processing

            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks_ignore=gt_masks_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='hmean-iou',
                 logger=None,
                 score_thr=None,
                 min_score_thr=0.3,
                 max_score_thr=0.9,
                 step=0.1,
                 rank_list=None,
                 **kwargs):
        """Evaluate the hmean metric.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            score_thr (float): Deprecated. Please use min_score_thr instead.
            min_score_thr (float): Minimum score threshold of prediction map.
            max_score_thr (float): Maximum score threshold of prediction map.
            step (float): The spacing between score thresholds.
            rank_list (str): json file used to save eval result
                of each image after ranking.
        Returns:
            dict[dict[str: float]]: The evaluation results.
        """
        assert utils.is_type_list(results, dict)

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['hmean-iou', 'hmean-ic13']
        metrics = set(metrics) & set(allowed_metrics)

        img_infos = []
        ann_infos = []
        for i in range(len(self)):
            img_info = {'filename': self.data_infos[i]['file_name']}
            img_infos.append(img_info)
            ann_infos.append(self.get_ann_info(i))

        eval_results = eval_hmean(
            results,
            img_infos,
            ann_infos,
            metrics=metrics,
            score_thr=score_thr,
            min_score_thr=min_score_thr,
            max_score_thr=max_score_thr,
            step=step,
            logger=logger,
            rank_list=rank_list)

        return eval_results
