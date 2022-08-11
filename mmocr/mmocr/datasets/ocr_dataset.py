# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import DATASETS

from mmocr.core.evaluation.ocr_metric import eval_ocr_metric
from mmocr.datasets.base_dataset import BaseDataset
from mmocr.utils import is_type_list


@DATASETS.register_module()
class OCRDataset(BaseDataset):

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['text'] = results['img_info']['text']

    def evaluate(self, results, metric='acc', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str: float]
        """
        # 已看過，評測預測模型的指標
        # results = 預測的結果資料，list[dict]，list長度就會是該dataset的圖像數量，dict就是預測的詳細資訊
        # metric = 需要評測的指標
        # logger = 紀錄用的

        # 檢查傳入的metric是否為list且當中為str，或是單純傳入str
        assert isinstance(metric, str) or is_type_list(metric, str)

        # 記錄下預測的文字以及標註的文字
        gt_texts = []
        pred_texts = []
        # 遍歷整個dataset的圖像數量
        for i in range(len(self)):
            # 獲取當前index圖像的資料
            item_info = self.data_infos[i]
            # 獲取標註的文串
            text = item_info['text']
            # 將結果放到gt_texts當中
            gt_texts.append(text)
            # 將預測的文字結果放到pred_texts當中
            pred_texts.append(results[i]['text'])

        # 透過eval_ocr_metric計算檢測的指標
        eval_results = eval_ocr_metric(pred_texts, gt_texts, metric=metric)

        return eval_results
