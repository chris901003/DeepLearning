import json
import copy

from PIL import Image, ImageDraw
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .distributed_utils import all_gather, is_main_process
from transforms import affine_points


def merge(img_ids, eval_results):
    # 已看過
    """将多个进程之间的数据汇总在一起"""
    all_img_ids = all_gather(img_ids)
    all_eval_results = all_gather(eval_results)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_results = []
    for p in all_eval_results:
        merged_eval_results.extend(p)

    merged_img_ids = np.array(merged_img_ids)

    # keep only unique (and in sorted order) images
    # 去除重复的图片索引，多GPU训练时为了保证每个进程的训练图片数量相同，可能将一张图片分配给多个进程
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_results = [merged_eval_results[i] for i in idx]

    return list(merged_img_ids), merged_eval_results


class EvalCOCOMetric:
    def __init__(self,
                 coco: COCO = None,
                 iou_type: str = "keypoints",
                 results_file_name: str = "predict_results.json",
                 classes_mapping: dict = None,
                 threshold: float = 0.2):
        """
        :param coco: coco apis，從驗證集的dataloader中拿出來的
        :param iou_type: 要計算iou的東西，這裡會是keypoints
        :param results_file_name: 暫時不確定作用
        :param classes_mapping: 預設為None
        :param threshold: 預設為0.2
        """
        # 已看過
        # 深拷貝一份coco apis
        self.coco = copy.deepcopy(coco)
        self.obj_ids = []  # 记录每个进程处理目标(person)的ids
        self.results = []
        self.aggregation_results = None
        self.classes_mapping = classes_mapping
        self.coco_evaluator = None
        assert iou_type in ["keypoints"]
        self.iou_type = iou_type
        self.results_file_name = results_file_name
        self.threshold = threshold

    def plot_img(self, img_path, keypoints, r=3):
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        for i, point in enumerate(keypoints):
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                         fill=(255, 0, 0))
        img.show()

    def prepare_for_coco_keypoints(self, targets, outputs):
        # 已看過
        # targets = 從dataloader出來的
        # outputs = 裡面有兩個資料
        # preds shape [batch_size, num_kps, 2]
        # maxvals shape [batch_size, num_kps, 1]
        # 遍历每个person的预测结果(注意这里不是每张，一张图片里可能有多个person)
        for target, keypoints, scores in zip(targets, outputs[0], outputs[1]):
            # 正常來說不會是0
            if len(keypoints) == 0:
                continue

            # 每一個人物都會有一個獨一無二的index，這裡是要避免有重複的，可以檢查一下
            obj_idx = int(target["obj_index"])
            if obj_idx in self.obj_ids:
                # 防止出现重复的数据
                continue

            # 記錄下來表示已經有加入過了
            self.obj_ids.append(obj_idx)
            # self.plot_img(target["image_path"], keypoints)

            # mask shape [num_kps, 1]，如果scores的值大於0.2會是True否則為False
            mask = np.greater(scores, 0.2)
            # 將所有大於0.2的值做加總後平均
            if mask.sum() == 0:
                k_score = 0
            else:
                k_score = np.mean(scores[mask])

            # keypoints shape [num_kps, 2]，scores shape [num_kps, 1]

            # keypoints shape [num_kps, 3]，分數會接在座標後面
            keypoints = np.concatenate([keypoints, scores], axis=1)
            # keypoints shape [num_kps * 3]
            keypoints = np.reshape(keypoints, -1)

            # We recommend rounding coordinates to the nearest tenth of a pixel
            # to reduce resulting JSON file size.
            # 將keypoints當中的數值取到小數後兩位即可，不需要留太多位數
            keypoints = [round(k, 2) for k in keypoints.tolist()]

            # 構建對於一個圖片的預測結果
            res = {"image_id": target["image_id"],
                   "category_id": 1,  # person
                   "keypoints": keypoints,
                   "score": target["score"] * k_score}

            # 將結果記錄下來
            self.results.append(res)

    def update(self, targets, outputs):
        # 已看過
        # targets = 從dataloader出來的
        # outputs = 裡面有兩個資料
        # preds shape [batch_size, num_kps, 2]
        # maxvals shape [batch_size, num_kps, 1]
        # 這裡只處理keypoints的iou
        if self.iou_type == "keypoints":
            self.prepare_for_coco_keypoints(targets, outputs)
        else:
            raise KeyError(f"not support iou_type: {self.iou_type}")

    def synchronize_results(self):
        # 已看過
        # 同步所有进程中的数据
        eval_ids, eval_results = merge(self.obj_ids, self.results)
        self.aggregation_results = {"obj_ids": eval_ids, "results": eval_results}

        # 主进程上保存即可
        if is_main_process():
            # results = []
            # [results.extend(i) for i in eval_results]
            # write predict results into json file
            # 寫成json檔案，這樣coco才可以計算mAP
            json_str = json.dumps(eval_results, indent=4)
            with open(self.results_file_name, 'w') as json_file:
                json_file.write(json_str)

    def evaluate(self):
        # 已看過
        # 只在主进程上评估即可
        if is_main_process():
            # accumulate predictions from all images
            # 拿到正確的coco apis
            coco_true = self.coco
            # 讀取預測輸出的json檔案
            coco_pre = coco_true.loadRes(self.results_file_name)

            # 放入進行計算
            self.coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType=self.iou_type)

            self.coco_evaluator.evaluate()
            self.coco_evaluator.accumulate()
            print(f"IoU metric: {self.iou_type}")
            self.coco_evaluator.summarize()

            coco_info = self.coco_evaluator.stats.tolist()  # numpy to list
            # 回傳計算完的結果
            return coco_info
        else:
            return None
