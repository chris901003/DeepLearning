# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    # 已看過
    # 由main.py呼叫，用來train一個epoch
    # ---------------------------------------------------------
    # model = 就是model，在多gpu下就是有經過DistributedDataParallel的model
    # criterion = 預測後處理
    # data_loader_train = 訓練集的資料
    # optimizer = 優化器
    # device = 訓練設備
    # epoch = 當前Epoch
    # clip_max_norm = gradient clipping max norm(上面args寫的，目前不確定是什麼)
    # ---------------------------------------------------------
    # 將model以及criterion都設定成train模式
    model.train()
    criterion.train()
    # 就是MetricLogger，主要是可以在多gpu下同步每個gpu的速度
    # 這個基本上可以算是模板了，到處都長這樣
    # 實例化一個MetricLogger class，並且將分隔符設定為兩個空白
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 在metric_logger添加一個key且value是SmoothedValue型態且已經指定好裡面的樣子
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 與上面相同，只是SmoothedValue裡面的樣子有點不一樣而已
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # 一個string內容為Epoch: [當前epoch]
    header = 'Epoch: [{}]'.format(epoch)
    # 設定多少個batch打印一次
    print_freq = 10

    # 遍歷一整個Epoch
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # 可以先去看看dataset怎麼處理__getitem__，在coco.py(CocoDetection)當中可以找到
        # ---------------------------------------------------------
        # samples = NestedTensor格式裡面有已經打包好的圖像tensor以及mask
        # tensor = 已經把照片的tensor在高寬上面統一，不足的地方用0做填充
        # mask = 填充的部分會是True，非填充部分為False
        # targets = [batch_size, annotations_per_image] = [tuple[list[dict]]] = [幾張照片[每張照片的標籤信息[每個標籤的內容]]]
        # ---------------------------------------------------------
        # 轉換到設備上
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 輸入網路進行預測
        # ---------------------------------------------------------
        # outputs (Dict)
        # {
        #   'pred_logits': shape [batch_size, num_queries, num_classes + 1],
        #   'pred_boxes': shape [batch_size, num_queries, 4],
        #   'aux_outputs': List[{
        #                           'pred_logits': shape [batch_size, num_queries, num_classes + 1],
        #                           'pred_boxes': shape [batch_size, num_queries, 4]
        #                      }]
        # }
        # ---------------------------------------------------------
        outputs = model(samples)
        # ---------------------------------------------------------
        # 將模型預測結果outputs與真實標記targets放入計算損失
        # loss_dict =
        # {
        #   loss_ce:用交叉商計算出來的損失值,
        #   class_error:錯誤率，以100%格式,
        #   loss_bbox:l1損失,
        #   loss_giou:giou損失,
        #   cardinality_error:正負樣本匹配損失,
        #   ...
        # }
        # ---------------------------------------------------------
        loss_dict = criterion(outputs, targets)
        # weight_dict = criterion在構建的時候有一個變數就是weight_dict裡面存放了loss_dict的key要對應上的loss權重
        # 可以到detr.py中的build函數篇下面的地方可以看到
        weight_dict = criterion.weight_dict
        # 遍歷全部的loss，計算出來的loss乘上相對應的權重後加總起來獲得最後的loss值
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # 對於多gpu我們loss會需要進行一些調整
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # 保存一個沒有乘上權重的loss dict
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # 保存一個有乘上權重的loss dict
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        # 經過多gpu調整後的loss總和
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        # 經過多gpu調整後的loss總和
        loss_value = losses_reduced_scaled.item()

        # 確保loss沒有爆炸，如果爆炸了就停止訓練，因為梯度爆炸也沒有辦法反向傳遞了
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 歸零優化器
        optimizer.zero_grad()
        # 反向傳遞
        losses.backward()
        if max_norm > 0:
            # clip_grad_norm = 可以設定梯度閾值，當梯度大於設定值時就會被限制，不會超過避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # metric_logger的一些東西，更新一下
        # ---------------------------------------------------------
        # update吃的參數是**kwargs，這裡簡單說明一下**kwargs傳過去會是什麼
        # 傳過去的東西都會變成dict格式，所以我們要傳的時候有兩種選擇
        # 第一種就是傳的本身就是dict，這個時候我們需要在變數前面加上**就可以了
        # 第二種就是傳一個值，這時候我們需要給他一個key，例如loss_value，我們就需要給一個key叫做loss
        # 就是下面的loss=loss_value，這樣就會在dict中key為loss且value是loss_value
        # 傳過去的所有東西都會彙整成一個dict，所以要注意這裡不可以有相同的key不然會報錯
        # ---------------------------------------------------------
        # 如果是第一個epoch都會創建新的且max_len都是預設的20
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    # 同步所有的設備
    metric_logger.synchronize_between_processes()
    # 最後在印出平均loss
    print("Averaged stats:", metric_logger)
    # 回傳一些訓練狀態，就完成一個Epoch的訓練了
    # 拿取metric_logger中的meters中所有資料的加權平均值
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    # ---------------------------------------------------------
    # model = 預測模型
    # criterion = 預測後處理
    # postprocessors = 將輸出變成coco api想要的格式
    # data_loader_val = 驗證集的dataloader
    # base_ds = 從dataset_val中拿到coco的api，可能是後面計算mAP時會用到吧
    # device = 指定設備
    # output_dir = 檔案輸出的位置
    # ---------------------------------------------------------
    # 已看過
    # model以及criterion調整成驗證模式
    model.eval()
    criterion.eval()

    # 建立MetricLogger有空來研究
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 這裡我們不會再需要紀錄lr所以只有class_error
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # 這裡我們只會有bbox的key，segm是在訓練segmentation才會有，iou_types = ('bbox')
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # 實例化CocoEvaluator，bast_ds=COCO(annotation.json)
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        # 這裡我們不會用到
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    # 開始遍歷驗證集
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # 轉換到設備上面
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 輸入到預測模型中，輸出會跟train時一樣，差別會在沒有輔助輸出了
        # ---------------------------------------------------------
        # outputs (Dict)
        # {
        #   'pred_logits': shape [batch_size, num_queries, num_classes + 1],
        #   'pred_boxes': shape [batch_size, num_queries, 4]
        # ---------------------------------------------------------
        outputs = model(samples)
        # 損失計算
        loss_dict = criterion(outputs, targets)
        # 損失權重
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        # 多gpu對於loss的調整
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # 乘上權重後的loss
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        # 沒有乘上權重的loss
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # metric_logger更新
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # 拿出這個batch中每張照片的原始圖像大小，並且在第0個維度上面拼接
        # orig_target_sizes shape [batch_size, 2]
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # 將預測結果以及原始圖片大小輸入到postprocessors中
        # ---------------------------------------------------------
        # results = {
        # 'scores':[batch_size, num_queries],
        # 'labels':[batch_size, num_queries],
        # 'boxes':[batch_size, num_queries,4]
        # }
        # ---------------------------------------------------------
        # 這裡確實把100個queries都拿進去了，但是理論上要過濾掉背景才對，yolo系列的會在這裡前經過nms
        # 我覺得至少要把置信度過小的過濾掉吧
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            # 預測segmentation才會用到，這裡我們先不會用到
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # res(Dict) = 拿圖片的id對上那張圖片預測出來的結果
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            # 這裡我們有coco_evaluator，不過這部分晚點再來看
            # 將image_id對應上預測結果輸入給coco_evaluator
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            # 這裡預測segmentation才會有，我們不會用到
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    # metric_logger同步每個設備
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    # 計算mAP
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    # 各種資料處理，先暫時不看
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    # 將驗證過程的東西往外傳
    return stats, coco_evaluator
