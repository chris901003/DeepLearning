import math
import sys
import time

import torch

import transforms
import train_utils.distributed_utils as utils
from .coco_eval import EvalCOCOMetric
from .loss import KpLoss


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    """
    :param model: 預測模型
    :param optimizer: 優化器
    :param data_loader: 資料集
    :param device: 訓練設備
    :param epoch: 當前為第幾個epoch
    :param print_freq: 多少個batch會打印一次當前狀況
    :param warmup: 是否使用熱身訓練
    :param scaler: amp會用到
    :return:
    """
    # 已看過
    # 訓練一個epochs
    # 將模型調整成訓練模式
    model.train()
    # 實例化metric_logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 加入學習率
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        # 構建學習率函數
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # 實例化損失計算
    mse = KpLoss()
    # mloss shape [1]
    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 獲取圖片以及標註訊息，要完整標註訊息可到my_dataset_coco的collate_fn裡面看
        # 原先images就已經是堆疊好的了，正常來說這行沒有作用
        # images shape [batch_size, 3, height, width]
        images = torch.stack([image.to(device) for image in images])

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # 將圖像放入模型進行正向傳播
            # results shape [batch_size, 關節點數量, 64, 48]
            results = model(images)

            # 將模型輸出與真實標籤拿去計算損失，回傳的就是計算完的損失值
            losses = mse(results, targets)

        # reduce losses over all GPUs for logging purpose
        # 多gpu會用到
        loss_dict_reduced = utils.reduce_dict({"losses": losses})
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失，這裡記錄的是平均損失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        # 提早停止訓練的條件
        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 清空優化器
        optimizer.zero_grad()
        if scaler is not None:
            # 使用amp的反向傳遞方式
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 不使用amp的反向傳遞方式
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        # 更新metric_logger的內容
        metric_logger.update(loss=losses_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    # 回傳損失值以及當前的學習率
    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device, flip=False, flip_pairs=None):
    """
    :param model: 預測模型
    :param data_loader: 驗證集的dataloader
    :param device: 預測設備
    :param flip: 是否有翻轉，這裡預設會是True
    :param flip_pairs: 如果啟用翻轉就會需要給，左右對應的index，可以到train.py看傳入的資料
    :return:
    """
    # 已看過
    # 檢查當啟用翻轉後有沒有傳入flip_pairs
    if flip:
        assert flip_pairs is not None, "enable flip must provide flip_pairs."

    # 將模型設定成驗證模式
    model.eval()
    # 實例化metric_logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    # coco驗證
    key_metric = EvalCOCOMetric(data_loader.dataset.coco, "keypoints", "key_results.json")
    # 開始驗證
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        # images shape [batch_size, 3, height, width]
        images = torch.stack([img.to(device) for img in image])

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            # 多gpu時的東西
            torch.cuda.synchronize(device)

        # 計算時間
        model_time = time.time()
        # 將圖像放入模型進行預測
        # outputs shape [batch_size, num_kps, height, width]
        outputs = model(images)
        # 預設flip會是True，這裡等於我們會對一張圖片進行兩次預測最後會取兩次的平均當作最後的結果，隨然會更加準確但是會讓效率降低
        if flip:
            # 對圖像翻轉，這裡是對最後一個寬度維度進行翻轉
            flipped_images = transforms.flip_images(images)
            # 對進行翻轉後的圖像再做一次預測
            flipped_outputs = model(flipped_images)
            # 傳入預測結果以及左右對應的index，接收翻轉好的結果
            # flipped_outputs shape [batch_size, num_kps, height, width]
            flipped_outputs = transforms.flip_back(flipped_outputs, flip_pairs)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            # 透過對熱力圖稍微調整一下可以獲得更好的效果，調整過後shape不改變
            flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
            # 與不進行水平翻轉的做一次平均
            outputs = (outputs + flipped_outputs) * 0.5

        # 結束推理時間
        model_time = time.time() - model_time

        # decode keypoint
        # 將熱力圖結果要映射回原圖上，使用原先映射過去的相反方法
        reverse_trans = [t["reverse_trans"] for t in targets]
        # post_processing設定是否需要對最大值位置做小幅度偏移
        # 將熱力圖映射回去，outputs裡面會有下面兩個值
        # preds shape [batch_size, num_kps, 2]
        # maxvals shape [batch_size, num_kps, 1]
        outputs = transforms.get_final_preds(outputs, reverse_trans, post_processing=True)

        # 更新coco api
        key_metric.update(targets, outputs)
        # 更新狀態
        metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 同步所有进程中的数据
    key_metric.synchronize_results()

    if utils.is_main_process():
        coco_info = key_metric.evaluate()
    else:
        coco_info = None

    return coco_info
