import torch
from torch import nn
import train_utils.distributed_utils as utils


def criterion(inputs, target):
    """
    :param inputs: {
                       'out': [batch_size, num_classes, 480, 480]
                       'aux': [batch_size, num_classes, 480, 480]
                   }
    :param target: [batch_size, 3, 480, 480]
    :return:
    """
    # 已看過
    losses = {}
    for name, x in inputs.items():
        # 進行交叉熵計算
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    # 依照比重計算損失
    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes):
    """
    :param model: 預測模型
    :param data_loader: 數據集
    :param device: 設備
    :param num_classes: 分類類別數
    :return:
    """
    # 已看過
    # 將模型調整成驗證模式
    model.eval()
    # 構建混淆矩陣
    confmat = utils.ConfusionMatrix(num_classes)
    # 構建metric_logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # 設定成不計算反向傳遞值
    with torch.no_grad():
        # 遍歷一個epoch
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            # 輸入到模型
            output = model(image)
            # 只取出最終輸出部分，不會用到輔助輸出
            output = output['out']

            # 更新混淆矩陣
            confmat.update(target.flatten(), output.argmax(1).flatten())

        # 多gpu時需要同步
        confmat.reduce_from_all_processes()

    # 回傳混淆矩陣實例對象
    return confmat


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler, print_freq=10, scaler=None):
    """
    :param model: 預測模型
    :param optimizer: 優化器
    :param data_loader: 數據集
    :param device: 設備
    :param epoch: 第幾個epoch
    :param lr_scheduler: 優化器學習率調整
    :param print_freq: 多少個batch會打印狀態
    :param scaler: amp會用到的
    :return:
    """
    # 已看過
    # 將模型設定為訓練模式
    model.train()
    # 構建metric_logger，每個數據之間用兩個空白
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 在metric_logger中添加一個lr的key且value型態是SmoothedValue
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # header = string表示當前是在哪個epoch
    header = 'Epoch: [{}]'.format(epoch)

    # 使用metric_logger中的log_every讀取一個batch的資料
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        # 拿到圖片以及正確的輸出，並且放入模型內進行預測
        # image, target shape [batch_size, channel, w, h]
        image, target = image.to(device), target.to(device)
        # 依據是否開啟amp會傳入enabled
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # 進行預測
            # output = {
            #   'out': [batch_size, num_classes, 480, 480]
            #   'aux': [batch_size, num_classes, 480, 480]
            # }
            output = model(image)
            # target = [batch_size, 3, 480, 480]
            # 進行損失計算
            loss = criterion(output, target)

        # 優化器清0
        optimizer.zero_grad()
        # 看有沒有使用amp
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 學習率更新，這裡就是一個batch就更新一次
        lr_scheduler.step()

        # 提取優化器中的學習率
        lr = optimizer.param_groups[0]["lr"]
        # 更新metric_logger
        metric_logger.update(loss=loss.item(), lr=lr)

    # 回傳全局的平均loss以及當前學習率
    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    """
    :param optimizer: 優化器
    :param num_step: 一個epoch有幾個batch
    :param epochs: 總共多少個epoch
    :param warmup: 是否要使用暖身訓練
    :param warmup_epochs:
    :param warmup_factor:
    :return:
    """
    # 已看過
    # 構建學習率調整函數
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        # 每一個batch就會調用一次，所以這裡的x每訓練玩一個batch就會加一
        if warmup is True and x <= (warmup_epochs * num_step):
            # 還在暖身階段
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    # 回傳學習率調整方法
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
