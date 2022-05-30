import os

import torch
from tqdm import tqdm

from utils.utils import get_lr


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    """
    :param model_train: 進入training模式的model
    :param model: 原始model
    :param ema: ema調整權重狀況
    :param yolo_loss: 損失計算實例化的class
    :param loss_history: 紀錄訓練過程的損失實例化的class
    :param eval_callback: 訓練一定epoch後驗證用的實例化class
    :param optimizer: 優化器
    :param epoch: 當前epoch
    :param epoch_step: 每個epoch中訓練data被分成幾塊
    :param epoch_step_val: 每個epoch中驗證data被分成幾塊
    :param gen: train_data_loader
    :param gen_val: val_data_loader
    :param Epoch: 總epoch數
    :param cuda: 是否使用gpu訓練
    :param fp16: 是否啟用amp
    :param scaler: 如果啟用amp
    :param save_period: 每多少輪會保存一次模型
    :param save_dir: 保存位置
    :param local_rank: 判斷要不要印出一些相關資料
    :return:
    """
    # 已看過
    # 由train.py調用
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            # 轉換到cuda上
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # 沒有啟用amp模式
            # ----------------------#
            #   前向传播
            # ----------------------#
            # outputs shape [(batch_size, 5 + num_classes, feature_weight, feature_height)]
            outputs = model_train(images)

            # ----------------------#
            #   计算损失
            # ----------------------#
            # 外層的list是batch_size，內層的list是每張圖有的gt_box
            # targets shape [[(center_x, center_y, w, h, class)]]
            loss_value, loss_dict = yolo_loss(outputs, targets)

            # ----------------------#
            #   反向传播
            # ----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            # 有開啟amp
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                # ----------------------#
                #   计算损失
                # ----------------------#
                loss_value, loss_dict = yolo_loss(outputs, targets)

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            # tqdm的一些東西，就是可以顯示當前的狀態
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'iou_loss': loss_dict['iou_loss'].item(),
                                'obj_loss': loss_dict['obj_loss'].item(),
                                'cls_loss': loss_dict['cls_loss'].item()})
            pbar.update(1)

    # 完成一個Epoch訓練
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train_eval(images)

            # ----------------------#
            #   计算损失
            # ----------------------#
            # 我們計算損失但是不會進行反向傳遞
            loss_value, loss_dict = yolo_loss(outputs, targets)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'iou_loss': loss_dict['iou_loss'].item(),
                                'obj_loss': loss_dict['obj_loss'].item(),
                                'cls_loss': loss_dict['cls_loss'].item()})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth"
                                                     % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
