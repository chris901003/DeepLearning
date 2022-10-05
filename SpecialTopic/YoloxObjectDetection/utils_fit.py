import os
import torch
from tqdm import tqdm
import numpy as np
import json
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from utils import get_lr, decode_outputs, non_max_suppression


def fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, fp16, scaler, save_period, save_dir, num_classes, local_rank=0, eval_period=-1,
                  coco_json_file=None, training_state=None, best_train_loss=False, best_val_loss=False, best_mAP=False,
                  save_optimizer=False, logger=None, email_send_to=None, save_log_period=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    loss = 0
    val_loss = 0
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, miniters=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, targets = batch[0], batch[1]

        # 可視化
        # from torchvision import transforms
        # from PIL import Image
        # import numpy as np
        # import cv2
        # target = targets[0]
        # img = transforms.ToPILImage()(images[0]).convert('RGB')
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # for bbox in target:
        #     xmin = int(bbox[0] - bbox[2] / 2)
        #     ymin = int(bbox[1] - bbox[3] / 2)
        #     xmax = int(bbox[0] + bbox[2] / 2)
        #     ymax = int(bbox[1] + bbox[3] / 2)
        #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        # img.show()
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(images)
            loss_value, loss_dict = yolo_loss(outputs, targets)
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                loss_value, loss_dict = yolo_loss(outputs, targets)
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'iou_loss': loss_dict['iou_loss'].item(),
                                'obj_loss': loss_dict['obj_loss'].item(),
                                'cls_loss': loss_dict['cls_loss'].item()})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, miniters=0.3)
    if best_train_loss and training_state is not None:
        train_loss = training_state.get('train_loss')
        if train_loss > (loss / len(gen)):
            training_state['train_loss'] = loss / len(gen)
            if save_optimizer:
                save = dict(model_weight=model.state_dict(), optimizer_weight=optimizer.state_dict(), Epoch=epoch + 1)
                torch.save(save, os.path.join(save_dir, f'yolox_best_train_loss.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, f'yolox_best_train_loss.pth'))
    model_train_eval = model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, ori_size, keep_ratio, image_path = batch[0], batch[1], batch[2], batch[3], batch[4]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
            optimizer.zero_grad()
            outputs = model_train_eval(images)
            loss_value, loss_dict = yolo_loss(outputs, targets)
        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'iou_loss': loss_dict['iou_loss'].item(),
                                'obj_loss': loss_dict['obj_loss'].item(),
                                'cls_loss': loss_dict['cls_loss'].item()})
            pbar.update(1)
    if best_val_loss and training_state is not None:
        eval_loss = training_state.get('val_loss')
        if eval_loss > (val_loss / len(gen_val)):
            training_state['val_loss'] = val_loss / len(gen_val)
            if save_optimizer:
                save = dict(model_weight=model.state_dict(), optimizer_weight=optimizer.state_dict(),
                            Epoch=epoch + 1)
                torch.save(save, os.path.join(save_dir, f'yolox_best_val_loss.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, f'yolox_best_val_loss.pth'))

    if local_rank == 0:
        pbar.clear()
        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

    if (eval_period != -1) and ((epoch + 1) % eval_period == 0):
        json_file = './coco_mAP_file.json'
        assert coco_json_file is not None, '如果需要進行coco mAP計算需要提供json檔'
        res = list()
        for iteration, batch in enumerate(tqdm(gen_val)):
            images, targets, ori_size, keep_ratio, image_path = batch[0], batch[1], batch[2], batch[3], batch[4]
            image_id = int(os.path.splitext(os.path.basename(image_path))[0])
            input_shape = images.shape[-2:]
            if cuda:
                images = images.cuda(local_rank)
            with torch.no_grad():
                outputs = model_train_eval(images)
            outputs = decode_outputs(outputs, input_shape)
            results = non_max_suppression(outputs, num_classes, input_shape, ori_size, keep_ratio, conf_thres=0.5,
                                          nms_thres=0.3)
            if results[0] is None:
                continue
            top_label = np.array(results[0][:, 6], dtype='int32').tolist()
            top_conf = (results[0][:, 4] * results[0][:, 5]).tolist()
            top_boxes = results[0][:, :4].tolist()
            for label, conf, boxes in zip(top_label, top_conf, top_boxes):
                ymin, xmin, ymax, xmax = boxes
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(ori_size[0][1], xmax)
                ymax = min(ori_size[0][0], ymax)
                bbox = [round(xmin, 2), round(ymin, 2), round(xmax - xmin, 2), round(ymax - ymin, 2)]
                data = dict(image_id=int(image_id), category_id=int(label) + 1, bbox=bbox, score=round(conf, 2))
                res.append(data)
        if len(res) == 0:
            print('Without detect any box, skip mAP validate.')
        else:
            print('Writing json file ...')
            with open(json_file, 'w') as f:
                json.dump(res, f)
            coco_anno = coco.COCO(coco_json_file)
            coco_det = coco_anno.loadRes(json_file)
            coco_eval = COCOeval(coco_anno, coco_det, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            os.remove(json_file)
            mAP = float(coco_eval.stats[0])
            if best_mAP and training_state is not None:
                if mAP > training_state['mAP']:
                    training_state['mAP'] = mAP
                    if save_optimizer:
                        save = dict(model_weight=model.state_dict(), optimizer_weight=optimizer.state_dict(),
                                    Epoch=epoch + 1)
                        torch.save(save, os.path.join(save_dir, f'yolox_best_mAP.pth'))
                    else:
                        torch.save(model.state_dict(), os.path.join(save_dir, f'yolox_best_mAP.pth'))
            logger.append_info('mAP', mAP)

    if (epoch + 1) % save_period == 0:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if save_optimizer:
            save = dict(model_weight=model.state_dict(), optimizer_weight=optimizer.state_dict(), Epoch=epoch + 1)
            torch.save(save, os.path.join(save_dir, f'{epoch + 1}_yolox_{round(val_loss / len(gen_val), 2)}.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(save_dir,
                                                        f'{epoch + 1}_yolox_{round(val_loss / len(gen_val), 2)}.pth'))

    logger.append_info('train_loss', round(loss / len(gen), 2))
    logger.append_info('val_loss', round(val_loss / len(gen_val), 2))
    if (epoch + 1) % save_log_period == 0:
        x_line = [x for x in range(1, epoch + 2)]
        color = [(133 / 255, 235 / 255, 207 / 255), (244 / 255, 94 / 255, 13 / 255)]
        logger.draw_picture(draw_type='x_y', save_path=f'{epoch + 1}_loss.png', x=[x_line, x_line],
                            y=['train_loss', 'val_loss'], x_label='Epoch', y_label='Loss', color=color,
                            line_style=['-', '--'], grid=True)
        logger.draw_picture(draw_type='x_y', save_path=f'{epoch + 1}_mAP.png', x=[x_line], y=['mAP'],
                            x_label='Epoch', y_label='mAP', grid=True)
        if len(email_send_to) > 0:
            for send_to in email_send_to:
                image_loss = os.path.join(logger.logger_root, f'{epoch + 1}_loss.png')
                logger.send_email(subject='Yolox Loss', send_to=send_to, image_info=image_loss)
                image_mAP = os.path.join(logger.logger_root, f'{epoch + 1}_mAP.png')
                logger.send_email(subject='Yolox mAP', send_to=send_to, image_info=image_mAP)
