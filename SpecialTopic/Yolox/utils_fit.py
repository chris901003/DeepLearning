import os
import torch
from tqdm import tqdm
from utils import get_lr


def fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch,
                  cuda, fp16, scaler, save_period, save_dir, local_rank=0):
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
    torch.save(model.state_dict(), './best_weight.pth')
