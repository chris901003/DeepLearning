import os
import torch
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from .utils_bbox import decode_outputs, non_max_suppression
import numpy as np


def run(model, yolo_loss, device, work_dir, train_epoch, train_dataloader, optimizer=None, loss_function=None,
        val_epoch=None, val_dataloader=None, val_coco_json=None, scaler=None):
    val_one_epoch(model, device, val_dataloader, 1, val_coco_json)
    if val_epoch is not None:
        assert val_dataloader is not None, '使用驗證模式需要提供驗證資料集'
        assert val_coco_json is not None, '無法計算mAP值'
        assert os.path.splitext(val_coco_json)[1] == '.json', '需提供json格式的coco標註檔'
    best_loss = 1000000
    for epoch in range(1, train_epoch + 1):
        avg_loss = train_one_epoch(model, yolo_loss, device, train_dataloader, optimizer, epoch, scaler=scaler)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(work_dir, 'best_model.pkt'))
        if val_epoch is not None and (epoch % val_epoch == 0):
            val_one_epoch(model, device, val_dataloader, epoch, val_coco_json)


def train_one_epoch(model, yolo_loss, device, dataloader, optimizer, epoch, scaler=None):
    model.train()
    total_loss = 0
    total_picture = 0
    avg_loss = 0
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch}: ',
              postfix=f'Avg loss {avg_loss}', mininterval=1) as pbar:
        for imgs, gt_bboxes, gt_labels in dataloader:
            total_picture += imgs.shape[0]
            gt_bboxes = [bboxes.to(device) for bboxes in gt_bboxes]
            gt_labels = [labels.to(device) for labels in gt_labels]
            targets = [torch.stack([torch.cat((box, label.unsqueeze(dim=-1)), dim=-1) for box, label in
                                    zip(gt_bboxes[idx], gt_labels[idx])])
                       for idx in range(len(gt_bboxes))]
            imgs = imgs.to(device)
            optimizer.zero_grad()
            if scaler is None:
                outputs = model(imgs)
                loss_value, loss_dict = yolo_loss(outputs, targets)
                loss_value.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = model(imgs)
                    loss_value, loss_dict = yolo_loss(outputs, targets)
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss_value.item()
            avg_loss = total_loss / total_picture
            pbar.set_postfix_str(f'Avg loss => {avg_loss}')
            pbar.update(1)
    return avg_loss


def val_one_epoch(model, device, dataloader, epoch, val_coco_json):
    model.eval()
    json_file = './coco_mAP_file.json'
    res = list()
    with torch.no_grad():
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch}: ', mininterval=1) as pbar:
            for imgs, scale_factor, images_path in dataloader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                outputs = decode_outputs(outputs, imgs.shape[-2:])
                results = non_max_suppression(outputs, 9, imgs.shape[-2:], imgs.shape[-2:], False, conf_thres=0.5,
                                              nms_thres=0.3)
                top_label = np.array(results[0][:, 6], dtype=np.int)
                top_label = top_label.tolist()
                top_conf = results[0][:, 4] * results[0][:, 5]
                top_conf = top_conf.tolist()
                top_boxes = results[0][:, :4]
                top_boxes = top_boxes.tolist()
                import cv2
                from PIL import Image
                image = cv2.imread(images_path[0])
                for labels, confs, boxes in zip(top_label, top_conf, top_boxes):
                    image_id = int(os.path.splitext(os.path.basename(images_path[0]))[0])
                    data = dict(image_id=image_id, category_id=labels, bbox=boxes, score=confs)
                    xmin, ymin, xmax, ymax = int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    res.append(data)
                pbar.update(1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image.show()
    print('Writing json file ...')
    with open(json_file, 'w') as f:
        json.dump(res, f)
    cocoGt = COCO(val_coco_json)
    cocoDt = cocoGt.loadRes(json_file)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    os.remove(json_file)
