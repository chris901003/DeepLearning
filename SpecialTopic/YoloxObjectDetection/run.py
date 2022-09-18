import os
import torch
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def run(model, device, work_dir, train_epoch, train_dataloader, optimizer=None, loss_function=None,
        val_epoch=None, val_dataloader=None, val_coco_json=None):
    val_one_epoch(model, device, val_dataloader, 1, val_coco_json)
    if val_epoch is not None:
        assert val_dataloader is not None, '使用驗證模式需要提供驗證資料集'
        assert val_coco_json is not None, '無法計算mAP值'
        assert os.path.splitext(val_coco_json)[1] == '.json', '需提供json格式的coco標註檔'
    best_loss = 1000000
    for epoch in range(1, train_epoch + 1):
        avg_loss = train_one_epoch(model, device, train_dataloader, epoch, optimizer, loss_function)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(work_dir, 'best_model.pkt'))
        if val_epoch is not None and (epoch % val_epoch == 0):
            val_one_epoch(model, device, val_dataloader, epoch, val_coco_json)


def train_one_epoch(model, device, dataloader, epoch, optimizer=None, loss_function=None):
    model.train()
    total_loss = 0
    total_picture = 0
    avg_loss = 0
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch}: ',
              postfix=f'Avg loss {avg_loss}', mininterval=1) as pbar:
        for imgs, gt_bboxes, gt_labels in dataloader:
            if optimizer is not None:
                optimizer.zero_grad()
            total_picture += imgs.shape[0]
            imgs = imgs.to(device)
            loss = model(imgs, gt_bboxes, gt_labels)
            if loss_function is not None:
                loss = loss_function(loss, gt_labels)
                loss.backward()
                optimizer.step()
            total_loss += loss
            avg_loss = total_loss / total_picture
            pbar.set_postfix_str(f'Avg loss => {avg_loss}')
            pbar.update(1)
    return avg_loss


def val_one_epoch(model, device, dataloader, epoch, val_coco_json):
    model.eval()
    json_file = './coco_mAP_file.json'
    res = list()
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch}: ', mininterval=1) as pbar:
        for imgs, scale_factor, images_path in dataloader:
            imgs = imgs.to(device)
            results = model(imgs, scale_factor=scale_factor, return_loss=False)
            for image_path, result in zip(images_path, results):
                image_id = int(os.path.splitext(os.path.basename(image_path))[0])
                for cls in range(len(result)):
                    for box in result[cls]:
                        tmp = box.tolist()
                        bbox = [tmp[0], tmp[1], tmp[2] - tmp[0], tmp[3] - tmp[1]]
                        data = dict(image_id=image_id, category_id=cls, bbox=bbox, score=tmp[4])
                        res.append(data)
            pbar.update(1)
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
