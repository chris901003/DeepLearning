import os
import torch
from tqdm import tqdm


def run(model, device, work_dir, train_epoch, train_dataloader, optimizer=None, loss_function=None,
        val_epoch=None, val_dataloader=None):
    if val_epoch is not None:
        assert val_dataloader is not None, '使用驗證模式需要提供驗證資料集'
    best_loss = 1000000
    for epoch in range(1, train_epoch + 1):
        avg_loss = train_one_epoch(model, device, train_dataloader, epoch, optimizer, loss_function)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(work_dir, 'best_model.pkt'))
        if val_epoch is not None and (epoch % val_epoch == 0):
            val_one_epoch(model, device, val_dataloader, epoch)


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


def val_one_epoch(model, device, dataloader, epoch):
    pass
