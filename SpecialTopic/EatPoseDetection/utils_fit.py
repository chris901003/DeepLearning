import os
from tqdm import tqdm
import torch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(model, device, optimizer, epoch, train_dataloader, eval_dataloader, Epoch, fp16, scaler, save_period,
                  save_dir, eval_period):
    loss = 0
    total_acc = 0
    total_topk_acc = 0
    print('Start Train')
    pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, miniters=0.3)
    model = model.train()
    for iteration, batch in enumerate(train_dataloader):
        imgs, labels = batch[0], batch[1]
        with torch.no_grad():
            imgs = imgs.to(device)
            labels = labels.to(device)
        optimizer.zero_grad()
        if not fp16:
            outputs = model(imgs, labels)
            loss_value = outputs['loss']
            acc = outputs['acc']
            topk_acc = outputs['topk_acc']
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(imgs, labels)
            loss_value = outputs['loss']
            acc = outputs['acc']
            topk_acc = outputs['topk_acc']
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        loss += loss_value.item()
        total_acc += acc.item()
        total_topk_acc += topk_acc.item()
        pbar.set_postfix(**{
            'loss': loss / (iteration + 1),
            'acc': total_acc / (iteration + 1),
            'top5 acc': total_topk_acc / (iteration + 1),
            'lr': get_lr(optimizer)
        })
        pbar.update(1)
    pbar.close()
    print('Finish Train')
    if (epoch + 1) % save_period == 0:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1
        }
        filename = f'{epoch + 1}_{round(total_acc / len(train_dataloader), 2)}.pth'
        torch.save(save_dict, os.path.join(save_dir, filename))
    if eval_dataloader is None or (epoch + 1) % eval_period != 0:
        return
    eval_loss = 0
    eval_total_acc = 0
    eval_total_topk_acc = 0
    print('Start Validation')
    pbar = tqdm(total=len(eval_dataloader), desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, miniters=0.3)
    model = model.eval()
    for iteration, batch in enumerate(eval_dataloader):
        imgs, labels = batch[0], batch[1]
        with torch.no_grad():
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs, labels, mode='val')
            loss_value = outputs['loss']
            acc = outputs['acc']
            topk_acc = outputs['topk_acc']
            eval_loss += loss_value.item()
            eval_total_acc += acc.item()
            eval_total_topk_acc += topk_acc.item()
        pbar.set_postfix(**{
            'loss': eval_loss / (iteration + 1),
            'acc': eval_total_acc / (iteration + 1),
            'top5 acc': eval_total_topk_acc / (iteration + 1),
        })
        pbar.update(1)
    pbar.clear()
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss : %.3f' % (loss / len(train_dataloader), eval_loss / len(eval_dataloader)))
    print('Total Acc: %.3f || Val Acc : %.3f' % (total_acc / len(train_dataloader),
                                                 eval_total_acc / len(eval_dataloader)))
