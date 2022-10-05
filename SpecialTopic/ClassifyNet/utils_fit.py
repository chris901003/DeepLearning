import torch
import os
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(model, device, optimizer, epoch, train_dataloader, val_dataloader, Total_Epoch, fp16, scaler,
                  save_period, save_path, training_state, best_train_loss, best_val_loss, save_optimizer, weight_name,
                  logger, email_send_to, save_log_period):
    train_loss = 0
    train_acc = 0
    train_topk_acc = 0
    print('Start train')
    pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{Total_Epoch}', postfix=dict, miniters=0.3)
    model = model.train()
    for iteration, batch in enumerate(train_dataloader):
        images, labels = batch
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
        optimizer.zero_grad()
        if not fp16:
            outputs = model(images, labels, with_loss=True)
            loss = outputs['loss']
            acc = outputs['acc']
            topk_acc = outputs['topk_acc']
            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(images, labels)
            loss = outputs['loss']
            acc = outputs['acc']
            topk_acc = outputs['topk_acc']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        train_loss += loss.item()
        train_acc += acc.item()
        train_topk_acc += topk_acc.item()
        pbar.set_postfix(**{
            'loss': train_loss / (iteration + 1),
            'acc': train_acc / (iteration + 1),
            'topk acc': train_topk_acc / (iteration + 1),
            'lr': get_lr(optimizer)
        })
        pbar.update(1)
    pbar.close()
    print('Finish train')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if (epoch + 1) % save_period == 0:
        if save_optimizer:
            save_dict = {
                'model_weight': model.state_dict(),
                'optimizer_weight': optimizer.state_dict(),
                'epoch': epoch + 1
            }
        else:
            save_dict = model.state_dict()
        save_name = os.path.join(save_path, f'{weight_name}_{epoch + 1}.pth')
        torch.save(save_dict, save_name)
    print('Start validation')
    eval_loss = 0
    eval_acc = 0
    eval_topk_acc = 0
    pbar = tqdm(total=len(val_dataloader), desc=f'Epoch {epoch + 1}/{Total_Epoch}', postfix=dict, miniters=0.3)
    model = model.eval()
    for iteration, batch in enumerate(val_dataloader):
        images, labels = batch
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, labels, with_loss=True)
            loss = outputs['loss']
            acc = outputs['acc']
            topk_acc = outputs['topk_acc']
            eval_loss += loss.item()
            eval_acc += acc.item()
            eval_topk_acc += topk_acc.item()
        pbar.set_postfix(**{
            'loss': eval_loss / (iteration + 1),
            'acc': eval_acc / (iteration + 1),
            'topk acc': eval_topk_acc / (iteration + 1)
        })
        pbar.update(1)
    pbar.clear()
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Total_Epoch))
    print('Total Loss: %.3f || Val Loss : %.3f' % (train_loss / len(train_dataloader), eval_loss / len(val_dataloader)))
    print('Total Acc: %.3f || Val Acc : %.3f' % (train_acc / len(train_dataloader), eval_acc / len(val_dataloader)))
    if best_train_loss and training_state['train_loss'] > (train_loss / len(train_dataloader)):
        if save_optimizer:
            save_dict = {
                'model_weight': model.state_dict(),
                'optimizer_weight': optimizer.state_dict(),
                'epoch': epoch + 1
            }
        else:
            save_dict = model.state_dict()
        save_name = os.path.join(save_path, f'{weight_name}_best_train.pth')
        torch.save(save_dict, save_name)
    if best_val_loss and training_state['val_loss'] > (eval_loss / len(val_dataloader)):
        if save_optimizer:
            save_dict = {
                'model_weight': model.state_dict(),
                'optimizer_weight': optimizer.state_dict(),
                'epoch': epoch + 1
            }
        else:
            save_dict = model.state_dict()
        save_name = os.path.join(save_path, f'{weight_name}_eval.pth')
        torch.save(save_dict, save_name)
    training_state['train_loss'] = min(training_state['train_loss'], (train_loss / len(train_dataloader)))
    training_state['val_loss'] = min(training_state['val_loss'], (eval_loss / len(val_dataloader)))
    print(f'Less train loss: {training_state["train_loss"]}')
    print(f'Less eval loss: {training_state["val_loss"]}')
    logger.append_info('train_loss', train_loss / len(train_dataloader))
    logger.append_info('train_acc', train_acc / len(train_dataloader))
    logger.append_info('train_topk_acc', train_topk_acc / len(train_dataloader))
    logger.append_info('val_loss', eval_loss / len(val_dataloader))
    logger.append_info('val_acc', eval_acc / len(val_dataloader))
    logger.append_info('val_topk_acc', eval_topk_acc / len(val_dataloader))
    if (epoch + 1) % save_log_period == 0:
        x_line = [x for x in range(1, epoch + 2)]
        color = [(133 / 255, 235 / 255, 207 / 255), (244 / 255, 94 / 255, 13 / 255)]
        logger.draw_picture(draw_type='x_y', save_path=f'{epoch + 1}_loss.png', x=[x_line, x_line],
                            y=['train_loss', 'val_loss'], x_label='Epoch', y_label='Loss', color=color,
                            line_style=['-', '--'], grid=True)
        logger.draw_picture(draw_type='x_y', save_path=f'{epoch + 1}_acc.png', x=[x_line, x_line],
                            y=['train_acc', 'val_acc'], x_label='Epoch', y_label='Acc', color=color,
                            line_style=['-', '--'], grid=True)
        if len(email_send_to) > 0:
            for send_to in email_send_to:
                image_loss = os.path.join(logger.logger_root, f'{epoch + 1}_loss.png')
                logger.send_email(subject='Classify Net Loss', send_to=send_to, image_info=image_loss)
                image_acc = os.path.join(logger.logger_root, f'{epoch + 1}_acc.png')
                logger.send_email(subject='Classify Net Acc', send_to=send_to, image_info=image_acc)
