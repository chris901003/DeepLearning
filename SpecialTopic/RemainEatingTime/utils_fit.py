from tqdm import tqdm
import torch
import os


def get_lr(optimizer):
    lr = optimizer.param_groups[0]['lr']
    return lr


def fit_one_epoch(model, device, optimizer, epoch, Total_Epoch, train_dataloader, eval_dataloader, fp16, scaler,
                  save_period, save_path, training_state, save_optimizer, weight_name, logger):
    train_loss = 0
    train_acc = 0
    print('Start train')
    pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1} / {Total_Epoch}', postfix=dict, miniters=0.3)
    model = model.train()
    optimizer.zero_grad()
    for iteration, batch in enumerate(train_dataloader):
        food_remain, time_remain = batch
        with torch.no_grad():
            food_remain = food_remain.to(device)
            time_remain = time_remain.to(device)
        if not fp16:
            outputs = model(food_remain, time_remain[:, :-1], time_remain[:, 1:], with_loss=True)
            loss = outputs['loss']
            acc = outputs['acc']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            train_acc += acc.item()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model(food_remain, time_remain[:, :-1], time_remain[:, 1:], with_loss=True)
            loss = outputs['loss']
            acc = outputs['acc']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss += loss.item()
            train_acc += acc.item()
        pbar.set_postfix(**{
            'loss': train_loss / (iteration + 1),
            'acc': train_acc / (iteration + 1),
            'lr': get_lr(optimizer)
        })
        pbar.update(1)
    pbar.close()
    print('Finish train')
    if (epoch + 1) % save_period == 0:
        if save_optimizer:
            save_dict = {
                'model_weight': model.state_dict(),
                'optimizer_weight': optimizer.state_dict(),
                'last_epoch': epoch + 1
            }
        else:
            save_dict = model.state_dict()
        save_name = os.path.join(save_path, f'{weight_name}_{epoch + 1}.pth')
        torch.save(save_dict, save_name)
    print('Start validation')
    eval_loss = 0
    eval_acc = 0
    pbar = tqdm(total=len(eval_dataloader), desc=f'Epoch {epoch + 1} / {Total_Epoch}', postfix=dict, miniters=0.3)
    model.eval()
    for iteration, batch in enumerate(eval_dataloader):
        food_remain, time_remain = batch
        with torch.no_grad():
            food_remain = food_remain.to(device)
            time_remain = time_remain.to(device)
            outputs = model(food_remain, time_remain[:, :-1], time_remain[:, 1:], with_loss=True)
        loss = outputs['loss']
        acc = outputs['acc']
        eval_loss += loss.item()
        eval_acc += acc.item()
        pbar.set_postfix(**{
            'loss': eval_loss / (iteration + 1),
            'acc': eval_acc / (iteration + 1)
        })
        pbar.update(1)
    pbar.close()
    print('Epoch:' + str(epoch + 1) + '/' + str(Total_Epoch))
    print('Total Loss: %.3f || Val Loss : %.3f' % (train_loss / len(train_dataloader),
                                                   eval_loss / len(eval_dataloader)))
    print('Total Acc: %.3f || Val Acc : %.3f' % (train_acc / len(train_dataloader), eval_acc / len(eval_dataloader)))
    if training_state['eval_loss'] > (eval_loss / len(eval_dataloader)):
        if save_optimizer:
            save_dict = {
                'model_weight': model.state_dict(),
                'optimizer_weight': optimizer.weight(),
                'last_epoch': epoch + 1
            }
        else:
            save_dict = model.state_dict()
        save_name = os.path.join(save_path, f'{weight_name}_eval.pth')
        torch.save(save_dict, save_name)
    training_state['train_loss'] = min(training_state['train_loss'], (train_loss / len(train_dataloader)))
    training_state['eval_loss'] = min(training_state['eval_loss'], (eval_loss / len(eval_dataloader)))
    print(f'Less train loss: {training_state["train_loss"]}')
    print(f'Less eval loss: {training_state["eval_loss"]}')
    logger.append_info('train_loss', train_loss / len(train_dataloader))
    logger.append_info('train_acc', train_acc / len(train_dataloader))
    logger.append_info('val_loss', eval_loss / len(eval_dataloader))
    logger.append_info('val_acc', eval_acc / len(eval_dataloader))
