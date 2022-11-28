import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


def train_one_epoch(model, epoch, Epoch, device, dataloader, loss_function, optimizer):
    tot_loss = 0
    acc = 0
    tot = 0
    pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch} / {Epoch}', postfix=dict, miniters=0.3)
    for iteration, (remain, remain_time) in enumerate(dataloader):
        optimizer.zero_grad()
        with torch.no_grad():
            remain = remain.to(device)
            remain_time = remain_time.to(device)
        prediction, _, _ = model(remain)
        prediction = prediction.permute((0, 2, 1))
        loss = loss_function(prediction, remain_time)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
        pred = prediction.argmax(dim=1)
        acc += pred.eq(remain_time).sum().item()
        tot += remain_time.numel()
        pbar.set_postfix(**{
            'loss': tot_loss / (iteration + 1),
            'acc': acc / tot
        })
        pbar.update(1)
    pbar.close()


def val_one_epoch(model, epoch, Epoch, device, dataloader, show, save_path):
    acc = 0
    tot = 0
    pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch} / {Epoch}', postfix=dict, miniters=0.3)
    for remain, remain_time in dataloader:
        with torch.no_grad():
            remain = remain.to(device)
            remain_time = remain_time.to(device)
            prediction, _, _ = model(remain)
        prediction = prediction.permute((0, 2, 1))
        pred = prediction.argmax(dim=1)
        acc += pred.eq(remain_time).sum().item()
        tot += remain_time.numel()
        pbar.set_postfix(**{
            'acc': acc / tot
        })
        pbar.update(1)
        if show:
            pred = pred.detach().cpu().numpy().flatten()
            correct = remain_time.cpu().numpy().flatten()
            x = remain.cpu().numpy().flatten()
            end_index = int(np.where(x == 102)[0])
            x = x[:end_index]
            pred = pred[:end_index]
            correct = correct[:end_index]
            plt.gca().invert_xaxis()
            plt.plot(x, correct, 'r-')
            plt.plot(x, pred, 'b-')
            plt.draw()
            plt.pause(0.01)
            plt.cla()
    torch.save(model.state_dict(), save_path)
