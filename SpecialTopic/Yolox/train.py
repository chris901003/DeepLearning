import argparse
import os
from torch.utils.data import DataLoader
import torch
from utils import get_classes
from net.yolo import YoloBody
from net.yolo_training import weights_init, YOLOLoss, get_lr_scheduler, set_optimizer_lr
import numpy as np
from torch.backends import cudnn
from torch import nn, optim
from dataset import build_dataset, custom_collate_fn
from dataloader import YoloDataset
from utils_fit import fit_one_epoch


def train():
    Cuda = opt.Cuda
    distributed = False
    sync_bn = False
    fp16 = False
    classes_path = '/Users/huanghongyan/Downloads/food_data_flag/classes.txt'
    models_path = opt.models_path
    input_shape = [640, 640]
    phi = opt.phi
    mosaic = True
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = opt.batch_size
    UnFreeze_Epoch = 300
    Unfreeze_batch_size = opt.batch_size // 2
    Freeze_Train = True
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    optimizer_type = 'sgd'
    momentum = 0.937
    weight_decay = 5e-4
    lr_decay_type = 'cos'
    save_period = 10
    save_dir = 'logs'
    eval_flag = True
    eval_period = 10
    num_workers = 4
    train_annotation_path = '/Users/huanghongyan/Downloads/food_data_flag/2012_train.txt'
    val_annotation_path = '/Users/huanghongyan/Downloads/food_data_flag/2012_val.txt'
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        raise NotImplementedError
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0
    class_names, num_classes = get_classes(classes_path)
    model = YoloBody(num_classes, phi)
    weights_init(model)
    if models_path != 'none':
        if local_rank == 0:
            print(f'Load weights {models_path}')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(models_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    yolo_loss = YOLOLoss(num_classes, fp16)
    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        raise NotImplementedError
    elif sync_bn:
        raise ValueError('sync_bn在單卡時無法使用')
    if Cuda:
        if distributed:
            raise NotImplementedError
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        with open(train_annotation_path, encoding='utf-8') as f:
            train_lines = f.readlines()
        with open(val_annotation_path, encoding='utf-8') as f:
            val_lines = f.readlines()
        num_train = len(train_lines)
        num_val = len(val_lines)
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size
        if distributed:
            raise NotImplementedError
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
        train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch,
                                    mosaic=mosaic, train=True)
        dataloader_cfg = {
            'dataset': train_dataset,
            'batch_size': batch_size,
            'shuffle': True,
            'num_workers': 1,
            'pin_memory': True,
            'drop_last': False,
            'collate_fn': custom_collate_fn
        }
        train_dataloader = DataLoader(**dataloader_cfg)
        val_dataloader = None
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch > Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                for param in model.backbone.parameters():
                    param.requires_grad = True
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError('Training data to small')
                if distributed:
                    raise NotImplementedError
                dataloader_cfg['batch_size'] = batch_size
                train_dataloader = DataLoader(**dataloader_cfg)
                UnFreeze_flag = True
            train_dataloader.dataset.epoch_now = epoch
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, epoch_step, epoch_step_val, train_dataloader,
                          val_dataloader, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--models-path', type=str, default='/Users/huanghongyan/Downloads/yolox_l.pth')
    parser.add_argument('--phi', type=str, default='l')
    parser.add_argument('--batch-size', type=int, default=2)
    opt = parser.parse_args()
    train()
