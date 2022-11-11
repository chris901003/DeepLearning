import argparse
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from SpecialTopic.ST.net.lr_scheduler import get_lr_scheduler_yolox, set_optimizer_lr_yolox
from SpecialTopic.ST.build import build_detector
from DatasetSource import cifar100_dataset_from_official
from dataset import Cifar100Dataset
import os
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser('使用分類網路進行剩餘量判斷')
    # 選擇使用的模型主幹，目前支援[ResNet, VIT, MobileVit]
    parser.add_argument('--model-type', type=str, default='VIT')
    # 使用的模型大小，這裡支援的尺寸會與使用的模型主幹有關
    parser.add_argument('--phi', type=str, default='l')
    # batch size，盡量調整到超過4，這樣BN層才不會出問題
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--Freeze-batch-size', type=int, default=512)
    # 加載預訓練權重，這裡指的是在ImageNet上預訓練權重
    parser.add_argument('--pretrained', type=str, default=r'C:\Checkpoint\VIT\vit_l.pth')

    parser.add_argument('--num-classes', type=int, default=100)
    parser.add_argument('--auto-fp16', action='store_false')

    # 起始Epoch數
    parser.add_argument('--Init-Epoch', type=int, default=0)
    # 在多少個Epoch前會將骨幹凍結，這邊建議訓練VIT時可以進行凍結
    parser.add_argument('--Freeze-Epoch', type=int, default=0)
    # 總共要訓練多少個Epoch
    parser.add_argument('--Total-Epoch', type=int, default=100)
    # 最大學習率
    parser.add_argument('--Init-lr', type=float, default=1e-3)
    # 指定使用優化器類別
    parser.add_argument('--optimizer-type', type=str, default='sgd')
    # 學習率下降曲線
    parser.add_argument('--lr-decay-type', type=str, default='cos')
    # 在使用DataLoader時的cpu數
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()
    return args


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, device, optimizer, epoch, train_dataloader, Total_Epoch, scaler):
    train_loss = 0
    train_acc = 0
    train_topk_acc = 0
    print('Start train')
    pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {epoch}/{Total_Epoch}', postfix=dict, miniters=0.3)
    model = model.train()
    for iteration, batch in enumerate(train_dataloader):
        images, labels = batch
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
        optimizer.zero_grad()
        if scaler is None:
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


def val(model, device, epoch, val_dataloader, Total_Epoch):
    eval_loss = 0
    eval_acc = 0
    eval_topk_acc = 0
    pbar = tqdm(total=len(val_dataloader), desc=f'Epoch {epoch}/{Total_Epoch}', postfix=dict, miniters=0.3)
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
    pbar.close()


def predict_answer():
    from PIL import Image
    args = parse_args()
    folder_path = './Training_data/1'
    pretrained = './resnet50.pth'
    answer_file = './410985048.txt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    support_image_format = ['.png']
    images_name = [image_name for image_name in os.listdir(folder_path)
                   if os.path.splitext(image_name)[1] in support_image_format]
    images_info = {int(os.path.splitext(image_name)[0]): image_name for image_name in images_name}
    images_info = sorted(images_info.items())
    transforms_data = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model_cfg = {
        'type': args.model_type,
        'phi': args.phi,
        'num_classes': args.num_classes,
        'pretrained': args.pretrained
    }
    model = build_detector(model_cfg)
    model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    model.eval()
    model = model.to(device)
    results = list()
    for _, image_name in tqdm(images_info):
        image = Image.open(os.path.join(folder_path, image_name))
        image = transforms_data(image)
        image = image.unsqueeze(dim=0)
        with torch.no_grad():
            image = image.to(device)
            preds = model(image)
        preds = preds.squeeze(dim=0)
        preds = preds.argmax().item()
        info = os.path.splitext(image_name)[0] + ' ' + str(preds)
        results.append(info)
    with open(answer_file, 'w') as f:
        for result in results:
            f.write(result)
            f.write('\n')
    print(f'Total {len(results)} pictures')


def main():
    args = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_classes = args.num_classes
    fp16 = torch.cuda.is_available() if args.auto_fp16 else False
    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    model_cfg = {
        'type': args.model_type,
        'phi': args.phi,
        'num_classes': num_classes,
        'pretrained': args.pretrained
    }
    model = build_detector(model_cfg)
    model = model.to(device)
    batch_size = args.batch_size
    freeze_batch_size = args.Freeze_batch_size
    if args.Init_Epoch < args.Freeze_Epoch:
        for param in model.backbone.parameters():
            param.requires_grad = False
        batch_size = freeze_batch_size
    # transform_data = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # train_dataset = cifar100_dataset_from_official(root='cifar100', train=True, download=True,
    #                                                transform_data=transform_data)
    # eval_dataset = cifar100_dataset_from_official(root='cifar100', train=False, download=True,
    #                                               transform_data=transform_data)
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True,
    #                               num_workers=args.num_workers)
    # eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
    #                              num_workers=args.num_workers)
    train_dataset = Cifar100Dataset(annotation_path='./train_annotation.txt')
    eval_dataset = Cifar100Dataset(annotation_path='./train_annotation.txt')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)

    Init_lr = args.Init_lr
    Min_lr = Init_lr * 0.01
    nbs = 1024
    lr_limit_max = 1e-3 if args.optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if args.optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    Init_lr_fit = Init_lr
    Min_lr_fit = Min_lr
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or 'bn' in k:
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {
        'adam': torch.optim.Adam(pg0, Init_lr_fit, betas=(0.937, 0.999)),
        'sgd': torch.optim.SGD(pg0, Init_lr_fit, momentum=0.937, nesterov=True)
    }[args.optimizer_type]
    optimizer.add_param_group({'params': pg1, 'weight_decay': 5e-4})
    optimizer.add_param_group({'params': pg2})
    lr_scheduler_func = get_lr_scheduler_yolox(args.lr_decay_type, Init_lr_fit, Min_lr_fit, args.Total_Epoch)
    for epoch in range(1, args.Total_Epoch + 1):
        if epoch == args.Freeze_Epoch:
            for param in model.backbone.parameters():
                param.requires_grad = True
            batch_size = args.batch_size
            train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True,
                                          num_workers=args.num_workers)
            eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                         num_workers=args.num_workers)
        set_optimizer_lr_yolox(optimizer, lr_scheduler_func, epoch)
        train(model, device, optimizer, epoch, train_dataloader, args.Total_Epoch, scaler)
        val(model, device, epoch, eval_dataloader, args.Total_Epoch)
        torch.save(model.state_dict(), './vit.pth')


if __name__ == '__main__':
    main()
    print('Finish')

