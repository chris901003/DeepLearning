# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    # 這些變數會在init_distributed_mode裡面做更改，所以這裡都不用動
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    # 多gpu初始化
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ---------------------------------------------------------
    # model = 我們需要用到的預測模型
    # criterion = 預測後處理
    # postprocessors = 將輸出變成coco api想要的格式
    # ---------------------------------------------------------
    model, criterion, postprocessors = build_model(args)
    # 轉移到設備上
    model.to(device)

    # model_without_ddp = 把多gpu的那層皮拿掉
    # 如果沒有distributed我們就直接把model付給model_without_ddp
    model_without_ddp = model
    # 多gpu並行訓練的東西
    # args.distributed是在init_distributed_mode中會添加上的變數
    if args.distributed:
        # 讓多gpu之間可以溝通
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # 因為外層多包了一個DistributedDataParallel，所以model不再是以前的model，反而是在model裡面的module才是
        # DistributedDataParallel裡面有一個變數module是存放原始的model
        model_without_ddp = model.module
    # 這方法可以大略估計初要花多少時間進行訓練，可以學習一下在其他模型上面也可以提供這個資訊
    # 遍歷整個model裡面的參數，由numel得到裡面有多少參數，加總後就可以知道需要訓練的參數總共有多少
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # 把需要訓練的參數分成backbone與非backbone(transformer部分)
    # 在先前我們有設定backbone部分的lr，所以這裡有分開
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # 選擇優化器並且把需要的訓練參數丟進去
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    # 這裡採用的學習率曲線是每次固定下降lr_drop
    # 預設是每200個Epoch才會下降為原始學習率的0.1倍(gamma如果不給的話就是0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # 構建dataset，注意不是dataloader
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        # 多gpu訓練需要將dataset分成若干份
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        # 沒有多gpu就正常就可以了
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # Sampler的方法
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    # 製作出DataLoader，這裡就不會有shuffle因為用batch_sampler取代了
    # 記得要自訂collate_fn
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # Segmentation才會進來
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        # Object Detection在這裡
        # 從dataset_val中拿到coco的api，可能是後面計算mAP時會用到吧
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # 紀錄訓練的檔案位置，如果沒有要記錄會是空
    output_dir = Path(args.output_dir)
    # args.resume = 從上次的checkpoint繼續訓練
    if args.resume:
        if args.resume.startswith('https'):
            # 從網路上加載權重
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            # 從本地的權重檔案加載
            checkpoint = torch.load(args.resume, map_location='cpu')
        # 權重key['model']加載到模型裡面
        model_without_ddp.load_state_dict(checkpoint['model'])
        # 如果是自己訓練到一半的那種會有保存
        # optimizer之前得狀態，lr_scheduler的數值可以拿回當時的學習率，epoch知道那時候是到第幾個epoch
        # 當然要在訓練模式下才有意義所以先檢查是不是在訓練模式下
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # 需要加一
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        # 訓練模式就直接進來，評分完後就直接結束
        # 這裡我先不去看evaluate，我先看完train的過程再回來看
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    # 開始訓練
    print("Start training")
    # 記錄一下開始時間
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # 如果是多gpu的話需要先調用set_epoch，來重新shuffle訓練圖片
            # epoch只是當作亂數種子的一部分而已
            sampler_train.set_epoch(epoch)
        # 交給train_one_epoch進行一個epoch的訓練
        # ---------------------------------------------------------
        # model = 就是model，在多gpu下就是有經過DistributedDataParallel的model
        # criterion = 預測後處理
        # data_loader_train = 訓練集的資料
        # optimizer = 優化器
        # device = 訓練設備
        # epoch = 當前Epoch
        # clip_max_norm = gradient clipping max norm(上面args寫的，目前不確定是什麼)
        # ---------------------------------------------------------
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        # 一個epoch後對學習率調整
        lr_scheduler.step()
        # 保存訓練結果
        if args.output_dir:
            # 變成一個List且其中的元素型態為PosixPath，這裡也只會有一個就是(output_dir/checkpoint.pth)，反正就是記錄下檔案位置
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            # 多gpu我們只保存主線程上的資料，因為每個gpu上的權重都是一樣的只需要保存一個的就可以了
            for checkpoint_path in checkpoint_paths:
                # 這個回圈我們也只會跑一或兩次，看當前epoch到哪裡
                # 總共保存了，模型當前的權重，優化器權重，學習率，epoch，超參數
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # 測試一下，這裡等我看完train_one_epoch再回來看
        # ---------------------------------------------------------
        # model = 預測模型
        # criterion = 預測後處理
        # postprocessors = 將輸出變成coco api想要的格式
        # data_loader_val = 驗證集的dataloader
        # base_ds = 從dataset_val中拿到coco的api，可能是後面計算mAP時會用到吧
        # device = 指定設備
        # output_dir = 檔案輸出的位置
        # ---------------------------------------------------------
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        # 後面就是輸出狀態了，晚點看看
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
