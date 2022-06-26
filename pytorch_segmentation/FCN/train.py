import os
import time
import datetime

import torch

from src import fcn_resnet50
from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import VOCSegmentation
import transforms as T


class SegmentationPresetTrain:
    # 由下方實例化，用在train資料集上的資料處理
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        :param base_size: 基礎圖片大小
        :param crop_size: 最後輸入網路圖片大小
        :param hflip_prob: 水平翻轉概率
        :param mean: 調整均值
        :param std: 調整方差
        """
        # 已看過
        # 最小尺寸與最大尺寸
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        # 實例化RandomResize，並且之後會將一系列轉換都放入list當中，一次進行forward
        trans = [T.RandomResize(min_size, max_size)]
        # 如果有概率進行水平翻轉就實例化水平翻轉操作
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        # 在list當中加上RandomCrop、ToTensor以及Normalize方法的實例化
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        # 已看過
        # 經過transforms中變化方式的__call__函數
        return self.transforms(img, target)


class SegmentationPresetEval:
    # 由下方實例化，用在val資料集上的資料處理
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 已看過
        # 這裡只傳入要輸入網路的圖片大小

        # 也是一系列的圖像變換，調整大小、轉成tensor以及標準化
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        # 已看過
        # 經過transforms中變化方式的__call__函數
        return self.transforms(img, target)


def get_transform(train):
    # 已看過
    # train與val會有不同的圖像預處理方式
    base_size = 520
    crop_size = 480

    return SegmentationPresetTrain(base_size, crop_size) if train else SegmentationPresetEval(base_size)


def create_model(aux, num_classes, pretrain=True):
    """
    :param aux: 是否啟用輔助分類
    :param num_classes: 分類類別數量
    :param pretrain: 是否使用預訓練權重
    :return:
    """
    # 構建model這裡用的是resnet50，傳入是否啟用輔助分類以及分類數量
    model = fcn_resnet50(aux=aux, num_classes=num_classes)

    # 加載預訓練權重
    if pretrain:
        weights_dict = torch.load("./fcn_resnet50_coco.pth", map_location='cpu')

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        # 加載預訓練權重，並且輸出沒有匹配到的權重
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


def main(args):
    # 已看過
    # 偵測使用設備，有gpu就用沒有就用cpu訓練
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 獲取batch_size
    batch_size = args.batch_size
    # segmentation nun_classes + background
    # 將背景也到分類數量中
    num_classes = args.num_classes + 1

    # 用来保存训练以及验证过程中信息，這裡是幫文件取名字
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> train.txt
    # 建立訓練資料集
    train_dataset = VOCSegmentation(args.data_path,
                                    year="2012",
                                    transforms=get_transform(train=True),
                                    txt_name="train.txt")

    # 建立驗證資料集
    # VOCdevkit -> VOC2012 -> ImageSets -> Segmentation -> val.txt
    val_dataset = VOCSegmentation(args.data_path,
                                  year="2012",
                                  transforms=get_transform(train=False),
                                  txt_name="val.txt")

    # 根據batch_size以及cpu核心數決定num_workers
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # 建立data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    # 創建模型，帶入是否要輔助輸出以及分類類別數
    model = create_model(aux=args.aux, num_classes=num_classes)
    # 轉移到設備上
    model.to(device)

    # 找出模型中哪些部分是需要訓練的
    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    # 找出模型中哪些部分是需要訓練的
    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    # 優化器
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # 如果有啟用amp就會使用到scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    # 傳入優化器，一個epoch有幾個batch，總共多少個epoch，是否要使用暖身訓練
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    # 如果有傳入之前訓練到哪裡，就會在這裡進行加載
    if args.resume:
        # 加載之前訓練的紀錄
        checkpoint = torch.load(args.resume, map_location='cpu')
        # 加載模型權重
        model.load_state_dict(checkpoint['models'])
        # 加載優化器狀態
        optimizer.load_state_dict(checkpoint['optimizer'])
        # 加載學習率
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # 更新要從哪個epoch開始
        args.start_epoch = checkpoint['epoch'] + 1
        # 如果有啟用amp就要加載scaler
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # 開始計算訓練時間
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 訓練一個epoch
        # mean_loss = 平均損失，lr = 當前學習率
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        # 進行驗證，回傳的是混淆矩陣實例對象
        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        # 將裡面的內容變成string格式
        val_info = str(confmat)
        # 打印出結果
        print(val_info)
        # write into txt
        with open(results_file, "a") as f:
            # 记录每个epoch对应的train_loss、lr以及验证集各指标
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        # 保存pth檔
        save_file = {"models": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, "save_weights/model_{}.pth".format(epoch))

    # 計算花費時間
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    # 已看過
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")

    # 資料集保存位置
    parser.add_argument("--data-path", default="/data/", help="VOCdevkit root")
    # 分類數目(不包含背景)
    parser.add_argument("--num-classes", default=20, type=int)
    # 是否啟用輔助分類
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    # 訓練設備
    parser.add_argument("--device", default="cuda", help="training device")
    # 訓練的batch_size大小
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    # 總共訓練幾個epoch
    parser.add_argument("--epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")

    # 初始學習率
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    # 優化器的動量大小
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # 優化器中的其他超參數
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 多少個batch會打印一次當前狀態
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    # 可以回到之前訓練到一半的狀態
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    # 起始epoch數
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # Mixed precision training parameters
    # 使用顯卡中的雙精度計算
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # 已看過
    # 接收執行時的參數
    args = parse_args()

    # 保存訓練資料的資料夾，如果資料夾不存在就建立一個新的
    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args)
