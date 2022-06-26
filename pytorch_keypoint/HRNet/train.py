import json
import os
import datetime

import torch
import numpy as np

import transforms
from models import HighResolutionNet
from my_dataset_coco import CocoKeypoint
from train_utils import train_eval_utils as utils


def create_model(num_joints, load_pretrain_weights=True):
    """
    :param num_joints: 關節點數量
    :param load_pretrain_weights: 是否使用預訓練權重
    :return:
    """
    # 已看過
    # 傳入基礎channel深度以及關節點數量
    model = HighResolutionNet(base_channel=32, num_joints=num_joints)
    
    if load_pretrain_weights:
        # 载入预训练模型权重
        # 链接:https://pan.baidu.com/s/1Lu6mMAWfm_8GGykttFMpVw 提取码:f43o
        # 先載入成dict格式
        weights_dict = torch.load("./hrnet_w32.pth", map_location='cpu')

        for k in list(weights_dict.keys()):
            # 如果载入的是imagenet权重，就删除无用权重
            if ("head" in k) or ("fc" in k):
                del weights_dict[k]

            # 如果载入的是coco权重，对比下num_joints，如果不相等就删除
            # 如果資料集不是coco就建議刪除最後輸出層
            if "final_layer" in k:
                if weights_dict[k].shape[0] != num_joints:
                    del weights_dict[k]

        # 載入權重
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        # 輸出沒有載入的部分
        if len(missing_keys) != 0:
            print("missing_keys: ", missing_keys)

    # 返回模型
    return model


def main(args):
    # 已看過
    # 檢測訓練設備
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 讀取人物關鍵點檔案，這裡不是coco的annotation檔案
    with open(args.keypoints_path, "r") as f:
        person_kps_info = json.load(f)

    # fixed_size預設為[256, 192]
    fixed_size = args.fixed_size
    # heatmap_hw = [64, 48]
    heatmap_hw = (args.fixed_size[0] // 4, args.fixed_size[1] // 4)
    # 讀取每一個關鍵點的權重，在計算損失時每個關鍵的損失權重有不同
    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((args.num_joints,))
    # 資料轉換方式
    data_transform = {
        # 訓練集調整方式
        "train": transforms.Compose([
            # 傳入上半身的index以及下半身的index
            transforms.HalfBody(0.3, person_kps_info["upper_body_ids"], person_kps_info["lower_body_ids"]),
            # 將圖像進行縮放以及旋轉還有最後輸入的大小
            transforms.AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=fixed_size),
            # 水平翻轉，傳入資料有左右對應的index，例如左手以及右手會是一組，還有翻轉概率
            transforms.RandomHorizontalFlip(0.5, person_kps_info["flip_pairs"]),
            # 傳入熱力圖的大小以及超參數以及關鍵點權重
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            # 將圖像轉換成tensor格式
            transforms.ToTensor(),
            # 將圖像標準化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        # 驗證集調整方式
        "val": transforms.Compose([
            # 縮放
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
            # 轉成tensor格式
            transforms.ToTensor(),
            # 標準化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    # coco數據集檔案位置
    data_root = args.data_path

    # load train data set
    # coco2017 -> annotations -> person_keypoints_train2017.json
    # 製作訓練集，傳入coco檔案位置模式轉換方式以及輸入圖像大小
    train_dataset = CocoKeypoint(data_root, "train", transforms=data_transform["train"], fixed_size=args.fixed_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    # nw設定越大讀取檔案會越快但是記憶體消耗會更大
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # 構建dataloader，這裡的collate_fn是自定的
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    pin_memory=True,
                                                    num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn)

    # load validation data set
    # coco2017 -> annotations -> person_keypoints_val2017.json
    # 構建驗證集，基本上跟上面的訓練集是一樣的
    val_dataset = CocoKeypoint(data_root, "val", transforms=data_transform["val"], fixed_size=args.fixed_size,
                               det_json_path=args.person_det)
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=val_dataset.collate_fn)

    # create models
    # 創建模型，帶入參數為關節點數量，coco預設數量為17
    model = create_model(num_joints=args.num_joints)
    # print(models)

    # 將模型放到設備上
    model.to(device)

    # define optimizer
    # 將需要訓練的參數拿出來
    params = [p for p in model.parameters() if p.requires_grad]
    # 設定優化器同時將需要訓練的參數放進去
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    # 如果有設定amp就會設定scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    # 這裡使用的是到設定的epoch就會將當前學習率乘上gamma
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        # 載入權重
        checkpoint = torch.load(args.resume, map_location='cpu')
        # 設定模型權重
        model.load_state_dict(checkpoint['models'])
        # 優化器權重
        optimizer.load_state_dict(checkpoint['optimizer'])
        # 學習率權重
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # 上次到哪個epoch中斷的
        args.start_epoch = checkpoint['epoch'] + 1
        # 如果有使用amp就需要載入amp
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    # 保存訓練過程的資料
    train_loss = []
    learning_rate = []
    val_map = []

    # 開始訓練
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        # 訓練一個epoch
        # mean_loss = 一個epoch中平均損失，lr = 當前學習率
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        # 將結果放到list當中
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # 更新學習率
        lr_scheduler.step()

        # evaluate on the test dataset
        # 進入驗證模式
        coco_info = utils.evaluate(model, val_data_set_loader, device=device,
                                   flip=True, flip_pairs=person_kps_info["flip_pairs"])

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # @0.5 mAP

        # save weights
        save_files = {
            'models': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "./save_weights/models-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(coco2017)
    parser.add_argument('--data-path', default='/data/coco2017', help='dataset')
    # COCO数据集人体关键点信息
    parser.add_argument('--keypoints-path', default="./person_keypoints.json", type=str,
                        help='person_keypoints.json path')
    # 原项目提供的验证集person检测信息，如果要使用GT信息，直接将该参数置为None，建议设置成None
    parser.add_argument('--person-det', type=str, default=None)
    parser.add_argument('--fixed-size', default=[256, 192], nargs='+', type=int, help='input size')
    # keypoints点数
    parser.add_argument('--num-joints', default=17, type=int, help='num_joints')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=210, type=int, metavar='N',
                        help='number of total epochs to run')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[170, 200], nargs='+', type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 学习率
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # AdamW的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size when training.')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
