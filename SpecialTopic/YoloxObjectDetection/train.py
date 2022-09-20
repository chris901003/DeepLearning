import argparse
import os
import torch
from torch.utils.data import DataLoader
from utils import get_specified_option
from detector import build_detector
from dataset import build_dataset, custom_collate_fn, custom_collate_fn_val
from net.run import run
from net.yolo_training import weights_init, YOLOLoss
import numpy as np
from torch import nn


def parse_args():
    parser = argparse.ArgumentParser(description='Train a yolox objection detector')
    parser.add_argument('--work-dir')
    parser.add_argument('--model-size', default='l', type=str)
    parser.add_argument('--num-classes', default=9, type=int)
    parser.add_argument('--use-depthwise', default=False, type=bool)
    parser.add_argument('--train-epoch', default=300, type=int)
    parser.add_argument('--val-epoch', default=3, type=int)
    parser.add_argument('--val-coco-json', default='/Users/huanghongyan/Downloads/data_annotation/val2017.json',
                        type=str)
    parser.add_argument('--pretrained', default='/Users/huanghongyan/Documents/DeepLearning/'
                                                'SpecialTopic/YoloxObjectDetection/work_dir/best_model.pkt')
    parser.add_argument('--fp16', default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.work_dir is None:
        args.work_dir = os.path.join(os.getcwd(), 'work_dir/')
    if not os.path.exists(args.work_dir):
        os.mkdir(args.work_dir)
    assert os.path.isdir(args.work_dir), f'指定的保存路徑 {args.work_dir} 需要是資料夾路徑'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # support_model_size = {
    #     'l': dict(deepen_factor=1.0, widen_factor=1.0, neck_in_channels=[256, 512, 1024]),
    #     's': dict(deepen_factor=0.33, widen_factor=0.5, neck_in_channels=[128, 256, 512]),
    # }
    # model_args = get_specified_option(support_model_size, args.model_size)
    # model_cfg = {
    #     'type': 'YOLOX',
    #     'input_size': (640, 640),
    #     'backbone': {
    #         'type': 'CSPDarknet',
    #         'deepen_factor': model_args['deepen_factor'],
    #         'widen_factor': model_args['widen_factor'],
    #         'use_depthwise': args.use_depthwise
    #     },
    #     'neck': {
    #         'type': 'YOLOXPAFPN',
    #         'in_channels': model_args['neck_in_channels'],
    #         'out_channels': model_args['neck_in_channels'][0],
    #         'num_csp_blocks': 1,
    #         'use_depthwise': args.use_depthwise
    #     },
    #     'bbox_head': {
    #         'type': 'YOLOXHead',
    #         'num_classes': args.num_classes,
    #         'in_channels': model_args['neck_in_channels'][0],
    #         'feat_channels': model_args['neck_in_channels'][0],
    #         'use_depthwise': args.use_depthwise
    #     },
    #     'train_cfg': {
    #         'assigner': {
    #             'type': 'SimOTAAssigner',
    #             'center_radius': 2.5
    #         }
    #     },
    #     'test_cfg': {
    #         'score_thr': 0.01,
    #         'nms': {
    #             'type': 'nms',
    #             'iou_threshold': 0.65
    #         }
    #     },
    #     'optimizer_cfg': {
    #         'type': 'SGD',
    #         'momentum': 0.9,
    #         'lr': 0.001,
    #         'weight_decay': 5e-4,
    #         'nesterov': True
    #     }
    # }
    model_cfg = {
        'type': 'YoloBody',
        'phi': args.model_size,
        'num_classes': args.num_classes
    }
    model = build_detector(model_cfg)
    weights_init(model)
    model = model.to(device)
    if args.pretrained is not None:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.pretrained, map_location='cpu')
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    yolo_loss = YOLOLoss(args.num_classes, args.fp16)
    # 需額外調整到[center_x, center_y, w, h]
    dataset_cfg = {
        'type': 'MultiImageMixDataset',
        'dataset': {
            'type': 'LabelImgYoloFormat',
            'annotations': '/Users/huanghongyan/Downloads/data_annotation/annotations',
            'images': '/Users/huanghongyan/Downloads/data_annotation/imgs',
            'pipelines': [
                {'type': 'LoadImageFromFile', 'save_key': 'img'},
                {'type': 'LoadAnnotations', 'img_key': 'img', 'save_key': ['gt_labels', 'gt_bboxes'], 'with_bbox': True}
            ]
        },
        'pipelines': [
            {'type': 'Mosaic', 'img_scale': (640, 640), 'pad_val': 114.0},
            # {'type': 'RandomAffine', 'scaling_ratio_range': (0.1, 2), 'border': (-320, -320)},
            # {'type': 'MixUp', 'img_scale': (640, 640), 'ratio_range': (0.8, 1.6), 'pad_val': 114.0},
            {'type': 'YOLOXHSVRandomAug'},
            {'type': 'RandomFlip', 'flip_ratio': 0.5},
            {'type': 'Resize', 'img_scale': (640, 640), 'keep_ratio': True},
            {'type': 'Pad', 'pad_to_square': True, 'pad_val': {'img': (114.0, 114.0, 114.0)}},
            {'type': 'FilterAnnotations', 'min_gt_bbox_wh': (1, 1), 'keep_empty': False},
            {'type': 'GTBBoxFormat', 'ori': 'xyxy', 'after': 'xywh'},
            {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels']}
        ]
    }
    dataset = build_dataset(dataset_cfg)

    if args.val_epoch is not None:
        # 構建驗證圖像的dataset
        val_dataset_cfg = {
            'type': 'LabelImgYoloFormat',
            'annotations': '/Users/huanghongyan/Downloads/data_annotation/annotations',
            'images': '/Users/huanghongyan/Downloads/data_annotation/imgs',
            'pipelines': [
                {'type': 'LoadImageFromFile', 'save_key': 'img'},
                {'type': 'LoadAnnotations', 'img_key': 'img', 'save_key': ['gt_labels', 'gt_bboxes'], 'with_bbox': True},
                {'type': 'Resize', 'img_scale': (640, 640), 'keep_ratio': False},
                {'type': 'GTBBoxFormat', 'ori': 'xyxy', 'after': 'xywh'},
                {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels', 'scale_factor', 'image_path']}
            ]
        }
        val_dataset = build_dataset(val_dataset_cfg)
    else:
        val_dataset = None

    # 可視化
    # results = dataset[0]
    # from PIL import Image
    # import cv2
    # bboxes = results['gt_bboxes']
    # labels = results['gt_labels']
    # image = cv2.cvtColor(results['img'], cv2.COLOR_BGR2RGB)
    # for bbox, label in zip(bboxes, labels):
    #     label = int(label)
    #     xmin = int(bbox[0] - bbox[2] / 2)
    #     ymin = int(bbox[1] - bbox[3] / 2)
    #     xmax = int(bbox[0] + bbox[2] / 2)
    #     ymax = int(bbox[1] + bbox[3] / 2)
    #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    #     cv2.putText(image, str(label), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 90, 44), 5, cv2.LINE_AA)
    # image = Image.fromarray(image)
    # image.show()

    # 測試模型正向推理
    # img = torch.randn((2, 3, 640, 640))
    # bbox = torch.Tensor([[0, 0, 50, 32], [10, 30, 74, 90], [210, 150, 367, 218], [382, 540, 439, 582]])
    # gt_bbox = [bbox, bbox]
    # gt_label = [torch.randint(0, 4, size=(4,)) for _ in range(2)]
    # out = model(img, gt_bbox, gt_label)

    dataloader_cfg = {
        'dataset': dataset,
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 1,
        'pin_memory': True,
        'drop_last': False,
        'collate_fn': custom_collate_fn
    }
    train_dataloader = DataLoader(**dataloader_cfg)

    if args.val_epoch is not None:
        val_dataloader_cfg = {
            'dataset': val_dataset,
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 1,
            'pin_memory': True,
            'drop_last': False,
            'collate_fn': custom_collate_fn_val
        }
        val_dataloader = DataLoader(**val_dataloader_cfg)
    else:
        val_dataloader = None

    if args.fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    train_epoch = args.train_epoch
    val_epoch = args.val_epoch
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {
        'adam': torch.optim.Adam(pg0, 0.001, betas=(0.937, 0.999)),
        'sgd': torch.optim.SGD(pg0, 0.001, momentum=0.937, nesterov=True)
    }['sgd']
    optimizer.add_param_group({"params": pg1, "weight_decay": 5e-4})
    optimizer.add_param_group({"params": pg2})
    run(model, yolo_loss, device, args.work_dir, train_epoch, train_dataloader, optimizer, val_epoch=val_epoch,
        val_dataloader=val_dataloader, val_coco_json=args.val_coco_json, scaler=scaler)
    print('Finish training')


if __name__ == '__main__':
    main()
    print('Finish')
