# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    # 已看過
    # 從dataset中拔出coco的api，目前沒有很清楚在幹啥
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        # CocoDetection裡面有這麼一段實例化Coco的Api
        # self.coco = COCO(annFile)
        # 下面是來自官方的解釋
        # COCO(annFile) = COCO apis classes that loads COCO annotation file and prepare data structures.
        return dataset.coco


def build_dataset(image_set, args):
    # 已看過
    # 這裡沒辦法用自訂義訓練集，所以之後要自定義資料集的之後再自己些吧
    # image_set = train or val
    if args.dataset_file == 'coco':
        # 預設是coco，原則上我們會用這個
        return build_coco(image_set, args)
    # coco_panoptic是做segmentation的
    # 這裡我們先不去處理全景分割，所以還是用上面的
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
