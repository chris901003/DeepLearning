# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        # 已看過
        # ---------------------------------------------------------
        # 這裡繼承了CocoDetection，這是一個由官方製作的api用來處理coco數據集
        # 簡單說明一下CocoDetection用法，以下來自torchvision官網介紹
        # root : coco圖片路徑
        # annFile : 標註文件路徑
        # transform : 圖像轉換(用於PIL) [可選擇不填，默認None]
        # target_transform : 標註轉換 [可選擇不填，默認None]
        # transforms : 圖像以及標注的轉換 [可選擇不填，默認None]
        # CocoDetection就是一種dataset得樣子了，也就是可以放入Dataloader裡，只不過要自己定義collect_function
        # CocoDetection在遍歷時給的輸出會是[Image, Label]，圖像(型態就要看有沒有做transform)
        # 標註(Coco裡面的各種標註，字典型態)，如果要gt_box就會是labels['bbox']就可以取得
        # ---------------------------------------------------------
        # 這裡我們只有給文件的位置而已，後面的transform自己等等做
        super(CocoDetection, self).__init__(img_folder, ann_file)
        # 轉換方式，包含圖像以及gt_box
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        # 已看過
        # 記得這個getitem一次只是處理一張照片
        # 開始讀取照片
        # 從父類CocoDetection的__getitem拿到index對應的圖片以及標籤
        # img shape = [PIL]
        # target shape = [every_photo_annotation(List[Dict])]
        img, target = super(CocoDetection, self).__getitem__(idx)
        # 由idx找到對應圖片的id
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        # 調用ConvertCocoPolysToMask的__call__方法
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            # 圖像轉換，同時gt_box也變成相對座標了[center_x, center_y, w, h]
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    # 這個是做segmentation才會用到，目前暫時不會用到
    # 未讀
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        # 已看過
        # return_maks預設為False
        self.return_masks = return_masks

    def __call__(self, image, target):
        """
        :param image: shape [PIL]
        :param target: Dict { 'image_id':圖像檔案名稱也是id, 'annotation':[evey_photo_annotation(List[Dict])]}
        """
        # 已看過
        # 原始圖像寬和高
        w, h = image.size

        # 取出圖片的id，並轉成tensor格式
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        # 拿出annotation(List[Dict])
        anno = target["annotations"]

        # 過濾出被標記為iscrowd的gt_box，通常有被標記的都是有遮擋的
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # 取出annotation裡面的bbox，這才是我們要的gt_box
        # 這裡拿出來的格式是[xmin, ymin, w, h]且為絕對座標
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        # 轉成tensor類別
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # 將boxes從[xmin, ymin, w, h] -> [xmin, ymin, xmax, ymax]格式
        boxes[:, 2:] += boxes[:, :2]
        # 限制最大最小值
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # 獲取每個gt_box對應的分類類別，同時轉成tensor格式
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # 如果要做segmentation的訓練的話才會用到，這裡我們不會用到
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # 如果需要訓練人物關節點的話就會用到，這裡我們不會用到
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # 過濾掉一些標註不合法的，在coco數據集中有不少標記有問題的圖片
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        # 記得classes也要同步過濾掉
        classes = classes[keep]
        # 這裡我們return_masks預設為False
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # 最後整理出我們真正的target
        # 這裡target的每個value都是List格式，除了image_id
        target = {"boxes": boxes, "labels": classes}
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        # 為了之後要做coco mAP計算要用到的
        # 每個gt_box的面積
        area = torch.tensor([obj["area"] for obj in anno])
        # iscrowd標籤，因為前面有過濾過了所以這裡基本上都是0
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        # 記得過濾掉剛剛不合法的gt_box
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # 給兩個跟大小有關的key，size之後應該會被改動
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # ---------------------------------------------------------
        # target中到底有什麼
        # boxes: 就是gt_box格式為 [xmin, ymin, xmax, ymax] (List(List))
        # labels: 每個gt_box的分類類別 (List(int))
        # image_id: 這張圖片的id (int)
        # area: 每個gt_box的大小 (List(int))
        # iscrowd: 每個gt_box的iscrowd標籤 (List(int))
        # orig_size: 圖片原始大小 ([int, int])
        # size: 目前是原始圖片大小，之後可能會做更動 ([int, int])
        # ---------------------------------------------------------
        return image, target


def make_coco_transforms(image_set):
    # 已看過
    # 轉成tensor後在做正則化
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    # 一系列操作
    # 這裡的T都是在datasets裡面的transforms實作的不是官方的
    # 經過T.Normalize過後gt_box會變成[center_x, center_y, w, h]且為相對座標
    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    # 已看過
    # 把路徑轉換成pathlib格式，並檢查檔案是否存在
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    # instances檔案裡面裝的就是目標檢測的gt_box標註
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }
    # 拿到圖片的檔案以及標註檔案
    img_folder, ann_file = PATHS[image_set]
    # image_set = 'train' or 'val'
    # args.masks預設為False
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
