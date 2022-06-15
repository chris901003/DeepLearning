import os

import torch.utils.data as data
from PIL import Image


class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        """
        :param voc_root: 數據集路徑
        :param year: 年份
        :param transforms: 轉換方式
        :param txt_name: 檔案內容是哪些圖片是要用到的
        """
        # 已看過
        super(VOCSegmentation, self).__init__()
        # 只支援兩個年份
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        # 圖片資料夾的路徑
        image_dir = os.path.join(root, 'JPEGImages')
        # mask圖片資料夾的路徑
        mask_dir = os.path.join(root, 'SegmentationClass')

        # 需用到哪些照片的檔案位置
        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        # 讀取照片名稱
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        # 照片檔案位置
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        # 分割圖片檔案位置
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        # 轉換方式
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        # 已看過
        # 讀取一張照片
        # images存放圖片的路徑，將圖片讀出後轉成RGB的Image格式
        img = Image.open(self.images[index]).convert('RGB')
        # masks存放分割圖片的路徑，將圖片讀出並且格式為Image，本身就是RGB
        target = Image.open(self.masks[index])

        # 如果有需要進行transform就會進行，可以到transforms中的__call__看有進行什麼操作
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        # 回傳的圖像大小都已經調整好了
        return img, target

    def __len__(self):
        # 已看過
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        # 已看過
        # list型態的images以及targets後面需要轉換成一個batch的tensor
        images, targets = list(zip(*batch))
        # 建立一個batch的tensor
        # batched_imgs, batched_targets shape [batch_size, channel, w, h]
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 已看過
    # 计算该batch数据中，channel, h, w的最大值
    # 正常來說因為通過前面的transform一個batch中的照片channel,h,w都會是一樣大
    # max_size = tuple(max_channel, max_h, max_w)
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # batch_shape = 一個batch的tensor的shape [batch_size, channel, w, h]
    batch_shape = (len(images),) + max_size
    # batched_imgs = 構建一個batch的tensor的底版，先把所有值設定成fill_value
    # batched_imgs shape [batch_size, channel, w, h]
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    # 將圖片貼上底版，如果真的有不夠的地方就會是預設，以左上角對齊的方式貼
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    # 回傳[batch_size, channel, w, h]
    return batched_imgs


# dataset = VOCSegmentation(voc_root="/data/", transforms=get_transform(train=True))
# d1 = dataset[0]
# print(d1)
