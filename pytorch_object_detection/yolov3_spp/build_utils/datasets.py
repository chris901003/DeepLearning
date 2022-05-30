import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from build_utils.utils import xyxy2xywh, xywh2xyxy

help_url = 'https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']


# get orientation in exif tag
# 找到图像exif信息中对应旋转信息的key值
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def exif_size(img):
    """
    获取图像的原始img size
    通过exif的orientation信息判断图像是否有旋转，如果有旋转则返回旋转前的size
    :param img: PIL图片
    :return: 原始图像的size
    """
    # 已看過
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270  顺时针翻转90度
            s = (s[1], s[0])
        elif rotation == 8:  # ratation 90  逆时针翻转90度
            s = (s[1], s[0])
    except:
        # 如果图像的exif信息中没有旋转信息，则跳过
        pass

    return s


class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self,
                 path,   # 指向data/my_train_data.txt路径或data/my_val_data.txt路径
                 # 这里设置的是预处理后输出的图片尺寸
                 # 当为训练集时，设置的是训练过程中(开启多尺度)的最大尺寸
                 # 当为验证集时，设置的是最终使用的网络大小
                 # 這裏設定的大小就是傳入網路的大小
                 img_size=416,
                 batch_size=16,
                 augment=False,  # 训练集设置为True(augment_hsv)，验证集设置为False，對比度亮度的增強
                 hyp=None,  # 超参数字典，其中包含图像增强会使用到的超参数，hyp.yaml
                 rect=False,  # 是否使用rectangular training，訓練的時候是False，驗證的時候是True
                 cache_images=False,  # 是否缓存图片到内存中
                 # rank在單gpu的時候是-1，多gpu實會是gpu的數量
                 single_cls=False, pad=0.0, rank=-1):
        # 已看過

        try:
            # 對path標準化
            path = str(Path(path))
            # parent = str(Path(path).parent) + os.sep
            if os.path.isfile(path):  # file
                # 读取对应my_train/val_data.txt文件，读取每一行的图片路劲信息
                with open(path, "r") as f:
                    f = f.read().splitlines()
            else:
                raise Exception("%s does not exist" % path)

            # 检查每张图片后缀格式是否在支持的列表中，保存支持的图像路径
            # img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
            self.img_files = [x for x in f if os.path.splitext(x)[-1].lower() in img_formats]
            self.img_files.sort()  # 防止不同系统排序不同，导致shape文件出现差异
        except Exception as e:
            raise FileNotFoundError("Error loading data from {}. {}".format(path, e))

        # 如果图片列表中没有图片，则报错
        n = len(self.img_files)
        assert n > 0, "No images found in %s. See %s" % (path, help_url)

        # batch index
        # 将数据划分到一个个batch中
        # 因為我們會把四張照片拼接成一張，所以這裡會有分組，但是這裡只是一個mask不是指定當前哪個照片要在哪組
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)
        # 记录数据集划分后的总batch数
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images 图像总数目
        self.batch = bi  # batch index of image 记录哪些图片属于哪个batch
        self.img_size = img_size  # 这里设置的是预处理后输出的图片尺寸
        self.augment = augment  # 是否启用augment_hsv
        self.hyp = hyp  # 超参数字典，其中包含图像增强会使用到的超参数
        self.rect = rect  # 是否使用rectangular training
        # 注意: 开启rect后，mosaic就默认关闭
        # 圖像擴充，就是把4張照片合成一張
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)

        # Define labels
        # 遍历设置图像对应的label路径
        # (./my_yolo_dataset/train/images/2009_004012.jpg) -> (./my_yolo_dataset/train/labels/2009_004012.txt)
        # 找到每張照片的標籤信息文件找到
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
                            for x in self.img_files]

        # Read image shapes (wh)
        # 查看data文件下是否缓存有对应数据集的.shapes文件，里面存储了每张图像的width, height
        # 到這裡才會形成shape文件
        sp = path.replace(".txt", ".shapes")  # shapefile path
        try:
            # 第一次進來會跳到下面，因為第一次不會有這個文件
            with open(sp, "r") as f:  # read existing shapefile
                # 把寬高拿出來 [[h, w],[h ,w], ..., [h, w]]
                s = [x.split() for x in f.read().splitlines()]
                # 判断现有的shape文件中的行数(图像个数)是否与当前数据集中图像个数相等
                # 如果不相等则认为是不同的数据集，故重新生成shape文件
                assert len(s) == n, "shapefile out of aync"
        except Exception as e:
            # print("read {} failed [{}], rebuild {}.".format(sp, e, sp))
            # tqdm库会显示处理的进度
            # 读取每张图片的size信息
            if rank in [-1, 0]:
                # 如果是在主進程的話會用tqdm將img_files迭代器給image_files
                image_files = tqdm(self.img_files, desc="Reading image shapes")
            else:
                image_files = self.img_files
            s = [exif_size(Image.open(f)) for f in image_files]
            # 将所有图片的shape信息保存在.shape文件中
            np.savetxt(sp, s, fmt="%g")  # overwrite existing (if any)

        # 记录每张图像的原始尺寸
        self.shapes = np.array(s, dtype=np.float64)

        # Rectangular Training https://github.com/ultralytics/yolov3/issues/232
        # 如果为ture，训练网络时，会使用类似原图像比例的矩形(让最长边为img_size)，而不是img_size x img_size
        # 注意: 开启rect后，mosaic就默认关闭
        # rect開啟後也會有圖片合併的效果，所以會把mosaic關閉
        # 訓練時候不會開啟，只有驗證的時候才會開啟
        # 驗證時開啟可以提高速度
        # 開啟的時候圖像大小不會變成img_size x img_size大小，而是等比例縮放到最大邊長等於img_size
        # 這樣不會對圖片進行裁切，可以不丟失照片信息但又可以減少計算量
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            # 计算每个图片的高/宽比
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            # argsort函数返回的是数组值从小到大的索引值
            # 按照高宽比例进行排序，这样后面划分的每个batch中的图像就拥有类似的高宽比
            # 這裏的irect存的是經過排序後哪個原始index應該在結果的哪個index
            # [1, 6, 2, 4, 3] => [0, 2, 4, 3, 1]
            irect = ar.argsort()

            # 根据排序后的顺序重新设置图像顺序、标签顺序以及shape顺序
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]
            # 全部相關東西都重新排序，這裏注意一下不要被之前的bi影響，那裡只是mask所以不需要跟著排

            # set training image shapes
            # 计算每个batch采用的统一尺度
            # 前面對高寬比進行拍序後可以盡可能的把長大於寬或寬大於長的放一起
            shapes = [[1, 1]] * nb  # nb: number of batches
            for i in range(nb):
                # 透過bi蒙版可以把一次取出想要的照片數量，舉個例子，這裡我們把四張圖片拼成一張
                # bi = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                # 那麼當i是0的時候我們就會拿出前四張圖
                ari = ar[bi == i]  # bi: batch index
                # 获取第i个batch中，最小和最大高宽比
                mini, maxi = ari.min(), ari.max()

                # 如果高/宽小于1(w > h)，将w设为img_size
                # 透過這樣可以維持原先比例，這裏是記錄下來等等要怎麼縮放
                # 這裏如果maxi越小表示長寬差越多，所以我們選maxi是為了在寬會放到img_size時讓長也可以放進去
                # 最後這個shapes會乘上img_size，所以1會等於img_size，非1的地方會先乘上img_size再向上取整到32的整數倍
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                # 如果高/宽大于1(w < h)，将h设置为img_size
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            # 计算每个batch输入网络的shape值(向上设置为32的整数倍)
            # 多出來的部分會用其他顏色進行填充
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int) * 32

        # cache labels
        self.imgs = [None] * n  # n为图像总数
        # label: [class, x, y, w, h] 其中的xywh都为相对值
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n
        extract_bounding_boxes, labels_loaded = False, False
        nm, nf, ne, nd = 0, 0, 0, 0  # number mission, found, empty, duplicate
        # 这里分别命名是为了防止出现rect为False/True时混用导致计算的mAP错误
        # 当rect为True时会对self.images和self.labels进行从新排序
        # 在開啟rect與不開啟rect的label是不一樣的，所以分成兩個不同檔案存檔
        if rect is True:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".rect.npy"  # saved labels in *.npy file
        else:
            np_labels_path = str(Path(self.label_files[0]).parent) + ".norect.npy"

        if os.path.isfile(np_labels_path):
            # 如果已經有生成好的話
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == n:
                # 如果载入的缓存标签个数与当前计算的图像数目相同则认为是同一数据集，直接读缓存
                self.labels = x
                # 並且將labels_loaded設置成True後面就不需再讀檔以及寫檔了
                labels_loaded = True

        # 处理进度条只在第一个进程中显示
        # tqdm好像就是會多一個進度條的樣子
        if rank in [-1, 0]:
            pbar = tqdm(self.label_files)
        else:
            pbar = self.label_files

        # 遍历载入标签文件
        for i, file in enumerate(pbar):
            if labels_loaded is True:
                # 如果存在缓存直接从缓存读取
                l = self.labels[i]
            else:
                # 从文件读取标签信息
                try:
                    with open(file, "r") as f:
                        # 读取每一行label，并按空格划分数据
                        # 這裏就是讀取label文件(my_yolo_dataset/train/labels/...txt)
                        # 裡面會有n行每行會有5個值(分類類別, x中心位置, y中心位置, w, h)，這裏都是相對位置
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except Exception as e:
                    print("An error occurred while loading the file {}: {}".format(file, e))
                    nm += 1  # file missing
                    continue

            # 如果标注信息不为空的话
            # l.shape[0]可以看到有多少個標籤，也就是有多少個目標匡
            if l.shape[0]:
                # 标签信息每行必须是五个值[class, x, y, w, h]
                assert l.shape[1] == 5, "> 5 label columns: %s" % file
                assert (l >= 0).all(), "negative labels: %s" % file
                assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: %s" % file

                # 检查每一行，看是否有重复信息
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows
                    nd += 1
                if single_cls:
                    # 這個這裏沒有用到
                    l[:, 0] = 0  # force dataset into single-class mode

                # 把剛剛從檔案讀取的label存進去
                self.labels[i] = l
                nf += 1  # file found

                # Extract object detection boxes for a second stage classifier
                if extract_bounding_boxes:
                    # 這裏默認是False，不會進來
                    p = Path(self.img_files[i])
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):
                        f = "%s%sclassifier%s%g_%g_%s" % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        # 将相对坐标转为绝对坐标
                        # b: x, y, w, h
                        b = x[1:] * [w, h, w, h]  # box
                        # 将宽和高设置为宽和高中的最大值
                        b[2:] = b[2:].max()  # rectangle to square
                        # 放大裁剪目标的宽高
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        # 将坐标格式从 x,y,w,h -> xmin,ymin,xmax,ymax
                        b = xywh2xyxy(b.reshape(-1, 4)).revel().astype(np.int)

                        # 裁剪bbox坐标到图片内
                        b[[0, 2]] = np.clip[b[[0, 2]], 0, w]
                        b[[1, 3]] = np.clip[b[[1, 3]], 0, h]
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), "Failure extracting classifier boxes"
            else:
                # 這張照片裡面沒有標籤
                ne += 1  # file empty

            # 处理进度条只在第一个进程中显示
            if rank in [-1, 0]:
                # 更新进度条描述信息
                pbar.desc = "Caching labels (%g found, %g missing, %g empty, %g duplicate, for %g images)" % (
                    nf, nm, ne, nd, n)
        assert nf > 0, "No labels found in %s." % os.path.dirname(self.label_files[0]) + os.sep

        # 如果标签信息没有被保存成numpy的格式，且训练样本数大于1000则将标签信息保存成numpy的格式
        if not labels_loaded and n > 1000:
            # 寫入檔案
            print("Saving labels to %s for faster future loading" % np_labels_path)
            np.save(np_labels_path, self.labels)  # save for next time

        # Cache images into memory for faster training (Warning: large datasets may exceed system RAM)
        # 緩存圖像
        if cache_images:  # if training
            # 這裏預設是False
            gb = 0  # Gigabytes of cached images 用于记录缓存图像占用RAM大小
            if rank in [-1, 0]:
                pbar = tqdm(range(len(self.img_files)), desc="Caching images")
            else:
                pbar = range(len(self.img_files))

            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                # imgs圖片，img_hw0原始圖片大小，img_hw經過等比例縮放後的大小
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes  # 用于记录缓存图像占用RAM大小
                if rank in [-1, 0]:
                    pbar.desc = "Caching images (%.1fGB)" % (gb / 1E9)

        # Detect corrupted images https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
        detect_corrupted_images = False
        # 檢測圖片有沒有問題，這裏預設都是False
        if detect_corrupted_images:
            from skimage import io  # conda install -c conda-forge scikit-image
            for file in tqdm(self.img_files, desc="Detecting corrupted images"):
                try:
                    _ = io.imread(file)
                except Exception as e:
                    print("Corrupted image detected: {}, {}".format(file, e))

    def __len__(self):
        # 已看過
        return len(self.img_files)

    def __getitem__(self, index):
        # 已看過
        # 拿到超參數
        hyp = self.hyp
        if self.mosaic:
            # load mosaic
            # 訓練的時候會使用，這裏可以將四張圖片進行拼接，這裏不是說一次可以處理四張照片，而是拿現在的指定照片與隨機三張照片
            # 做合併，所以總共訓練張數不會除以四，這點要特別注意
            img, labels = load_mosaic(self, index)
            shapes = None
        else:
            # 如果沒有用mosaic理論上就會有rect，所以就可以得到batch_shapes資訊
            # load image
            img, (h0, w0), (h, w) = load_image(self, index)

            # letterbox
            # 先找到這個index屬於哪個batch，再去拿這個batch的shape
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            # 使用letterbox進行裁剪使得對應上shape
            # img結果圖像，ratio長寬的變化比例，pad加上的灰色區塊大小這裏是左右或上下加再一起
            # 所以h/2才可以知道上面加了多少灰色填充
            img, ratio, pad = letterbox(img, shape, auto=False, scale_up=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # load labels
            labels = []
            x = self.labels[index]
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                labels = x.copy()  # label: class, x, y, w, h
                # 注意一下，這裏有乘上ratio是說在縮放回原始圖片的絕對位置後又在經過一次縮放
                # (w, h)是原始經過load_image後的圖片大小
                # 應為我們先透過指定大小把圖片load近來又為了拼接四張圖片所以圖有縮放了一次
                # 所以才會有ratio又有(w, h)
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            # 如果是用mosaic則裡面已經有訪攝變換過了，所以這邊就會跳過
            if not self.mosaic:
                img, labels = random_affine(img, labels,
                                            degrees=hyp["degrees"],
                                            translate=hyp["translate"],
                                            scale=hyp["scale"],
                                            shear=hyp["shear"])

            # Augment colorspace
            # 也是對圖像做增強
            augment_hsv(img, h_gain=hyp["hsv_h"], s_gain=hyp["hsv_s"], v_gain=hyp["hsv_v"])

        # 看有多少目標邊界匡
        nL = len(labels)  # number of labels
        if nL:
            # convert xyxy to xywh
            # 再把labels裡存的坐上角以及右下角座標變成中心座標以及寬高
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0-1
            # 轉換回相對座標
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.augment:
            # random left-right flip
            lr_flip = True  # 随机水平翻转
            if lr_flip and random.random() < 0.5:
                # 圖片反轉
                img = np.fliplr(img)
                if nL:
                    # 目標匡的x中心也要左右反轉
                    labels[:, 1] = 1 - labels[:, 1]  # 1 - x_center

            # random up-down flip
            # 這裏設定是不會進行上下翻轉
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]  # 1 - y_center

        labels_out = torch.zeros((nL, 6))  # nL: number of labels
        if nL:
            # 這裏把label的5個值給labels_out而index0後面在說要放什麼
            # label存的是 [目標標籤, 中心x, 中心y, w, h]
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert BGR to RGB, and HWC to CHW(3x512x512)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # img_files是圖片路徑
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes, index

    def coco_index(self, index):
        """该方法是专门为cocotools统计标签信息准备，不对图像和标签作任何处理"""
        o_shapes = self.shapes[index][::-1]  # wh to hw

        # load labels
        x = self.labels[index]
        labels = x.copy()  # label: class, x, y, w, h
        return torch.from_numpy(labels), o_shapes

    @staticmethod
    def collate_fn(batch):
        # 已看過
        img, label, path, shapes, index = zip(*batch)  # transposed
        for i, l in enumerate(label):
            # 會對label的[0]index給值，給的就是在這個batch中是第幾張圖片
            # 原因是下面會做cat拼接
            l[:, 0] = i  # add target image index for build_targets()
        t = torch.stack(img, 0)
        t2 = torch.cat(label, 0)
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes, index


def load_image(self, index):
    # 已看過
    # 用來把圖片載入到內存中
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    # 現看看對應位置有沒有放入緩存，如果沒有就會讀取圖片
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, "Image Not Found " + path
        h0, w0 = img.shape[:2]  # orig hw
        # img_size 设置的是预处理后输出的图片尺寸
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def load_mosaic(self, index):
    """
    将四张图片拼接在一张马赛克图像中
    :param self:
    :param index: 需要获取的图像索引
    :return:
    """
    # loads images in a mosaic
    # 已看過

    labels4 = []  # 拼接图像的label信息
    # img_size最終圖片輸出大小
    s = self.img_size
    # 随机初始化拼接图像的中心点坐标
    # 會有四張照片要拼接，我們會給一個中心點，因為有四張照片要拼接所以一開始圖的大小會是2xS，然後我們隨機找一個中心點
    # 之後第一張圖的右下角會對上中心點，第二張圖的左下角會對上中心點，第三張圖的右上角會對上中心點，第四張圖的左上角會對上中心點
    xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
    # 从dataset中随机寻找三张图像进行拼接
    # 真的是隨機採取喔，這裏list加法就是extend => [10] + [1, 3, 7] = [10, 1, 3, 7]
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    # 遍历四张图像进行拼接
    for i, index in enumerate(indices):
        # load image
        # 使用load_image方法載入圖片，其實這裏跟使用create_cache裡的是一樣的函數
        # 但是這裏載入後不會放進緩存裡，因為load_image只負責載入圖片
        # img就是圖片，(h, w)已經經過等比例縮放後的高和寬
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        # 以a做結尾的都是在指最後圖片的座標位置
        # 以b做結尾的都使在指以當前讀取圖片的座標位置
        if i == 0:  # top left
            # 创建马赛克图像，先用114也就是灰色來全部填充
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            # x1a(將第一張圖片放入時的左上角x座標)後面都是這樣推理
            # x2a(對於第一張圖就是原始圖片的右下角)
            # 這些東西都可以用上面說的圖片擺放方式推出來，耐心地看一下就會懂了
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            # 计算截取的图像区域信息(以xc,yc为第一张图像的右下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            # 计算截取的图像区域信息(以xc,yc为第二张图像的左下角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第三张图像的右上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
        elif i == 3:  # bottom right
            # 计算马赛克图像中的坐标信息(将图像填充到马赛克图像中)
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            # 计算截取的图像区域信息(以xc,yc为第四张图像的左上角坐标填充到马赛克图像中，丢弃越界的区域)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # 将截取的图像区域填充到马赛克图像的相应位置
        # 這裏再多說一下，y2a - y1a = y2b - y2a，x2a - x1a = x2b- x1b
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        # 计算pad(图像边界与马赛克边界的距离，越界的情况为负值)
        # 可以看一下對於框框會偏移多少，這裏也是絕對座標
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels 获取对应拼接图像的labels信息
        # [class_index, x_center, y_center, w, h]
        # 這裏的labels就是從檔案讀取出來的，因為是相對位置，所以可以直接乘上圖篇後來的高和寬還是可以匡著原本的物體
        x = self.labels[index]
        labels = x.copy()  # 深拷贝，防止修改原数据
        if x.size > 0:  # Normalized xywh to pixel xyxy format
            # 计算标注数据在马赛克图像中的坐标(绝对坐标)
            # 這裏因為成上了w或h所以才變成絕對座標
            # 由於padw和padh都是絕對座標所以可以加
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw   # xmin
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh   # ymin
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw   # xmax
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh   # ymax
        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        # 確保有檢測目標
        labels4 = np.concatenate(labels4, 0)
        # 设置上下限防止越界
        # np.clip(輸入的數組, min, max, 結果要輸出到哪個數組)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

    # Augment
    # 随机旋转，缩放，平移以及错切
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=self.hyp['degrees'],
                                  translate=self.hyp['translate'],
                                  scale=self.hyp['scale'],
                                  shear=self.hyp['shear'],
                                  border=-s // 2)  # border to remove
    # 通過random_affine的img會從原來的2s * 2s變成s * s，當然labels4裡面的匡也會有變動

    return img4, labels4


def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
    """随机旋转，缩放，平移以及错切"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    # 这里可以参考我写的博文: https://blog.csdn.net/qq_37541097/article/details/119420860
    # targets = [cls, xyxy]
    # 已看過
    # 都是影像處理的東西

    # 最终输出的图像尺寸，等于img4.shape / 2
    # border = -s // 2
    # 最後的圖片大小會是s，也就是說在這裡會把圖片從2s變s
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    # 生成旋转以及缩放矩阵
    R = np.eye(3)  # 生成对角阵
    a = random.uniform(-degrees, degrees)  # 随机旋转角度
    s = random.uniform(1 - scale, 1 + scale)  # 随机缩放因子
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    # 生成平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    # 生成错切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        # 进行仿射变化
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        # [4*n, 3] -> [n, 8]
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        # 对transform后的bbox进行修正(假设变换后的bbox变成了菱形，此时要修正成矩形)
        x = xy[:, [0, 2, 4, 6]]  # [n, 4]
        y = xy[:, [1, 3, 5, 7]]  # [n, 4]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T  # [n, 4]

        # reject warped points outside of image
        # 对坐标进行裁剪，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]

        # 计算调整后的每个box的面积
        area = w * h
        # 计算调整前的每个box的面积
        area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
        # 计算每个box的比例
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
        # 选取长宽大于4个像素，且调整前后面积比例大于0.2，且比例小于10的box
        i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def augment_hsv(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    # 这里可以参考我写的博文:https://blog.csdn.net/qq_37541097/article/details/119478023
    # 已看過
    # 這裏也是對圖像進行增強
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img: 原始圖片
    :param new_shape: 期望調整的大小
    :param color: 空白處填充顏色
    :param auto:
    :param scale_fill:
    :param scale_up:
    :return:
    """
    # 已看過

    # shape原始圖片大小
    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        # 將最終調整大小從int轉成tuple
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    # 最終調整大小與原始圖片的比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 使得最長邊會等於最終輸出大小，較小的就用灰色填充
    if not scale_up:  # only scale down, do not scale up (for better test mAP)
        # 对于大于指定输入大小的图片进行缩放,小于的不变
        # 訓練時為True，驗證時為False
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    # 這裏的new_unpad是先[1]在[0]，注意一下雖然不知道為什麼
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍
        # 不會進來
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        # 不會進來
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    # 因為前面構建new_unpad是反的
    # 這裏的cv2.resize()，吃的new_unpad格式是(w, h)所以我們才在一開始時是反的，這裏不會有錯，哪裡該邊寬或變窄都是正常的
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def create_folder(path="./new_folder"):
    # Create floder
    if os.path.exists(path):
        shutil.rmtree(path)  # dalete output folder
    os.makedirs(path)  # make new output folder




