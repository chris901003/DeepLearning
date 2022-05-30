import glob
import math
import os
import random
import time

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from build_utils import torch_utils  # , google_utils

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch_utils.init_seeds(seed=seed)


def check_file(file):
    # Searches for file if not found locally
    # 已看過
    if os.path.isfile(file):
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found


def xyxy2xywh(x):
    # 已看過
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # 已看過
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将预测的坐标信息转换回原图尺度
    :param img1_shape: 缩放后的图像尺度
    :param coords: 预测的box信息
    :param img0_shape: 缩放前的图像尺度
    :param ratio_pad: 缩放过程中的缩放比例以及pad
    :return:
    """
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = max(img1_shape) / max(img0_shape)  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    # 把範圍固定好
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # 已看過
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False):
    # 已看過
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.t()

    # Get the coordinates of bounding boxes
    # x1y1x2y2是用來看傳入的給好左上右下位置，還是只給中心點以及寬高
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    # 看是需要什麼類型的iou就算什麼給他
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # iou(3, n) = wh_iou(anchors(3, 2), gwh(n, 2))
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    # 已看過
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


class FocalLoss(nn.Module):
    # 這裡沒有啟用
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    # 已看過
    return 1.0 - 0.5 * eps, 0.5 * eps


def compute_loss(p, targets, model):  # predictions, targets, model
    # 已看過
    device = p[0].device
    # 分類損失
    lcls = torch.zeros(1, device=device)  # Tensor(0)
    # 定位損失
    lbox = torch.zeros(1, device=device)  # Tensor(0)
    # 判斷是否有匡到目標損失，這裏計算的是goiu值
    lobj = torch.zeros(1, device=device)  # Tensor(0)
    # tcls = 每個正樣本的正確class
    # tbox = 每個正樣本的正確中心寬高的回歸參數
    # indices = 每個正樣本在特徵圖上的pixel，此batch中的哪張照片
    # anch = 每個正樣本對應上的anchor長寬，這裡的長寬是相對大小
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)

    # Define criteria
    # pos_weight這裏透過超參數設定，reduction設定成將結果平均後輸出
    # 分類損失
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device), reduction=red)
    # obj損失
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device), reduction=red)

    # class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    # 這個我們沒有用到，因為我們把eps設成0，cp會是1，cn會是0
    cp, cn = smooth_BCE(eps=0.0)

    # focal loss
    # 默認不去使用
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # per output
    # 遍歷每個yolo_layer的輸出
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image_idx, anchor_idx, 中心點座標y, 中心點座標x
        # pi shape [batch_size, anchor_per_pixel, feature_width, feature_height, 5 + num_classes]
        # tobj的shape會跟pi一樣只是先拿0做填充
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        # b所存的是每個正樣本對應上這個batch中的哪張圖片，所以我們取長度也可以知道有多少正樣本
        nb = b.shape[0]  # number of positive samples
        if nb:
            # 对应匹配到正样本的预测信息
            # pi shape [batch_size, anchor_per_pixel, feature_width, feature_height, 5 + num_classes]
            # ps shape [正樣本數量, 5 + num_classes]
            # ps 就是把我們要的提取出來
            # b會選擇出這個batch中的哪張圖，a會選出哪個anchor，gj會找到寬度座標，gi會找到高度座標
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # GIoU
            # pxy = (x, y)中心回歸參數，也可以說是偏移量，這裏的都是預測的，就是機器給的
            pxy = ps[:, :2].sigmoid()
            # pwh透過公式後面需要再乘上anchors的高度以及寬度
            pwh = ps[:, 2:4].exp().clamp(max=1E3) * anchors[i]
            # 在維度1上作拼接
            pbox = torch.cat((pxy, pwh), 1)  # predicted box
            # 我們拿預測出來的回歸參數與真正的回歸參數去計算giou值
            # .t()是轉置的意思，不用問為什麼這裡pbox需要轉置而tbox不用，因為等等進去會先把tbox轉置
            # x1y1x2y2是用來看傳入的給好左上右下位置，還是只給中心點以及寬高
            # pbox與tbox都是[x中心偏移量, y中心偏移量, w回歸參數, h回歸參數]，所以進去算iou時我們是假裝(0, 0)是原先位置
            giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
            lbox += (1.0 - giou).mean()  # giou loss

            # Obj
            # 這裡如果我們定義只是判斷前景或背景那就只有0或1
            # 這裡用的是giou ratio所以會有點不同
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Class
            if model.nc > 1:  # cls loss (only if multiple classes)
                # t會是一個跟ps[:, 5:]一樣的shape並且一開始用cn填充，cn這裏是0
                # ps[:, 5]就是針對每個正樣本，他預測是哪個類別的機率 shape [正樣本個數, num_classes]
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                # 把正樣本應對應上正確的類別標籤的地方變成cp，這裡的cp是1
                # 通過這樣後t就是真實的目標標籤
                t[range(nb), tcls[i]] = cp
                # 計算類別分類損失
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        # pi[..., 4]是預測的，tobj是真實的
        # 透過這兩個可以計算出obj的損失
        lobj += BCEobj(pi[..., 4], tobj)  # obj loss

    # 乘上每种损失的对应权重
    # 這裏的都是超參數
    lbox *= h['giou']
    lobj *= h['obj']
    lcls *= h['cls']

    # loss = lbox + lobj + lcls
    # 回傳損失值
    return {"box_loss": lbox,
            "obj_loss": lobj,
            "class_loss": lcls}


def build_targets(p, targets, model):
    # 已看過
    """
    p : [yolo_layers, batch_size, anchors_per_pixel, feature_width, feature_height, 4 + 1 + num_classes]
    p : (3, 4, 3, 15, 15, 25)
    targets : [total_gt_box, 6] (整個batch的gt_box, (image_idx, class, x, y , w, h))
    # image_idx是這個batch的那個圖片，同時這裏的座標信息是相對座標信息
    """
    # Build targets for compute_loss(), input targets(image_idx,class,x,y,w,h)
    # nt = 當前這個batch有多少目標匡
    nt = targets.shape[0]
    # tcls = 每個正樣本的正確class
    # tbox = 每個正樣本的正確中心寬高的回歸參數
    # indices = 每個正樣本在特徵圖上的pixel，此batch中的哪張照片
    # anch = 每個正樣本對應上的anchor長寬，這裡的長寬是相對大小
    tcls, tbox, indices, anch = [], [], [], []
    # gain 對於每個target的6個值給增益，用來把anchor映射到feature_map上
    gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain

    multi_gpu = type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    # 一層一層yolo_layer遍歷
    for i, j in enumerate(model.yolo_layers):  # j: [89, 101, 113]
        # 获取该yolo predictor对应的anchors
        # 注意anchor_vec是anchors缩放到对应特征层上的尺度
        # anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90],
        # [156, 198], [474, 326]]
        # anchor_vec = anchor / stride
        # anchors是相對值
        anchors = model.module.module_list[j].anchor_vec if multi_gpu else model.module_list[j].anchor_vec
        # p[i].shape: [batch_size, 3, grid_h, grid_w, num_params]
        gain[2:] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain[2, 3, 4, 5]
        # gain會變成[1, 1, grid_w, grid_h, grid_w, grid_h] (1, 1, 15, 15, 15, 15) 15就是當前特徵圖的大小
        # 下面會看到gain * target這樣就可以把相對座標轉換成絕對座標
        # na = 這個特徵層上每個pixel有多少個anchors
        na = anchors.shape[0]  # number of anchors
        # [3] -> [3, 1] -> [3, nt]
        # 3種anchor與gt_box的表
        at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)

        # Match targets to anchors
        # a = 空列表, t = 把target中的相對座標變成絕對座標, offsets = 0 可以暫時不用管
        a, t, offsets = [], targets * gain, 0
        if nt:  # 如果存在target的话
            # 每層的anchor模板就有三個，可以看上面的anchors註解
            # 通过计算anchor模板与所有target的wh_iou来匹配正样本
            # j: [3, nt] , iou_t = 0.20
            # 這裏計算的iou其實只有粗略的計算，沒有像Fast RCNN那樣每個anchors都拿去算，這裡只有算模板
            # 這裏只把target的寬高拿出來，因為我們是把anchor與target左上角對其後計算iou
            j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
            # t.repeat(na, 1, 1): [nt, 6] -> [3, nt, 6]
            # 获取正样本对应的anchor模板与target信息
            # a會保存哪個anchor有對上gt_box
            # t會保存這個anchor對應上的gt_box的座標
            # 因為t的第一個index會保存是這個batch中的第幾張照片所以不會搞混
            # a與t要共同使用，兩個對應上的index是指同一件事情
            # 這樣我們可以透過a與t獲得正樣本對應上的anchor以及對應上的gt_box信息
            # 這裡我們不知道具體是在特徵圖上的哪個pixel的anchor，只知道是哪種的anchor
            a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

        # Define
        # long等于to(torch.int64), 数值向下取整
        # 把前兩個元素提取出來
        b, c = t[:, :2].long().T  # image_idx, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        # 這裏offset是0
        # 這裏gij透過向上取整可以還原到特徵圖上的pixel，也就可以找到坐上角的座標
        # 假設原先在特徵圖上中心座標為[3.6, 4.3]透過向上取整[3, 4]就可以把這個當作anchor的左上角座標
        # 也就是把這個gt_box指認到應該要辨別出他的anchor上，負責辨別他的anchor
        gij = (gxy - offsets).long()  # 匹配targets所在的grid cell左上角坐标
        # 把x座標給gi，把y座標給gj
        # 需要轉至後才能分i與j，原始shape[gt_box, 2] => [2, gt_box]
        gi, gj = gij.T  # grid xy indices

        # Append
        # gain[3]: grid_h, gain[2]: grid_w
        # indices把相關資料整理起來
        # 中心點座標透過clamp將範圍限制在特徵圖上
        # image_idx, anchor_idx, grid indices(y, x)
        indices.append((b, a, gj.clamp_(0, gain[3]-1), gi.clamp_(0, gain[2]-1)))
        # 計算出從左上角到對應gt_box的中心的偏移量，也就是我們期望的中心點回歸參數
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # gt box相对anchor的x,y偏移量以及w,h
        # 把我們每個有對應上正樣本的anchors匹配上，也就是我們如果要找indices[0]對應上的anchor就可以調用anch[0]
        # 因為indices的a只有存第幾號anchor但是後面會有不同層的yolo相同index會有不同大小的anchor所以現在把完整anchor保留
        anch.append(anchors[a])  # anchors
        # 把對應上的class保存
        tcls.append(c)  # class
        # 做檢查
        if c.shape[0]:  # if any targets
            # 目标的标签数值不能大于给定的目标类别数
            assert c.max() < model.nc, 'Model accepts %g classes labeled from 0-%g, however you labelled a class %g. ' \
                                       'See https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data' % (
                                           model.nc, model.nc - 1, c.max())
    # tcls = 每個正樣本的正確class
    # tbox = 每個正樣本的正確中心寬高的回歸參數
    # indices = 每個正樣本在特徵圖上的pixel，此batch中的哪張照片
    # anch = 每個正樣本對應上的anchor長寬，這裡的長寬是相對大小
    return tcls, tbox, indices, anch


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6,
                        multi_label=True, classes=None, agnostic=False, max_num=100):
    """
    Performs  Non-Maximum Suppression on inference results

    param: prediction[batch, num_anchors, (num_classes+1+4) x num_anchors]
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """
    # 已看過
    # 這裡主要在做的就是NMS非極大值抑制

    # Settings
    # prediction shape [batch_size, total_anchor, (5 + num_classes)]
    merge = False  # merge for best mAP
    # 設定最大最小寬高
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    # 超過設定時間會直接停止
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    # 取出多少個分類
    nc = prediction[0].shape[1] - 5  # number of classes
    # multi_label = 看是不是多分類問題
    multi_label &= nc > 1  # multiple labels per box
    # output shape [batch_size] 預設都為None，應該是要分不同圖片進行輸出
    output = [None] * prediction.shape[0]
    # 遍歷這次batch的每一張圖片
    # predict一次就只會給一張，如果是在test一次就會給batch_size張
    for xi, x in enumerate(prediction):  # image index, image inference 遍历每张图片
        # xi 代表現在的index
        # x shape [total_anchors, (5 + num_classes)]
        # Apply constraints
        # 把obj小於conf_thres(0.1)的anchor排除，conf_thres是由giou ratio計算出來的
        x = x[x[:, 4] > conf_thres]  # confidence 根据obj confidence虑除背景目标
        # 刪除過小目標或是過大目標，應該不可能會有過大目標，因為max_wh是1024我們輸入的圖片只有416
        x = x[((x[:, 2:4] > min_wh) & (x[:, 2:4] < max_wh)).all(1)]  # width-height 虑除小目标

        # If none remain process next image
        # 如果已經沒有半個anchor就continue跳到下張圖片
        if not x.shape[0]:
            continue

        # Compute conf
        # 計算置信度，這裡由分類的概率乘上giou ratio當作置信度
        x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        # box = 轉換一下，把格式從中心(x, y)寬高，轉成左上角座標以及右下角座標
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:  # 针对每个类别执行非极大值抑制
            # .nonzero返回的是在這個tensor中那些位置不是零的(x, y)
            # i = x座標, j = y座標
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).t()
            # box[i] = 找到所屬的anchor
            # x[i, j + 5] = 找到所屬的anchor中的哪個類別
            # j = 記錄下預測出來的類別
            # 把上面三個東西在維度1上做拼接最後可以得到 x shape [預測類別大於conf_thres的anchor, 6]
            # 6 = 邊界匡座標(4) +  預測目標概率(1) + 目標標籤(1)
            x = torch.cat((box[i], x[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:  # best class only  直接针对每个类别中概率最大的类别进行非极大值抑制处理
            # 這裡沒有用
            conf, j = x[:, 5:].max(1)
            x = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]

        # Filter by class
        if classes:
            # 這裡沒有用
            x = x[(j.view(-1, 1) == torch.tensor(classes, device=j.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        # 看一下經過篩選後還剩下多少匡
        n = x.shape[0]  # number of boxes
        if not n:
            # 如果已經沒有目標匡了就continue跳過這張照片
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        # 這裡的NMS是對於類別進行，也就是分類到同一類別的才會做NMS
        # c這裡我們走else部分，拿到的就會是分類的類別
        c = x[:, 5] * 0 if agnostic else x[:, 5]  # classes
        # boxes對於每種分類類別做一定的偏移，這樣可以一次做完全部的NMS，這個偏移可以讓不同分類的anchor不會互相重疊
        # score還是原本的score
        boxes, scores = x[:, :4].clone() + c.view(-1, 1) * max_wh, x[:, 4]  # boxes (offset by class), scores
        # 使用pytorch官方給的nms處理方法，這邊設定成大於iou_thres(0.6)就會進行nms處理
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        # 最多保留max_num個框框在最後圖片上
        i = i[:max_num]  # 最多只保留前max_num个目标信息
        # 這裡merge預設是False
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                # i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        # 把結果放到output中
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    # 回傳最後結果 [通過nms後的anchor數量, 6] 6 = 上面有提過
    return output


def get_yolo_layers(model):
    # 已看過
    bool_vec = [x['type'] == 'yolo' for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]  # [82, 94, 106] for yolov3


def kmean_anchors(path='./data/coco64.txt', n=9, img_size=(640, 640), thr=0.20, gen=1000):
    # Creates kmeans anchors for use in *.cfg files: from build_utils.build_utils import *; _ = kmean_anchors()
    # n: number of anchors
    # img_size: (min, max) image size used for multi-scale training (can be same values)
    # thr: IoU threshold hyperparameter used for training (0.0 - 1.0)
    # gen: generations to evolve anchors using genetic algorithm
    from build_utils.datasets import LoadImagesAndLabels

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        iou = wh_iou(wh, torch.Tensor(k))
        max_iou = iou.max(1)[0]
        bpr, aat = (max_iou > thr).float().mean(), (iou > thr).float().mean() * n  # best possible recall, anch > thr
        print('%.2f iou_thr: %.3f best possible recall, %.2f anchors > thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, IoU_all=%.3f/%.3f-mean/best, IoU>thr=%.3f-mean: ' %
              (n, img_size, iou.mean(), max_iou.mean(), iou[iou > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    def fitness(k):  # mutation fitness
        iou = wh_iou(wh, torch.Tensor(k))  # iou
        max_iou = iou.max(1)[0]
        return (max_iou * (max_iou > thr).float()).mean()  # product

    # Get label wh
    wh = []
    dataset = LoadImagesAndLabels(path, augment=True, rect=True)
    nr = 1 if img_size[0] == img_size[1] else 10  # number augmentation repetitions
    for s, l in zip(dataset.shapes, dataset.labels):
        wh.append(l[:, 3:5] * (s / s.max()))  # image normalized to letterbox normalized wh
    wh = np.concatenate(wh, 0).repeat(nr, axis=0)  # augment 10x
    wh *= np.random.uniform(img_size[0], img_size[1], size=(wh.shape[0], 1))  # normalized to pixels (multi-scale)
    wh = wh[(wh > 2.0).all(1)]  # remove below threshold boxes (< 2 pixels wh)

    # Kmeans calculation
    from scipy.cluster.vq import kmeans
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.Tensor(wh)
    k = print_results(k)

    # # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    for _ in tqdm(range(gen), desc='Evolving anchors'):
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            print_results(k)
    k = print_results(k)

    return k
