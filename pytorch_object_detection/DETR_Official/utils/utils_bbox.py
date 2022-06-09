import numpy as np
import torch
from torchvision.ops import nms, boxes


def yolo_correct_boxes(outputs, input_shape, image_shape, letterbox_image):
    # 已看過
    # ---------------------------------------------------------
    # outputs (Dict)
    # {
    #   'pred_logits': shape [batch_size, num_queries, num_classes + 1],
    #   'pred_boxes': shape [batch_size, num_queries, 4]
    # }
    # 預測匡是相對座標
    # ---------------------------------------------------------
    # image_shape = 原始圖像大小, letterbox_image = 有沒有用灰邊
    # -----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    # -----------------------------------------------------------------#
    # 顯取出中心點座標以及寬高，這裡要將設備轉到cpu上並且轉成numpy格式
    box_xy = outputs['pred_boxes'][:, :, 0:2].cpu().numpy()
    box_wh = outputs['pred_boxes'][:, :, 2:4].cpu().numpy()
    # 倒過來取
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    # 轉成numpy格式
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    # 這裡傳入的box是相對位置而且是加過灰邊後的也就是有被經過預處理的
    # 我們是先把圖片做resize後傳入網路中，所以網路中就是拿resize過的圖片進行預測邊界匡
    # 所以很合理的再放回原圖的時候要先脫去resize這一步驟再還原到原圖上
    if letterbox_image:
        # -----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        # -----------------------------------------------------------------#
        # new_shape就是我們把最大邊的長度變成input_shape長度後，原始圖像的高寬會變成多少
        # 這裡的動作與加上灰邊時是一樣的，只是壓縮成一行
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        # 首先我們先看對於左上角偏移了多少，除以2的原因是因為灰邊會在上下都有或是左右都有
        # 後面還有一個input_shape是因為這裡的gt_box是相對位置
        offset = (input_shape - new_shape) / 2. / input_shape
        # 看一下從new_shape到input_shape被拉伸了多少
        scale = input_shape / new_shape

        # 把邊匡部分去除，縮放回沒有resize的大小
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    # 最後在縮放回原圖
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes
