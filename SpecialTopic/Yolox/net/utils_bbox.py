import numpy as np
import torch
from torchvision.ops import nms, boxes


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    # 已看過
    # box_xy = (center_x, center_y), box_wh = (w, h), input_shape = 原始圖像經過縮放的大小
    # image_shape = 原始圖像大小, letterbox_image = 有沒有用灰邊
    # -----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    # -----------------------------------------------------------------#
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


def decode_outputs(outputs, input_shape):
    # 已看過
    grids = []
    strides = []
    # 記錄下每個特徵圖的大小
    hw = [x.shape[-2:] for x in outputs]
    # ---------------------------------------------------#
    #   outputs输入前代表每个特征层的预测结果
    #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 4 + 1 + num_classes, 6400
    #   batch_size, 5 + num_classes, 40, 40
    #   batch_size, 5 + num_classes, 20, 20
    #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
    #   堆叠后为batch_size, 8400, 5 + num_classes
    # ---------------------------------------------------#
    # 先將圖片的高寬展平，將展平後的維度進行拼接，在條整一下順序
    # 最後變成[每張圖片, 每個特整點, 對應上的預測結果]
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    # ---------------------------------------------------#
    #   获得每一个特征点属于每一个种类的概率
    # ---------------------------------------------------#
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    # 上面有記錄下每個輸出特徵圖的高和寬
    for h, w in hw:
        # ---------------------------#
        #   根据特征层的高宽生成网格点
        # ---------------------------#
        grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
        # ---------------------------#
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400, 2
        # ---------------------------#
        grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        # shape會是[batch_size, h * w]
        shape = grid.shape[:2]

        grids.append(grid)
        # 這裡strides會記錄下這個grid對應上原圖的縮放比例，並且stride的shape[shape[0], shape[1], 1] = [batch_size, w * h, 1]
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
    # ---------------------------#
    #   将网格点堆叠到一起
    #   1, 6400, 2
    #   1, 1600, 2
    #   1, 400, 2
    #
    #   1, 8400, 2
    # ---------------------------#
    # 把每個特徵圖的特徵點做拼接
    grids = torch.cat(grids, dim=1).type(outputs.type())
    strides = torch.cat(strides, dim=1).type(outputs.type())
    # ------------------------#
    #   根据网格点进行解码
    # ------------------------#
    # 原始左上角點加上偏移，還有對長寬做變化
    # 最後乘上特徵圖對應原圖的比例就可以縮放回原圖大小
    # 這裡先縮放回原圖大小是因為現在的數字都是相對於自己當前的特徵圖的，所以我們現在先把他放回原圖上
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    # -----------------#
    #   归一化
    # -----------------#
    # 再把他變回相對座標
    outputs[..., [0, 2]] = outputs[..., [0, 2]] / input_shape[1]
    outputs[..., [1, 3]] = outputs[..., [1, 3]] / input_shape[0]
    # outputs shape [batch_size, total_feature_pixel, 5 + num_classes]
    return outputs


def non_max_suppression(prediction, num_classes, input_shape, image_shape,
                        letterbox_image, conf_thres=0.5, nms_thres=0.4):
    # 已看過
    """
    :param prediction: shape [batch_size, total_feature_pixel, 5 + num_classes]
    :param num_classes: num_classes
    :param input_shape: 指定縮放大小預設[640, 640]
    :param image_shape: 原始圖片大小
    :param letterbox_image: 有沒有加上灰邊
    :param conf_thres: 置信度需大於這個值才會被認為是有目標物的
    :param nms_thres: nms閾值
    :return:
    """
    # ----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    # ----------------------------------------------------------#
    # 原先輸入的是(center_x, center_y, w, h)
    # 現在要轉換成(xmin, ymin, xmax, ymax)
    # 先複製出一個shape一樣的表
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    # 置信度保持不變
    prediction[:, :, :4] = box_corner[:, :, :4]

    # 構建出一個output列表，這個會是最後的return東西，先暫時設成None表示沒有框框
    output = [None for _ in range(len(prediction))]
    # ----------------------------------------------------------#
    #   对输入图片进行循环，一般只会进行一次
    # ----------------------------------------------------------#
    for i, image_pred in enumerate(prediction):
        # ----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度，看清楚是種類
        #   class_pred  [num_anchors, 1]    种类
        # ----------------------------------------------------------#
        # class_conf在這裡是指對於覺得是這個類別的概率，class_pred是說這個是哪個index
        # 這個索引拿去對照就可以知道是哪個類別了
        # 一個存的是值另一個是索引
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        # ----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        # ----------------------------------------------------------#
        # 我們這裡用框框內是否有目標的預測概率乘上分類類別概率來看有沒有超過閾值
        # conf_mask裡面的代表超過閾值也就是他是我們想要的框框
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        if not image_pred.size(0):
            continue
        # -------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        # -------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        # 把大於閾值的部分篩選出來
        detections = detections[conf_mask]

        # 來自torchvision官網說明num的用法
        # boxes (Tensor[N, 4]) – boxes where NMS will be performed.
        # They are expected to be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
        # scores (Tensor[N]) – scores for each one of the boxes
        # idxs (Tensor[N]) – indices of the categories for each one of the boxes.
        # iou_threshold (float) – discards all overlapping boxes with IoU > iou_threshold
        # 這裡感覺比之前在另一個模板更加聰明了
        nms_out_index = boxes.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thres,
        )

        # 把通過nms處理過後的資料過濾出來，作為最後會輸出在圖片上的框框
        output[i] = detections[nms_out_index]

        # #------------------------------------------#
        # #   获得预测结果中包含的所有种类
        # #------------------------------------------#
        # unique_labels = detections[:, -1].cpu().unique()

        # if prediction.is_cuda:
        #     unique_labels = unique_labels.cuda()
        #     detections = detections.cuda()

        # for c in unique_labels:
        #     #------------------------------------------#
        #     #   获得某一类得分筛选后全部的预测结果
        #     #------------------------------------------#
        #     detections_class = detections[detections[:, -1] == c]

        #     #------------------------------------------#
        #     #   使用官方自带的非极大抑制会速度更快一些！
        #     #------------------------------------------#
        #     keep = nms(
        #         detections_class[:, :4],
        #         detections_class[:, 4] * detections_class[:, 5],
        #         nms_thres
        #     )
        #     max_detections = detections_class[keep]
            
        #     # # 按照存在物体的置信度排序
        #     # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
        #     # detections_class = detections_class[conf_sort_index]
        #     # # 进行非极大抑制
        #     # max_detections = []
        #     # while detections_class.size(0):
        #     #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
        #     #     max_detections.append(detections_class[0].unsqueeze(0))
        #     #     if len(detections_class) == 1:
        #     #         break
        #     #     ious = bbox_iou(max_detections[-1], detections_class[1:])
        #     #     detections_class = detections_class[1:][ious < nms_thres]
        #     # # 堆叠
        #     # max_detections = torch.cat(max_detections).data
            
        #     # Add max detections to outputs
        #     output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
        
        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            # box_xy = (center_x, center_y), box_wh = (w, h), input_shape = 原始圖像經過縮放的大小
            # image_shape = 原始圖像大小, letterbox_image = 有沒有用灰邊
            # 這邊的座標已經是縮放回原圖大小的而且是左上右下的座標
            output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    # output 7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
    return output
