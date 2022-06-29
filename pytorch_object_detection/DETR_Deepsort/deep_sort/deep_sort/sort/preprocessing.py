# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """
    :param boxes: shape [num_boxes, 4] (xmin, ymin, width, height)
    :param max_bbox_overlap: 閾值，用在nms當中
    :param scores: shape [num_boxes]
    :return:
    """
    # 已看過

    # 如果沒有標註匡就直接返回空列表
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    # 最後回傳的內容
    pick = []

    # 取得座標(xmin, ymin, xmax, ymax)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    # 計算標註匡面積
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 根據置信度分數或是ymax進行排序，這裡回傳的idxs的內容會是index，依據index就可以排序
    # 這裡的排序是由小到大，所以會先捨棄掉的會是置信度較小的標註匡
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        # 取出idxs的最後一個
        last = len(idxs) - 1
        # 取得對應的index
        i = idxs[last]
        # 添加在pick裡面，表示這個標註匡我們一定會選
        pick.append(i)

        # 將當前的座標與所有還沒有被遍歷到的座標進行比對
        # 這裡做的事情跟計算iou相同，只是這裡一次與多個標註匡進行重疊計算
        # xx1, yy1, xx2, yy2 shape [len(idxs-1)]
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 計算重合部分的高寬
        # w, h shape [len(idxs-1)]
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # IOU
        # 計算當前標註匡與剩下的標註匡之間的iou值
        overlap = (w * h) / area[idxs[:last]]

        # 將當前標註匡以及重疊大於設定閾值的標註匡移除idxs
        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    # 回傳經過非極大值抑制後哪些index的標註匡是可以留下的
    # pick shape [num_boxes]
    return pick
