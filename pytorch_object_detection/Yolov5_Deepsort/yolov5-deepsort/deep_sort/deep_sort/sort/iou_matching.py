# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def iou(bbox, candidates):
    # 计算两个框的IOU
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    # bbox = 正在追蹤的對象上一時刻的座標位置 shape ndarray [4]
    # candidates = 圖像的標註匡，透過yolo給出的座標位置 shape ndarray [num_predict_box, 4]

    # bbox_tl = (xmin, ymin)，bbox_br = (xmax, ymax)
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    # 與bbox相同，只是會是多個
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    # np.c_ Translates slice objects to concatenation along the second axis.
    # np.newaxis = None的別名，也就是會擴維
    # tl, br shape = [num_predict_box, 2]，透過這兩個就可以計算重疊面積
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    # 計算重疊部分的高寬shape [num_predict_box, 2]
    wh = np.maximum(0., br - tl)

    # 高寬相乘shape [num_predict_box]，這裡的值就是重疊面積
    area_intersection = wh.prod(axis=1)
    # 匹配匡面積
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    # iou值shape [num_predict_box]
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    # 已看過
    # 將tracks以及detections內容傳入同時也將對應的index傳入，去計算成本
    # 计算tracks和detections之间的IOU距离成本矩阵
    """An intersection over union distance metric.

    用于计算tracks和detections之间的iou距离矩阵

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    # 理論上來說都會傳入這兩個陣列，所以不會需要我們手動初始化
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # 構建一個N*M大小的矩陣且全為0，到時用來表示一個track_index對應上一個detection_index需要的cost
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    # 遍歷整個track_indices內容
    for row, track_idx in enumerate(track_indices):
        # 如果該追蹤對象的time_since_update大於一就直接將值設定成無窮大
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        # 取出追蹤對象上次的座標位置並且轉成(xmin, ymin, width, height)格式
        bbox = tracks[track_idx].to_tlwh()
        # 取出檢測到的匡同時將座標轉成(xmin, ymin, width, height)格式
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        # 將匡數入計算iou，之後再用1-就會獲的成本值
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix
