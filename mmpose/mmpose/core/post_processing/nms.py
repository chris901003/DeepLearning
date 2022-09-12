# ------------------------------------------------------------------------------
# Adapted from https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
# Original licence: Copyright (c) Microsoft, under the MIT License.
# ------------------------------------------------------------------------------

import numpy as np


def nms(dets, thr):
    """Greedily select boxes with high confidence and overlap <= thr.

    Args:
        dets: [[x1, y1, x2, y2, score]].
        thr: Retain overlap < thr.

    Returns:
         list: Indexes to keep.
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]

    return keep


def oks_iou(g, d, a_g, a_d, sigmas=None, vis_thr=None):
    """Calculate oks ious.

    Args:
        g: Ground truth keypoints.
        d: Detected keypoints.
        a_g: Area of the ground truth object.
        a_d: Area of the detected object.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.

    Returns:
        list: The oks ious.
    """
    """ 計算oks的iou
    Args:
        g: 當前人物關節點資訊，ndarray shape [num_joints * 3]
        d: 剩下的人物關節點資訊，ndarray shape [num_people, num_joints * 3]
        a_g: 當前人物的面積
        a_d: 剩下人物的面積
        sigmas: 關鍵點標註的標準差
        vis_thr: 關鍵點可見性閾值
    """
    if sigmas is None:
        # 如果沒有給定sigmas資訊就會是默認的值
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0
    # 進行計算iou值
    vars = (sigmas * 2)**2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros(len(d), dtype=np.float32)
    for n_d in range(0, len(d)):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx**2 + dy**2) / vars / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if vis_thr is not None:
            ind = list(vg > vis_thr) and list(vd > vis_thr)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / len(e) if len(e) != 0 else 0.0
    return ious


def oks_nms(kpts_db, thr, sigmas=None, vis_thr=None, score_per_joint=False):
    """OKS NMS implementations.

    Args:
        kpts_db: keypoints.
        thr: Retain overlap < thr.
        sigmas: standard deviation of keypoint labelling.
        vis_thr: threshold of the keypoint visibility.
        score_per_joint: the input scores (in kpts_db) are per joint scores

    Returns:
        np.ndarray: indexes to keep.
    """
    """ OKS極大值抑制算法
    Args:
        kpts_db: 關節點資訊，包含關節點座標以及總體置信度以及面積
        thr: 閾值
        sigmas: 關鍵點標註的標準差
        vis_thr: 讓關節點顯示出的閾值
        score_per_joint: 輸入的score是每個關節點或是整體的score
    """
    if len(kpts_db) == 0:
        # 如果沒有檢測到關節點就會直接回傳空
        return []

    if score_per_joint:
        # 如果輸入的score是依據每個關節點就會到這裡，進行整體的平均值
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        # 如果輸入的score是整體的平均就會到這裡，直接進行保存
        scores = np.array([k['score'] for k in kpts_db])

    # 將關節點資訊進行攤平，kpts shape [num_people, num_joints * 3]
    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    # 將面積資訊提取出來
    areas = np.array([k['area'] for k in kpts_db])

    # 將置信度進行排序
    order = scores.argsort()[::-1]

    # 最終回傳的list
    keep = []
    # 從置信度大的開始進行遍歷
    while len(order) > 0:
        # 獲取還未處理中置信度最大的index
        i = order[0]
        # 將該index保存到keep當中表示該人物是有效人物
        keep.append(i)

        # 透過oks_iou進行計算
        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, vis_thr)

        # 獲取與目前標註到不相同人的index
        inds = np.where(oks_ovr <= thr)[0]
        # 將order進行更新
        order = order[inds + 1]

    # 將keep轉成ndarray
    keep = np.array(keep)

    # 將結果回傳，當中表示要使用哪些index的資料
    return keep


def _rescore(overlap, scores, thr, type='gaussian'):
    """Rescoring mechanism gaussian or linear.

    Args:
        overlap: calculated ious
        scores: target scores.
        thr: retain oks overlap < thr.
        type: 'gaussian' or 'linear'

    Returns:
        np.ndarray: indexes to keep
    """
    assert len(overlap) == len(scores)
    assert type in ['gaussian', 'linear']

    if type == 'linear':
        inds = np.where(overlap >= thr)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(-overlap**2 / thr)

    return scores


def soft_oks_nms(kpts_db,
                 thr,
                 max_dets=20,
                 sigmas=None,
                 vis_thr=None,
                 score_per_joint=False):
    """Soft OKS NMS implementations.

    Args:
        kpts_db
        thr: retain oks overlap < thr.
        max_dets: max number of detections to keep.
        sigmas: Keypoint labelling uncertainty.
        score_per_joint: the input scores (in kpts_db) are per joint scores

    Returns:
        np.ndarray: indexes to keep.
    """
    if len(kpts_db) == 0:
        return []

    if score_per_joint:
        scores = np.array([k['score'].mean() for k in kpts_db])
    else:
        scores = np.array([k['score'] for k in kpts_db])

    kpts = np.array([k['keypoints'].flatten() for k in kpts_db])
    areas = np.array([k['area'] for k in kpts_db])

    order = scores.argsort()[::-1]
    scores = scores[order]

    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0
    while len(order) > 0 and keep_cnt < max_dets:
        i = order[0]

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]],
                          sigmas, vis_thr)

        order = order[1:]
        scores = _rescore(oks_ovr, scores[1:], thr)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]

    return keep