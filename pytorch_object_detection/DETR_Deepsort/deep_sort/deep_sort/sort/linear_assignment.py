# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
# The linear sum assignment problem is also known as minimum weight matching in bipartite graphs.
from scipy.optimize import linear_sum_assignment as linear_assignment
from . import kalman_filter


INFTY_COST = 1e+5

# min_cost_matching 使用匈牙利算法解决线性分配问题。
# 传入 门控余弦距离成本 或 iou cost


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    # 已看過
    # track_indices與detection_indices都會有東西傳入

    # track_indices = 在tracks當中，還未找到匹配的detection的track index
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    # detection_indices = 這一張圖中有標註的但是沒有匹配上的標註匡的index
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    # 如果其中之一沒有東西表示沒有東西可以match所以直接離開
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    # 计算成本矩阵
    # 將tracks以及detections內容傳入同時也將對應的index傳入，去計算成本
    # cost_matrix shape = ndarray [len(track_indices), len(tracks)]，裡面的值就是iou的cost
    # 這裡不一定會是計算iou需要看distance_metric傳入的不同會有不同的計算方式
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)
    # 將cost大於閾值的地方全部變成無限大
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    # 执行匈牙利算法，得到指派成功的索引对，行索引为tracks的索引，列索引为detections的索引
    # 透過匈牙利算法可以獲得最佳匹配方式，這個算法如果有問題可以到detr中看
    # 這裡會依據tracks與detections的數量決定輸出的shape，也就是說會依照較少的為主
    # row_indices, col_indices shape [min(len(detection_indices), len(tracks_indices))]
    row_indices, col_indices = linear_assignment(cost_matrix)

    # 構建三個要回傳的東西，暫時先為空
    matches, unmatched_tracks, unmatched_detections = [], [], []
    # 如果有在row_indices表示該tracks有對應到
    # 如果有在col_indices表示該detections有對應到

    # 找出未匹配的detections
    for col, detection_idx in enumerate(detection_indices):
        # 這裡我們是用遍歷中的index
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    # 找出未匹配的tracks
    for row, track_idx in enumerate(track_indices):
        # 這裡我們是用遍歷中的index
        if row not in row_indices:
            unmatched_tracks.append(track_idx)

    # 遍历匹配的(track, detection)索引对
    for row, col in zip(row_indices, col_indices):
        # 取出匹配上的track以及detection
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        # 如果相应的cost大于阈值max_distance，也视为未匹配成功
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            # 如果有匹配到的話就放到匹配的地方去
            matches.append((track_idx, detection_idx))
    # matches = 有匹配到的track以及detection，兩兩會一組所以裡面會是一個tuple，shape List[tuple(track_idx, detection_idx)]
    # unmatched_tracks = 沒有匹配到的track，shape List[]，裡面的值會是track的index
    # unmatched_detections = 沒有匹配到的detections，shape List[]，裡面的值會是detections的index
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
        距离度量：
        输入：一个轨迹和检测列表，以及一个N个轨迹索引和M个检测索引的列表。 
        返回：NxM维的代价矩阵，其中元素(i，j)是给定轨迹索引中第i个轨迹与
        给定检测索引中第j个检测之间的关联成本。
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
        门控阈值。成本大于此值的关联将被忽略。
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
        级联深度应设置为最大轨迹寿命。
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
        当前时间步的预测轨迹列表。
    detections : List[detection.Detection]
        A list of detections at the current time step.
        当前时间步的检测列表。
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
        轨迹索引列表，用于将 cost_matrix中的行映射到tracks的
         轨迹（请参见上面的说明）。 默认为所有轨迹。
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.
        将 cost_matrix中的列映射到的检测索引列表
         detections中的检测（请参见上面的说明）。 默认为全部检测。

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    返回包含以下三个条目的元组：
    
    匹配的跟踪和检测的索引列表，
    不匹配的轨迹索引的列表，
    未匹配的检测索引的列表。

    """
    
    # 分配track_indices和detection_indices两个列表
    # tracks是已經在追蹤的物體，detections是這次圖像輸出的物體

    # track_indices是已經在tracks當中且為已經鎖定的index
    # track_indices都會是已經有給的了，所以不會在這裡生成
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    # detection_indices通常不會給，所以會在這裡生成
    if detection_indices is None:
        # 構建出一個list裡面的值從0到這次檢測出來的框框數量減1
        detection_indices = list(range(len(detections)))

    # 初始化匹配集matches M ← ∅ 
    # 未匹配检测集unmatched_detections U ← D
    # 將detection_indices的值放到unmatched_detections裏面
    unmatched_detections = detection_indices
    # 製造一個matches空間，這裡放的會是已經追縱到的且鎖定的
    matches = []
    # 由小到大依次对每个level的tracks做匹配，也就是說我們會先對最近有配對上的追蹤目標進行配對
    # cascade_depth = 最大軌跡壽命
    for level in range(cascade_depth):
        # 如果没有detections，退出循环
        # 我們會用detections的東西去匹配，所以如果這幀沒有檢測到任何目標就會直接跳出
        if len(unmatched_detections) == 0:  # No detections left
            break

        # time_since_update紀錄的是從上次對該追蹤目標有匹配上過了多久
        # 当前level的所有tracks索引
        # 步骤6：Select tracks by age
        # 最一開始tracks會是空的
        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        # 如果当前level没有track，继续
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue
            
        # 步骤7：调用min_cost_matching函数进行匹配
        # 這裡我們只會在意有匹配到配對以及沒有匹配到的detections，沒有匹配到的track就不管了
        # unmatched_detections會進到下輪繼續匹配，已經有匹配上的就會被拿掉
        matches_l, _, unmatched_detections = \
            min_cost_matching(
                distance_metric, max_distance, tracks, detections,
                track_indices_l, unmatched_detections)
        # 步骤8
        # 將有配對到的組合記錄下來
        matches += matches_l
    # 除了有匹配到的track剩下的就是為匹配的
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))  # 步骤9
    return matches, unmatched_tracks, unmatched_detections


'''
门控成本矩阵：通过计算卡尔曼滤波的状态分布和测量值之间的距离对成本矩阵进行限制，
成本矩阵中的距离是track和detection之间的外观相似度。
如果一个轨迹要去匹配两个外观特征非常相似的 detection，很容易出错；
分别让两个detection计算与这个轨迹的马氏距离，并使用一个阈值gating_threshold进行限制，
就可以将马氏距离较远的那个detection区分开，从而减少错误的匹配。
'''


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
        代价矩阵中与不可行关联相对应的条目设置此值。 默认为一个很大的值。
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.
        如果为True，则在门控期间仅考虑状态分布的x，y位置。默认为False。

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """

    # kf = 卡爾曼濾波實例對象
    # cost_matrix = cost矩陣，由該追蹤對象先前的特徵向量與當前標註匡的特徵向量，匹配上需要的cost
    # cost_matrix shape [track_previous_feature, num_detections]
    # tracks = 所有正在追蹤的對象
    # detections = 這一幀有追蹤到的對象
    # track_indices = 還沒有被匹配到的track的index
    # detection_indices = 還沒有被匹配到的detection的index
    # gated_cost = 閾值，這裡預設都會是無窮大
    # only_position = 預設為False，就是看是否只關注位置部分

    # 根据通过卡尔曼滤波获得的状态分布，使成本矩阵中的不可行条目无效。
    # gating_dim預設會是4
    gating_dim = 2 if only_position else 4  # 测量空间维度
    # 马氏距离通过测算检测与平均轨迹位置的距离超过多少标准差来考虑状态估计的不确定性。
    # 通过从逆chi^2分布计算95%置信区间的阈值，排除可能性小的关联。
    # 四维测量空间对应的马氏阈值为9.4877
    # 從kalman_filter.chi2inv95取出我們需要的數值，gating_threshold=9.4877
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # 取出還沒有匹配上的detection的座標訊息(center_x, center_y, ratio, height)
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    # 遍歷需要配匹的追蹤目標
    for row, track_idx in enumerate(track_indices):
        # 將追蹤目標取出來
        track = tracks[track_idx]
        # KalmanFilter.gating_distance 计算状态分布和测量之间的选通距离
        # gating_distance shape = ndarray [len(measurements)]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        # 將距離大於閾值的部分設定成無窮
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
