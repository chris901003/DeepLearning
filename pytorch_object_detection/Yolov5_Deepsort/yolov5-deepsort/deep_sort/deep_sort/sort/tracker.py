# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
        测量与轨迹关联的距离度量
    max_age : int
        Maximum number of missed misses before a track is deleted.
        删除轨迹前的最大未命中数
    n_init : int
        Number of frames that a track remains in initialization phase.
        确认轨迹前的连续检测次数。如果前n_init帧内发生未命中，则将轨迹状态设置为Deleted
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        # 已看過
        # metric = matching實例化對象
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        # 多少次沒有追蹤到就丟棄
        self.max_age = max_age
        # 連續多少次追蹤到新的對象，可以將對象變成確定態
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()  # 实例化卡尔曼滤波器
        self.tracks = []   # 保存一个轨迹列表，用于保存一系列轨迹
        self._next_id = 1  # 下一个分配的轨迹id
 
    def predict(self):
        """Propagate track state distributions one time step forward.
        将跟踪状态分布向前传播一步

        This function should be called once every time step, before `update`.
        """
        # 已看過
        # 預測已經在追蹤的對象的下一個位置，將卡爾曼濾波器傳入
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.
        执行测量更新和轨迹管理

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # 已看過
        # detections就是新一次的標注內容，一個list裡面是Detection格式
        # Run matching cascade.
        # matches = 匹配的跟踪和检测的索引列表，shape List[tuple(track_index, detection_index)]
        # unmatched_tracks = 不匹配的轨迹索引的列表，原先有在追蹤但是這一幀消失，shape List[track_index]
        # unmatched_detections = 未匹配的检测索引的列表，這一幀有標註到但是不在之前追蹤列表上面，shape List[detection_index]
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # 更新追蹤內容
        # Update track set.
        
        # 1. 针对匹配上的结果
        # 遍歷有配對上的追蹤目標
        for track_idx, detection_idx in matches:
            # 更新tracks中相应的detection
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        
        # 2. 针对未匹配的track, 调用mark_missed进行标记
        # track失配时，若Tantative则删除；若update时间很久也删除
        # 依據追蹤對象狀態更新追蹤對象的狀態
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # 3. 针对未匹配的detection， detection失配，进行初始化
        for detection_idx in unmatched_detections:
            # 進行初始化，會將這些需要追蹤的物體放入到self.tracks當中作為接下來要追蹤的對象
            self._initiate_track(detections[detection_idx])
        
        # 得到最新的tracks列表，保存的是标记为Confirmed和Tentative的track
        # 更新tracks列表，將已經離開畫面的物體從列表中刪除，只保留須要追蹤的目標
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        # 將已經在追蹤的目標Track拿出來
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # features = 已確認追蹤對象的特徵向量
        # targets = 每一個特徵向量對應上的對象id
        # 透過上面敘述可以知道len(features)跟len(targets)會是一樣的
        features, targets = [], []
        # 遍歷須要追蹤的目標
        for track in self.tracks:
            # 获取所有Confirmed状态的track id
            # 非確認狀態的追蹤對象會直接跳過
            if not track.is_confirmed():
                continue
            # 将Confirmed状态的track的features添加到features列表
            # track.features是一個list裡面存了每一幀的特徵向量
            features += track.features
            # 获取每个feature对应的trackid，這裡自己稍微注意一下到底存了什麼進去
            targets += [track.track_id for _ in track.features]
            # 將追蹤對象的特徵向量清除
            track.features = []
        # 距离度量中的特征集更新
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        # 已看過
        # detections就是新一次的標注內容，一個list裡面是Detection格式

        def gated_metric(tracks, dets, track_indices, detection_indices):
            """
            :param tracks: 正在追蹤的對象列表
            :param dets: 這一幀有匡選到的目標
            :param track_indices: 要計算的正在追蹤對象的index
            :param detection_indices: 要計算的有匡選的目標
            :return:
            """

            # 將要匹配的匡選對象的特徵向量取出來，shape [detection_indices, channel]
            features = np.array([dets[i].feature for i in detection_indices])
            # 取出要匹配的的追蹤對象的追蹤id，shape [track_indices]
            targets = np.array([tracks[i].track_id for i in track_indices])
            
            # 通过最近邻（余弦距离）计算出成本矩阵（代价矩阵）
            # cost_matrix shape [len(track_indices), len(detection_indices)]
            cost_matrix = self.metric.distance(features, targets)
            # 计算门控后的成本矩阵（代价矩阵）
            # 透過計算距離後會將距離大於閾值部分的cost設定成無窮
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # 区分开confirmed tracks和unconfirmed tracks
        # 將正在追蹤的目標分成已經鎖定上去的以及有追蹤到但是為鎖定的
        # 這裡存的會是tracks的index
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        # 第一階段的匹配
        # 对确定态的轨迹进行级联匹配，得到匹配的tracks、不匹配的tracks、不匹配的detections
        # matching_cascade 根据特征将检测框匹配到确认的轨迹。
        # 传入门控后的成本矩阵
        # matches_a = 匹配的跟踪和检测的索引列表
        # unmatched_tracks_a = 不匹配的轨迹索引的列表
        # unmatched_detections = 未匹配的检测索引的列表，第一次出現的會在這裡
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.        
        # 将未确定态的轨迹和刚刚没有匹配上的轨迹组合为 iou_track_candidates 
        # 并进行基于IoU的匹配

        # 刚刚没有匹配上的轨迹
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        # 并非刚刚没有匹配上的轨迹，可能是有幾幀沒有匹配到，不是上一幀
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]

        # 在第二次匹配時我們只會將上一幀有偵測到的但第一次匹配失敗的track拿進去，因為第二次是用iou匹配
        # 如果上次沒有匹配到那使用iou匹配會導致匹配錯誤，就是不同物體但是匹配到一起
        # 二次嘗試匹配
        # 对级联匹配中还没有匹配成功的目标再进行IoU匹配
        # min_cost_matching 使用匈牙利算法解决线性分配问题。
        # 传入 iou_cost，尝试关联剩余的轨迹与未确认的轨迹。
        # matches_b = 有匹配到的track以及detection，兩兩會一組所以裡面會是一個tuple，shape List[tuple(track_idx, detection_idx)]
        # unmatched_tracks_b = 沒有匹配到的track，shape List[]，裡面的值會是track的index
        # unmatched_detections = 沒有匹配到的detections，shape List[]，裡面的值會是detections的index
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        # 组合两部分匹配，只有一部分有匹配到就算有追蹤到
        # matches shape = list[tuple(track_index, detection_index)]
        matches = matches_a + matches_b
        # 原先有在追蹤但是這一幀沒追蹤到的部分
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        # 將最後結果回傳回去
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        # 已看過
        # detection裡面會有三個資訊(confidence, feature, tlwh)
        # confidence = 置信度
        # feature = 該目標的特徵向量 ndarray shape [512]
        # tlwh = 座標位置 (xmin, ymin, width, height)
        # 透過.to_xyah後傳入的是(center_x, center_y, aspect ratio, height)，aspect ratio = height / width
        # mean shape = numpy [8]
        # covariance shape = numpy [8, 8]
        mean, covariance = self.kf.initiate(detection.to_xyah())
        # 添加到tracks當中
        # 從這裡可以知道track裡面存的資料型態會是Track這種格式
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        # 唯一id會加一
        self._next_id += 1
