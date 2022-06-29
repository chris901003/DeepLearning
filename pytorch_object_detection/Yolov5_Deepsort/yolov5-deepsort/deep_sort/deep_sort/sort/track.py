# vim: expandtab:ts=4:sw=4


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    单个目标track状态的枚举类型。 
    新创建的track分类为“Tentative”，直到收集到足够的证据为止。 
    然后，跟踪状态更改为“Confirmed”。 
    不再活跃的tracks被归类为“Deleted”，以将其标记为从有效集中删除。

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    具有状态空间（x，y，a，h）并关联速度的单个目标轨迹（track），
    其中（x，y）是边界框的中心，a是宽高比，h是高度。

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
        初始状态分布的均值向量
    covariance : ndarray
        Covariance matrix of the initial state distribution.
        初始状态分布的协方差矩阵
    track_id : int
        A unique track identifier.
        唯一的track标识符
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
        确认track之前的连续检测次数。 在第一个n_init帧中
        第一个未命中的情况下将跟踪状态设置为“Deleted” 
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
        跟踪状态设置为Deleted之前的最大连续未命中数；代表一个track的存活期限
         
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.
        此track所源自的检测的特征向量。 如果不是None，此feature已添加到feature缓存中。

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
        初始状态分布的均值向量
    covariance : ndarray
        Covariance matrix of the initial state distribution.
        初始状态分布的协方差矩阵
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
        测量更新总数
    age : int
        Total number of frames since first occurence.
        自第一次出现以来的总帧数
    time_since_update : int
        Total number of frames since last measurement update.
        自上次测量更新以来的总帧数
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
        feature缓存。每次测量更新时，相关feature向量添加到此列表中

    """

    # 在有新的追蹤目標出現時會需要針對該目標實例化一個Track物件，在tracker.py被實例化
    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        """
        :param mean: numpy shape [8]
        :param covariance: numpy shape [8, 8]
        :param track_id: 每一個追蹤對象都會有一個獨立的id，這個id只會越來越大除非程式終止
        :param n_init: 需連續追蹤到幾次
        :param max_age: 多少次沒有追蹤到就會被刪除
        :param feature: 該追蹤對象的特徵向量
        """
        # 已看過
        # 記錄下一些帶入的參數
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        # hits代表匹配上了多少次，匹配次数超过n_init，设置Confirmed状态
        # 設置成Confirmed狀態表示已經開始追蹤
        # hits每次调用update函数的时候+1 
        self.hits = 1
        # 和time_since_update功能重复
        self.age = 1
        # 每次调用predict函数的时候就会+1；   每次调用update函数的时候就会设置为0
        # 上一次有被匹配到是多久以前，主要用來檢查是否切換到delete狀態以及第二次匹配時要不要放入進行匹配
        self.time_since_update = 0

        # 初始化一个Track的时设置Tentative状态
        # 一開始初始化時會先製成Tentative模式表示有目標需要追蹤，但是還需要再多幾幀才能進入確定追蹤狀態
        # 這裡一開始會是1，真正開始追蹤時會是2，最後離開畫面時會是3表示結束追蹤
        self.state = TrackState.Tentative
        # 每个track对应多个features, 每次更新都会将最新的feature添加到列表中
        # 每個時間點的特徵向量都會被存進去
        self.features = []
        # 將初始的特徵向量存入進去
        if feature is not None:
            self.features.append(feature)

        # 一些賦值
        self._n_init = n_init 
        self._max_age = max_age

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        使用卡尔曼滤波器预测步骤将状态分布传播到当前时间步

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        # 已看過
        # mean shape = ndarray [8]，裡面會是四個跟座標有關係的值後面四個是跟加速度有關係的值
        # covariance shape = ndarray [8, 8]，斜方差矩陣
        # 回傳回來的shape與傳過去相同
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        # 一些紀錄的值加一
        self.age += 1
        # 在每次傳入新的一幀時都會將time_since_update加一
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        执行卡尔曼滤波器测量更新步骤并更新feature缓存

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        # kf = 卡爾曼濾波的實例化對象
        # detection = 對應上的detection的詳細內容

        # 更新卡爾曼濾波當中的內容
        # 我們會將detection中的座標位置轉換成(center_x, center_y, aspect ratio, height)
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        # 將特徵向量添加到features裏面保存
        self.features.append(detection.feature)

        self.hits += 1
        # 這一幀有匹配到了就將time_since_update變成0
        self.time_since_update = 0
        # hits代表匹配上了多少次，匹配次数超过n_init，设置Confirmed状态
        # 连续匹配上n_init帧的时候，转变为确定态
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        # 在追蹤的對象沒有匹配上的標註匡
        # 如果在处于Tentative态的情况下没有匹配上任何detection，转变为删除态。
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            # 如果time_since_update超过max_age，设置Deleted状态
            # 即失配连续达到max_age次数的时候，转变为删除态
            # 表示這個目標可能已經離開畫面，最後就不需要再進行追蹤了
            self.state = TrackState.Deleted 

    # 以下三種檢查追蹤目標狀態，由這些接口可以判斷對追蹤目標做什麼操作
    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        # 已看過
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        # 已看過
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        # 已看過
        return self.state == TrackState.Deleted
