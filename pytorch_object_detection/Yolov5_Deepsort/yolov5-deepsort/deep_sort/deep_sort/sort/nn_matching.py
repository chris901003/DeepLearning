# vim: expandtab:ts=4:sw=4
import numpy as np


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    
    用于计算成对点之间的平方距离
    a ：NxM 矩阵，代表 N 个样本，每个样本 M 个数值 
    b ：LxM 矩阵，代表 L 个样本，每个样本有 M 个数值 
    返回的是 NxL 的矩阵，比如 dist[i][j] 代表 a[i] 和 b[j] 之间的平方和距离
    参考：https://blog.csdn.net/frankzd/article/details/80251042

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    用于计算成对点之间的余弦距离
    a ：NxM 矩阵，代表 N 个样本，每个样本 M 个数值 
    b ：LxM 矩阵，代表 L 个样本，每个样本有 M 个数值 
    返回的是 NxL 的矩阵，比如 c[i][j] 代表 a[i] 和 b[j] 之间的余弦距离
    参考：
    https://blog.csdn.net/u013749540/article/details/51813922
    

    """
    # a shape = List[ndarray[512]]，List長度就是對於這個追蹤對象之前記錄下的特徵向量
    # b shape = ndarray [num_detection, 512]，num_detection就是這次有偵測到的標註匡
    # data_is_normalized預設為False
    if not data_is_normalized:
        # 我們會進來這裡
        # np.linalg.norm 求向量的范式，默认是 L2 范式
        # l2范式就是全部平方後相加最後再開根號
        # 透過這個就可以標準化
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    # a shape = ndarray [num_feature, 512]
    # b shape = ndarray [num_detection, 512]
    # return shape [num_feature, num_detection]
    # 余弦距离 = 1 - 余弦相似度
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    # 我們會使用這個函數計算特徵之間的匹配cost
    # x shape = List[ndarray[512]]，List長度就是對於這個追蹤對象之前記錄下的特徵向量
    # y shape = ndarray [num_detection, 512]，num_detection就是這次有偵測到的標註匡
    # distances shape [追蹤對象有紀錄的特徵向量長度, num_detections]
    distances = _cosine_distance(x, y)
    # 對於一個detection我們找一個餘弦距離最小的當作，當前追蹤對象與detection的餘弦距離
    # return shape [num_detections]
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    对于每个目标，返回最近邻居的距离度量, 即与到目前为止已观察到的任何样本的最接近距离。

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
        匹配阈值。 距离较大的样本对被认为是无效的匹配。
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.
        如果不是None，则将每个类别的样本最多固定为该数字。 
        删除达到budget时最古老的样本。

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.
        一个从目标ID映射到到目前为止已经观察到的样本列表的字典

    """
    # 由deep_sort.py實例化
    def __init__(self, metric, matching_threshold, budget=None):
        """
        :param metric: 設定要用哪種模式計算，有歐式距離或是餘弦距離可以使用，預設為cosine
        :param matching_threshold: 閾值，預設為0.2
        :param budget: 控制一個追蹤目標我們只保留多少個特徵向量，這裡預設為100
        """
        # 已看過
        if metric == "euclidean":
            # 欧式距离
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            # 余弦距离，預設會使用這個
            self._metric = _nn_cosine_distance
        else:
            # 如果輸入不在上述兩個就會報錯
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        # 一些賦值
        self.matching_threshold = matching_threshold
        # budge用于控制 feature 的数目
        # 控制一個追蹤目標我們只保留多少個特徵向量，這裡預設為100
        self.budget = budget
        # 存放正在追蹤目標的特徵向量
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.
        用新的数据更新测量距离

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.
        传入特征列表及其对应id，partial_fit构造一个活跃目标的特征字典。

        """
        # features = 特徵向量shape [num_confirmed(大約), channel] (已經正在追蹤的對象數量, 特徵向量的深度)
        # targets = 每個特徵向量對應到的對象id shape [num_confirmed(大約)]
        # 前面會是大約是因為在追蹤狀態變成confirmed之前每個追蹤對象的特徵向量是不會每次都被刪除的
        # 也就是第一次變成confirmed狀態的追蹤對象會有3個左右的特徵向量，所以這裡只是大約
        # active_targets = 已經confirmed的追蹤對象的追蹤id shape [num_confirmed]，這個就是確定的了

        # 遍歷features以及targets
        for feature, target in zip(features, targets):
            # 对应目标下添加新的feature，更新feature集合
            # samples字典    d: feature list}
            # setdefault用法 = 如果字典中沒有target我們就會創建一個key為target且value為[]的鍵值對
            # 之後我們會在value的地方做append將特徵向量添加進去
            self.samples.setdefault(target, []).append(feature)
            # 對於一個追蹤目標我們只會存放100個特徵向量，太久以前的就會被捨棄
            if self.budget is not None:
                # 對於一個追蹤目標我們只考慮最進的100個特徵向量
                self.samples[target] = self.samples[target][-self.budget:]

        # 透過這層過濾後會將已經沒有在追蹤的對象從samples中刪除
        # 筛选激活的目标；samples是一个字典{id->feature list}
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        
        计算features和targets之间的距离，返回一个成本矩阵（代价矩阵）
        """
        # features = 將要匹配的匡選對象的特徵向量取出來，shape [detection_indices, channel]
        # targets = 取出要匹配的的追蹤對象的追蹤id，shape [track_indices]
        # 構建一個shape = ndarray [len(targets), len(features)]且全為0
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            # 透過追蹤對象之前的特徵向量與現在的標註匡的特徵向量計算cost
            # samples[targets]裡面會有這個追蹤對象之前的特徵向量，shape list[ndarray(512)]
            # list長度就會是這個追蹤對象記錄下來的特徵向量，最多只會紀錄往前的100個
            # 這裡的_metric我們會用的是_nn_cosine_distance函數
            # _metric shape [num_detections]
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
