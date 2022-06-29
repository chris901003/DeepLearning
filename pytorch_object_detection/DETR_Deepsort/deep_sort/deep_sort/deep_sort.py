import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']  # __all__ 提供了暴露接口用的”白名单“


class DeepSort(object):
    # 由deepsortor.py進行實例化
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7,
                 max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        # 已看過

        # 检测结果置信度阈值
        self.min_confidence = min_confidence
        # 非极大抑制阈值，设置为1代表不进行抑制
        self.nms_max_overlap = nms_max_overlap

        # 用于提取一个batch图片对应的特征
        # 這裡只會提取特徵，也就是會把原先訓練模型時最後分類部分去除，同時最後輸出也會有點改變
        # 傳入以訓練好的模型權重以及設備
        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        # 最大余弦距离，用于级联匹配，如果大于该阈值，则忽略
        max_cosine_distance = max_dist
        # 每个类别gallery最多的外观描述子的个数，如果超过，删除旧的
        # 也就是最多可以記錄前多少個的資料
        nn_budget = 100
        # NearestNeighborDistanceMetric 最近邻距离度量
        # 对于每个目标，返回到目前为止已观察到的任何样本的最近距离（欧式或余弦）。
        # 由距离度量方法构造一个 Tracker。
        # 第一个参数可选 'cosine' or 'euclidean'
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # 追蹤器實例化對象
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img):
        """
        :param bbox_xywh: 標註匡位置 shape [num_boxes, 4] (center_x, center_y, w, h)，座標為絕對位置
        :param confidences: 標註匡置信度 shape [num_boxes]
        :param ori_img: 原始圖像 numpy shape [height, width, channel] (1080, 1920, 3)
        :return:
        """
        # 已看過
        # 獲取圖像的高寬
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        # 从原图中抠取bbox对应图片并计算得到相应的特征
        # features = ndarray shape [num_boxes, 512]
        features = self._get_features(bbox_xywh, ori_img)
        # 獲得左上角點座標以及寬高
        # bbox_tlwh shape [num_boxes, 4] (xmin, ymin, width, height)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        # 筛选掉小于min_confidence的目标，并构造一个Detection对象构成的列表
        # detections裡面就是多個Detection對象，list的長度就是置信度大於閾值的標註匡
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences)
                      if conf > self.min_confidence]

        # run on non-maximum supression
        # 這裡的都已經有經過置信度閾值，所以會比原先少一點
        # 將標註匡內容拿出來
        # boxes shape = ndarray [num_boxes, 4] (xmin, ymin, width, height)
        boxes = np.array([d.tlwh for d in detections])
        # 拿出置信度分數，scores shape = ndarray [num_boxes]
        scores = np.array([d.confidence for d in detections])
        # 這裡有點小疑問，進行非極大值抑制時不需要依據類別嗎？
        # 進入非極大值抑制，我們將標註匡座標以及置信度分數放進去
        # indices shape [num_boxes]，裡面是index表示哪幾個index的標註匡可以留下來
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        # 將留下來的放入detections沒有留下的就丟棄
        detections = [detections[i] for i in indices]

        # update tracker
        # 将跟踪状态分布向前传播一步
        self.tracker.predict()
        # 执行测量更新和跟踪管理
        # 傳入新一次的標注內容
        # 這裡面有一堆操作，全部都在這裡完成
        self.tracker.update(detections)

        # output bbox identities
        # 最後要輸出的內容
        outputs = []
        # 遍歷正在追中的對象
        for track in self.tracker.tracks:
            # 只會回傳已經確定在追蹤的對象
            # time_since_update > 1 = 表示該目標有在被追蹤，但是這一幀沒有中蹤到，所以我們會跳過該目標
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # 將track裡面記錄的座標方式轉換(center_x, center_y, aspect ratio, height) -> (xmin, ymin, width, height)
            box = track.to_tlwh()
            # (xmin, ymin, width, height) -> (xmin, ymin, xmax, ymax)
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            # 獲取追中對象的追蹤id
            track_id = track.track_id
            # 將追蹤匡以及追蹤匡id放入outputs當中
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            # 往維度0做堆疊，之後傳出去
            # outputs shape [num_tracking, 5]，5的內容就是上面那些
            outputs = np.stack(outputs, axis=0)
        return outputs


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    # 将bbox的(center_x, center_y, width, height)转换成(xmin, ymin, width, height)
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        # 已看過
        # (center_x, center_y, width, height) -> (xmin, ymin, width, height)
        # 座標轉換
        # 根據不同的型態進行深度拷貝
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        # (xmin, ymin, width, height)
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2]/2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3]/2.
        return bbox_tlwh

    # 将bbox的[x,y,w,h] 转换成[x1,y1,x2,y2]
    # 某些数据集例如 pascal_voc 的标注方式是采用[x，y，w，h]
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    def _xywh_to_xyxy(self, bbox_xywh):
        # 已看過
        # (center_x, center_y, width, height) -> (xmin, ymin, xmax, ymax)
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width-1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height-1)
        # return (xmin, ymin, xmax, ymax)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        # 已看過
        # (xmin, ymin, width, height) -> (xmin, ymin, xmax, ymax)
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width-1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height-1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    # 获取抠图部分的特征
    def _get_features(self, bbox_xywh, ori_img):
        """
        :param bbox_xywh: 標註匡位置 shape [num_boxes, 4] (center_x, center_y, w, h)，座標為絕對位置
        :param ori_img: 原始圖像 numpy shape [height, width, channel] (1080, 1920, 3)
        :return:
        """
        # 存放標註匡的圖像，我們會從原圖上面把標註位置的圖像摳下來
        im_crops = []
        # 遍歷所有邊界匡
        for box in bbox_xywh:
            # (xmin, ymin, xmax, ymax)
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            # 找到原始圖像中對應位置的地方，將圖像摳下來
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        # 如果有標註匡就對標註匡的內容進行特徵提取
        if im_crops:
            # 对抠图部分提取特征，這裡使用特徵提取器
            # features = ndarray shape [num_boxes, 512]
            features = self.extractor(im_crops)
        else:
            # 如果沒有標註匡就返回空列表
            features = np.array([])
        # features = ndarray shape [num_boxes, 512]
        return features


