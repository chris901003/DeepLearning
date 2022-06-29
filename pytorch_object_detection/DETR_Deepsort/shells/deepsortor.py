from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


class Deepsortor:
    def __init__(self, configFile):
        cfg = get_config()
        cfg.merge_from_file(configFile)
        # reid_ckpt = 權重檔案位置
        # max_dist = 計算距離時的最大距離，當距離大於閾值時就會被捨棄
        # min_confidence = 當置信度低於閾值時會將預測匡捨棄
        # nms_max_overlap = 在使用nms時，如果重疊iou大於閾值時會被當作同一物體的預測匡
        # max_iou_distance = iou距離閾值
        # max_age = 當一個追蹤對象超過max_age幀沒有被追蹤到的話就會判定丟失追蹤對象，將對象刪除
        # n_init = 需要對一個追蹤對象連續追蹤到n_init次後才算有追蹤到，之後才會在畫面中顯示
        # nn_budget = 一個鎖定的追蹤對象可以存放多久以前的特徵向量
        # use_cuda = 是否使用cuda進行追蹤
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    def update(self, xywhs, confss, image):
        bboxes2draw = []
        outputs = self.deepsort.update(xywhs, confss, image)
        for value in list(outputs):
            x1, y1, x2, y2, track_id = value
            bboxes2draw.append((x1, y1, x2, y2, '', track_id))
        return image, bboxes2draw
