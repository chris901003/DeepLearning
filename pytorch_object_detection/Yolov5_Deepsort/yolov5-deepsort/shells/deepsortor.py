from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort


class Deepsortor:
    # 由detector.py實例化
    def __init__(self, configFile):
        # configFile = DeepSort配置文件檔案位置
        # 已看過
        cfg = get_config()
        # 將DeepSort的配置文件內容放到cfg字典當中
        cfg.merge_from_file(configFile)
        # 構建DeepSort實例對象，將相關配置超參數傳遞過去
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

    def update(self, xywhs, confss, image):
        """
        :param xywhs: 標註匡位置 shape [num_boxes, 4] (center_x, center_y, w, h)，座標為絕對位置
        :param confss: 標註匡置信度 shape [num_boxes]
        :param image: 原始圖像 numpy shape [height, width, channel] (1080, 1920, 3)
        :return:
        """
        # 已看過
        bboxes2draw = []
        # Pass detections to deepsort
        # 將資料傳給deep sort處理
        # outputs shape [num_tracking, 5]
        # 五個東西分別為 (xmin, ymin, xmax, ymax, tracking_id)
        outputs = self.deepsort.update(xywhs, confss, image)

        for value in list(outputs):
            x1, y1, x2, y2, track_id = value
            bboxes2draw.append(
                (x1, y1, x2, y2, '', track_id)
            )

        # 將圖像以及標注相關資料回傳
        # bboxes2draw shape = list[tuple(6)] (xmin, ymin, xmax, ymax , '', track_id)
        return image, bboxes2draw