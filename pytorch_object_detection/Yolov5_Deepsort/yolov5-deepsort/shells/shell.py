import torch

from shells.deepsortor import Deepsortor
from shells.detector import Detector
from shells import tools

# 在這裡都不會對yolo太過著墨，基本上就是大概看一下他的輸出會是什麼就可以了


class Shell(object):
    def __init__(self, deepsort_config_path, yolo_weight_path):
        """
        :param deepsort_config_path: DeepSort配置文件檔案位置
        :param yolo_weight_path: Yolo_v5訓練權重位置
        """
        # 已看過
        # 實例化DeepSort
        self.deepsortor = Deepsortor(configFile=deepsort_config_path)
        # 實例化Yolo_v5
        self.detector = Detector(yolo_weight_path, imgSize=640, threshould=0.3, stride=1)
        # 幀率計算
        self.frameCounter = 0

    def update(self, im):
        # 已看過
        # im = numpy shape [height, width, channel] (1080, 1920, 3)

        # 構建最後會回傳的東西
        retDict = {
            'frame': None,
            'list_of_ids': None,
            'obj_bboxes': []
        }

        # 計算總共有多少幀
        self.frameCounter += 1

        # yolov5
        # 透過yolo來找到一張圖片中的預測匡位置
        # bboxes = List[tuple[5]]，list的長度就會是圖片中標註匡的數量，tuple裡面會有5個資訊
        # 分別為[xmin, ymin, xmax, ymax, 類別名稱(str), 置信度(tensor)]，這裡的座標訊息是原圖上的絕對座標
        _, bboxes = self.detector.detect(im)
        # 儲存資料的空間
        bbox_xywh = []
        confs = []

        # 如果有偵測到標註匡才會進來
        if len(bboxes):
            # Adapt detections to deep sort input format
            # 遍歷所有標註匡，這裡我們不在意是哪個類別
            for x1, y1, x2, y2, _, conf in bboxes:
                # 轉換座標表達方式(xmin, ymin, xmax, ymax) -> (center_x, center_y, w, h)
                obj = [
                    int((x1 + x2) / 2), int((y1 + y2) / 2),
                    x2 - x1, y2 - y1
                ]
                # 添加到list裏面
                bbox_xywh.append(obj)
                confs.append(conf)
            # 將list轉換成tensor格式
            # xywhs shape [num_boxes, 4]
            xywhs = torch.Tensor(bbox_xywh)
            # confss shape [num_boxes]
            confss = torch.Tensor(confs)

            # 將標注訊息以及置信度以及原圖傳到DeepSort中
            # im = 原始圖像
            # obj_bboxes shape = list[tuple(6)] (xmin, ymin, xmax, ymax , '', track_id)
            # list長度就是有被追中到的對象
            # 其中每個對象會有裡面這五種參數
            im, obj_bboxes = self.deepsortor.update(xywhs, confss, im)

            # 绘制 deepsort 结果
            # 將追蹤結果與原始圖片放入就可以獲得標註後結果
            image = tools.plot_bboxes(im, obj_bboxes)

            retDict['frame'] = image
            retDict['obj_bboxes'] = obj_bboxes

        # 最後回傳回去
        return retDict
