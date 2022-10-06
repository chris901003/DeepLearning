import torch
import numpy as np
import cv2
from SpecialTopic.YoloxObjectDetection.api import init_model, detect_image
from SpecialTopic.ST.utils import get_classes


class YoloxObjectDetection:
    def __init__(self, phi, pretrained, classes_path, confidence, nms, filter_edge, cfg,
                 average_time_check=30, track_percentage=0.5, average_time_output=1, intersection_area_threshold=0.7,
                 new_track_box=60, device='auto'):
        """
        Args:
            average_time_check: 會將最近多少幀的結果進行平均，如果有被追蹤的對象在期間都沒有新的圖像就會捨棄
                如果期間內追蹤到的圖像超過指定數量就會放到下一步進行處理
            track_percentage: 指定期間內有追蹤到多少比例的幀數就會被傳送出去
            average_time_output: 多少幀會傳送一次圖像到下一層
            intersection_area_threshold: 目前偵測到的匡與上次偵測到的匡有多少比例重合就會認定為同一個目標(交集與並集比)
            new_track_box: 連續追蹤到多少幀才會認定為新目標
        """
        self.phi = phi
        self.pretrained = pretrained
        self.classes_path = classes_path
        self.confidence = confidence
        self.nms = nms
        self.filter_edge = filter_edge
        self.cfg = cfg
        self.average_time_check = average_time_check
        self.track_percentage = track_percentage
        self.average_time_output = average_time_output
        self.intersection_area_threshold = intersection_area_threshold
        self.new_track_box = new_track_box
        self.device = device
        if self.device == 'auto':
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 獲取類別訊息
        self.num_classes, self.classes_name = get_classes(self.classes_path)
        # 初始化模型，同時會將訓練權重加入
        self.object_detection_model = init_model(cfg=cfg, pretrained=pretrained, num_classes=self.num_classes, phi=phi,
                                                 device=self.device)
        # 將模型設定成驗證模式
        self.object_detection_model = self.object_detection_model.eval()
        # 追蹤匡的index，從程式啟動後只會越來越大直到超過10000
        self.track_index = 0
        # 當前是第幾幀，該數字超過一定值後會歸0，反正也不會衝突到，因為超過average_time_check後前面圖像會刪除
        self.current_frame_index = 0
        self.mod_frame_index = max(average_time_check, new_track_box) * 10
        # 目前有被追蹤的標註匡，track_index會作為key，其餘作為value
        # 需要保存資料[position(ndarray), scores(ndarray), label(str), frame_index(list)]
        self.track_box = dict()
        # 需要保存資料[position(ndarray), scores(ndarray), label(str), frame_index(ndarray)]
        self.waiting_track_box = list()
        self.support_api = {
            'detect_single_picture': self.detect_single_picture
        }

    def __call__(self, call_api, input):
        func = self.support_api.get(call_api, None)
        assert func is not None, f'Yolox object detection沒有提供{call_api}函數使用'
        results = func(**input)
        return results

    def detect_single_picture(self, image, image_type):
        image = self.change_to_ndarray(image, image_type)
        self.image_height, self.image_width = image.shape[:2]
        results = dict()
        detect_results = detect_image(model=self.object_detection_model, device=self.device, image_info=image,
                                      input_shape=[640, 640], num_classes=self.num_classes, confidence=self.confidence,
                                      nms_iou=self.nms)
        labels, scores, boxes = detect_results
        for label, score, box in zip(labels, scores, boxes):
            ymin, xmin, xmax, ymax = box
            current_position = np.array([[xmin, ymin, xmax, ymax]])
            if self.filter_edge:
                if ymin < 0 or xmin < 0 or ymax >= self.image_height or xmax >= self.image_width:
                    continue
            match_track_index = self.matching_tracking_box(box, self.classes_name[label])
            if match_track_index != -1:
                self.track_box[match_track_index]['position'] = np.append(
                    self.track_box[match_track_index]['position'], current_position, axis=0)
                self.track_box[match_track_index]['scores'] = np.array(
                    self.track_box[match_track_index]['scores'], score)
                self.track_box[match_track_index]['frame_index'] = np.array(
                    self.track_box[match_track_index]['frame_index'], self.current_frame_index)
            else:
                match_waiting_track_index = self.match_waiting_track(box, self.classes_name[label])
                if match_waiting_track_index != -1:
                    self.waiting_track_box[match_waiting_track_index]['position'] = np.append(
                        self.waiting_track_box[match_waiting_track_index]['position'], current_position, axis=0)
                    self.waiting_track_box[match_waiting_track_index]['scores'] = np.array(
                        self.waiting_track_box[match_waiting_track_index]['scores'], score)
                    self.waiting_track_box[match_waiting_track_index]['frame_index'] = np.array(
                        self.waiting_track_box[match_waiting_track_index]['frame_index'], self.current_frame_index)
                else:
                    waiting_track_info = {
                        'position': current_position, 'scores': np.array(score), 'label': self.classes_name[label],
                        'frame_index': np.array(self.current_frame_index)
                    }
                    self.waiting_track_box.append(waiting_track_info)

        # 請處理current_frame_index繞一圈回去會發生相減問題
        
        # 移除原先正在追蹤但是現在跟丟的目標
        self.remove_tracking_box()
        # 移除等待加入追蹤的標註框
        self.remove_wait_tracking_box()
        # 將已經追蹤到足夠次數的標註匡加入到正在追蹤標註匡當中
        self.waiting_to_tracking()
        return results

    @staticmethod
    def change_to_ndarray(image, image_type):
        if image_type == 'ndarray':
            return image
        elif image_type == 'PIL' or image_type == 'Image':
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            return image
        else:
            raise ValueError(f'目前沒有實作{image_type}的轉換方式')

    def matching_tracking_box(self, box, label_name):
        for track_index, track_box_info in self.track_box.items():
            # 只有類別相同的才有需要計算
            label = track_box_info['label']
            if label != label_name:
                continue
            position = track_box_info['position']
            average_intersection = self.compute_intersection(box, position)
            if average_intersection >= self.intersection_area_threshold:
                return track_index
        return -1

    def match_waiting_track(self, box, label_name):
        for index, track_box_info in enumerate(self.waiting_track_box):
            label = track_box_info['label']
            if label != label_name:
                continue
            position = track_box_info['position']
            average_intersection = self.compute_intersection(box, position)
            if average_intersection >= self.intersection_area_threshold:
                return index
        return -1

    def compute_intersection(self, box, position):
        # 計算交並比，比對是否為同一個目標對象
        ymin, xmin, ymax, xmax = box
        current_box = np.array([xmin, ymin, xmax, ymax])
        current_box = current_box[None, :]
        top_left_min = np.maximum(current_box[:, :2], position[:, :2])
        bottom_right_min = np.minimum(current_box[:, 2:], position[:, 2:])
        area_points = np.concatenate((top_left_min, bottom_right_min), axis=1)
        union_height = area_points[:, 2] - area_points[:, 0]
        union_height = np.clip(union_height, 0, self.image_height)
        union_width = area_points[:, 3] - area_points[:, 1]
        union_width = np.clip(union_width, 0, self.image_width)
        union_area = union_width * union_height

        top_left_max = np.minimum(current_box[:, :2], position[:, :2])
        bottom_right_max = np.maximum(current_box[:, 2:], position[:, 2:])
        area_points = np.concatenate((top_left_max, bottom_right_max), axis=1)
        intersect_height = area_points[:, 2] - area_points[:, 0]
        intersect_height = np.clip(intersect_height, 0, self.image_height)
        intersect_width = area_points[:, 3] - area_points[:, 1]
        intersect_width = np.clip(intersect_width, 0, self.image_width)
        intersect_area = intersect_width * intersect_height

        iou = union_area / intersect_area
        average_iou = iou.mean()
        return average_iou

    def remove_tracking_box(self):
        # track_box，目前有被追蹤的標註匡，track_index會作為key，其餘作為value
        # 需要保存資料[position(ndarray), scores(ndarray), label(str), frame_index(list)]
        last_frame_index = {k: v['frame_index'][-1] for k, v in self.track_box.items()}
        delete_key = [k for k, v in last_frame_index.items() if self.current_frame_index - v > self.average_time_check]
        [self.track_box.pop(k) for k in delete_key]

    def remove_wait_tracking_box(self):
        # 將等待加入追蹤的標註匡進行過濾，過久沒有追蹤到的就會取消
        self.waiting_track_box = [wait_box for wait_box in self.waiting_track_box
                                  if self.current_frame_index - wait_box['frame_index'][-1] > 1]

    def waiting_to_tracking(self):
        # 將已經追蹤夠久的預備等待追蹤目標轉到追蹤目標
        for track_info in self.waiting_track_box:
            keep_tracking_length = track_info['frame_index'].shape[0]
            if keep_tracking_length > self.new_track_box:
                self.track_box[str(self.track_index)] = track_info
                self.track_index += (self.track_index + 1) % 10000
        self.waiting_track_box = [wait_box for wait_box in self.waiting_track_box
                                  if wait_box['frame_index'].shape[0] <= self.new_track_box]

    def __repr__(self):
        print('Using Yolox model for object detection')
        print('==================================================')
        print(f'Model size: {self.phi}')
        print(f'Pretrain weight: {self.pretrained}')
        print(f'Classes file path: {self.classes_path}')
        print(f'Filter confidence: {self.confidence}')
        print(f'Filter outer box: {self.filter_edge}')
        print(f'Model config: {self.cfg}')
        print('==================================================')
