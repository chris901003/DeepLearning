import torch
import numpy as np
import cv2
import os
import json
from SpecialTopic.YoloxObjectDetection.api import init_model as yolox_init
from SpecialTopic.YoloxObjectDetection.api import detect_image as yolox_detect_image
from SpecialTopic.SegmentationNet.api import init_module as segformer_init_model
from SpecialTopic.SegmentationNet.api import detect_single_picture as segformer_detect_single_picture
from SpecialTopic.ST.utils import get_classes


class YoloxWithSegmentationToTrack:
    def __init__(self, phi, yolox_pretrained, yolox_classes_path, segformer_classes_path, confidence, nms, filter_edge,
                 yolox_cfg, remain_module_file, save_last_period=60, iou_diff_threshold=5, main_classes_idx=0,
                 inference_image_size=(640, 640), average_time_check=30, track_percentage=0.5, tracking_keep_period=120,
                 last_track_send=5, average_time_output=1, intersection_area_threshold=0.2, new_track_box=60,
                 new_track_box_max_interval=3, device='auto', yolox_label_to_segformer_json='transform.json'):
        """
        Args:
            phi: 目標檢測模型大小
            yolox_pretrained: 目標檢測模型權重路徑
            yolox_classes_path: 目標檢測分類類別檔案
            segformer_classes_path: 分割網路類別檔案
            confidence: 目標檢測框置信度
            nms: 非極大值抑制處理閾值
            filter_edge: 是否要將匡到邊緣的目標刪除
            yolox_cfg: 目標檢測模型詳細設定，預設為auto
            remain_module_file: 分割模型的config資料，根據不同類別會使用不同權重的分割模型
            save_last_period: 對於一個追丟的目標會保存多少幀
            iou_diff_threshold: 當追丟時會使用分割的iou來進行輔助追蹤，兩者之間的iou差異容忍值
            main_classes_idx: 分割後那些部分是食物
            inference_image_size: 輸入到目標檢測的圖像大小
            average_time_check: 設定一段時間，這部分會與track_percentage配合，決定是否要將追蹤對象資料往下個階段
            track_percentage: 指定期間內有追蹤到多少比例的幀數就會被傳送出去，指定時間就會是average_time_check
            tracking_keep_period: 最多可以讓正在追蹤的對象追丟多少幀
            last_track_send: 正在被追蹤的目標匡距離上次偵測到超過多少幀後就不會傳送出去
            average_time_output: 多少幀會傳送一次圖像到下一層
            intersection_area_threshold: 目前偵測到的匡與上次偵測到的匡有多少比例重合就會認定為同一個目標(交集與並集比)
            new_track_box:  連續追蹤到多少幀才會認定為新目標
            new_track_box_max_interval: 新偵測到的匡最多可以有多少幀不被檢查到，如果超過interval就會被刪除
            device: 運行設備
            yolox_label_to_segformer_json: 將yolox的label轉到segformer的對應index
        """
        self.phi = phi
        self.yolox_pretrained = yolox_pretrained
        self.yolox_classes_path = yolox_classes_path
        self.segformer_classes_path = segformer_classes_path
        self.confidence = confidence
        self.nms = nms
        self.filter_edge = filter_edge
        self.yolox_cfg = yolox_cfg
        self.remain_module_file = remain_module_file
        self.save_last_period = save_last_period
        self.main_classes_idx = main_classes_idx
        self.iou_diff_threshold = iou_diff_threshold
        self.inference_image_size = inference_image_size
        self.average_time_check = average_time_check
        self.track_percentage = track_percentage
        self.tracking_keep_period = tracking_keep_period
        self.last_track_send = last_track_send
        self.average_time_output = average_time_output
        self.intersection_area_threshold = intersection_area_threshold
        self.new_track_box = new_track_box
        self.new_track_box_max_interval = new_track_box_max_interval
        if device == 'auto':
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.yolox_label_to_segformer_json = yolox_label_to_segformer_json
        assert average_time_check <= tracking_keep_period, '設定值不合理'
        self.yolox_classes_name, self.yolox_num_classes = get_classes(self.yolox_classes_path)
        self.segformer_classes_name, self.segformer_num_classes = get_classes(self.segformer_classes_path)
        self.object_detection_model = yolox_init(cfg=yolox_cfg, pretrained=yolox_pretrained,
                                                 num_classes=self.yolox_num_classes, phi=phi, device=device)
        self.segformer_modules = self.build_segformer_modules()
        self.object_detection_model = self.object_detection_model.eval()
        self.yolox_to_segformer_class = self.parse_yolox_to_segformer_json()
        self.track_index = 0
        self.mod_track_index = 10000
        self.current_frame_index = 0
        self.mod_frame_index = max(average_time_check, new_track_box, tracking_keep_period) * 10
        self.output_countdown = average_time_output - 1
        # 目前有被追蹤的標註匡，track_index會作為key，其餘作為value
        # track_box = {
        #   'position': average_time_check時間內的位置信息，ndarray[average_time_check, 4]
        #   'scores': 目標檢測類別置信度，ndarray[average_time_check]
        #   'label': 目標檢測分類類別(str)
        #   'last_miss: 上一個frame是否有追丟
        #   'last_iou': 上一個frame的iou值
        #   'frame_index': 有追蹤到的frame_index(list)，用來檢查最後追蹤到時間，以及一段時間內被追蹤到幾次
        # }
        self.track_box = dict()
        self.waiting_track_box = list()
        self.support_api = {
            'detect_single_picture': self.detect_single_picture,
            'get_num_wait_tracking_object': self.get_num_wait_tracking_object,
            'get_num_tracking_object': self.get_num_tracking_object
        }
        self.logger = None

    def build_segformer_modules(self):
        """ 主要是創建分割網路要使用的模型
        """
        segformer_modules_dict = dict()
        with open(self.remain_module_file, 'r') as f:
            module_config = json.load(f)
        for module_name, module_info in module_config.items():
            phi = module_info['phi']
            pretrained = module_info['pretrained']
            if not os.path.exists(pretrained):
                model = None
                print(f'Segformer support tracking {module_name}未加載成功，該類別將不會有輔助追蹤功能')
            else:
                model = segformer_init_model(model_type='Segformer', phi=phi, pretrained=pretrained,
                                             num_classes=self.segformer_num_classes)
            segformer_modules_dict[module_name] = model
        return segformer_modules_dict

    def parse_yolox_to_segformer_json(self):
        """ 讀取將yolox的標註類別對應到segformer的檔案
        """
        with open(self.yolox_label_to_segformer_json, 'r') as f:
            transform_dict = json.load(f)
        transform_list = transform_dict['FoodDetection9']
        transform_dict = dict()
        for yolox_label, segformer_label in zip(transform_list[0], transform_list[1]):
            transform_dict[yolox_label] = segformer_label
        return transform_dict

    def __call__(self, call_api, input):
        func = self.support_api.get(call_api, None)
        assert func is not None, self.logger['logger'].critical(f'Yolox object detection沒有提供{call_api}函數使用')
        if input is None:
            results = func()
        else:
            results = func(**input)
        return results

    def detect_single_picture(self, image, image_type, deep_image=None, deep_draw=None, force_get_detect=False):
        self.logger['logger'].debug('Detect single picture')
        self.logger['logger'].debug(f'Current frame {self.current_frame_index}')
        image = self.change_to_ndarray(image, image_type)
        self.image_height, self.image_width = image.shape[:2]
        detect_results = yolox_detect_image(model=self.object_detection_model, device=self.device, image_info=image,
                                            input_shape=self.inference_image_size, num_classes=self.yolox_num_classes,
                                            confidence=self.confidence, nms_iou=self.nms)
        labels, scores, boxes = detect_results
        self.mark_tracking_miss_to_true()
        not_track_index = list()
        for idx, (label, score, box) in enumerate(zip(labels, scores, boxes)):
            ymin, xmin, ymax, xmax = box
            if self.filter_edge:
                if xmin < 0 or ymin < 0 or xmax >= self.image_width or ymax >= self.image_height:
                    continue
            else:
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(self.image_width, xmax), min(self.image_height, ymax)
            current_position = np.array([[xmin, ymin, xmax, ymax]])
            match_track_index = self.matching_tracking_box(box, self.yolox_classes_name[label])
            clip_box = (xmin, ymin, xmax, ymax)
            food_box_iou = self.get_food_box_iou(image, clip_box, self.yolox_classes_name[label])
            self.logger['logger'].debug(f'xmin: {xmin}, ymin: {ymin}, iou: {food_box_iou}')
            if match_track_index != -1:
                self.logger['logger'].info(f'Track ID: [ {match_track_index} ] match')
                self.track_box[match_track_index]['position'] = np.append(
                    self.track_box[match_track_index]['position'], current_position, axis=0)
                self.track_box[match_track_index]['scores'] = np.append(
                    self.track_box[match_track_index]['scores'], score)
                self.track_box[match_track_index]['frame_index'] = np.append(
                    self.track_box[match_track_index]['frame_index'], self.current_frame_index)
                self.track_box[match_track_index]['last_miss'] = False
                self.track_box[match_track_index]['last_iou'] = food_box_iou
            else:
                not_track_index.append(idx)
        for idx, (label, score, box) in enumerate(zip(labels, scores, boxes)):
            if idx not in not_track_index:
                continue
            ymin, xmin, ymax, xmax = box
            if self.filter_edge:
                if xmin < 0 or ymin < 0 or xmax >= self.image_width or ymax >= self.image_height:
                    continue
            else:
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(self.image_width, xmax), min(self.image_height, ymax)
            current_position = np.array([[xmin, ymin, xmax, ymax]])
            clip_box = (xmin, ymin, xmax, ymax)
            food_box_iou = self.get_food_box_iou(image, clip_box, self.yolox_classes_name[label])
            self.logger['logger'].debug(f'xmin: {xmin}, ymin: {ymin}, iou: {food_box_iou}')
            match_waiting_track_index = self.match_waiting_track(box, self.yolox_classes_name[label])
            if match_waiting_track_index != -1:
                self.logger['logger'].debug(f'Waiting track index: [ {match_waiting_track_index} ]')
                self.waiting_track_box[match_waiting_track_index]['position'] = np.append(
                    self.waiting_track_box[match_waiting_track_index]['position'], current_position, axis=0)
                self.waiting_track_box[match_waiting_track_index]['scores'] = np.append(
                    self.waiting_track_box[match_waiting_track_index]['scores'], score)
                self.waiting_track_box[match_waiting_track_index]['frame_index'] = np.append(
                    self.waiting_track_box[match_waiting_track_index]['frame_index'], self.current_frame_index)
                self.waiting_track_box[match_waiting_track_index]['last_miss'] = False
                self.waiting_track_box[match_waiting_track_index]['last_iou'] = food_box_iou
            else:
                # 這裡會透過iou將沒有匹配上的盡量匹配到正在追蹤的對象上
                iou_match_track_index = self.iou_match_track(food_box_iou, self.yolox_classes_name[label])
                if iou_match_track_index != -1:
                    self.logger['logger'].info(f'Track ID: [ {iou_match_track_index} ] match')
                    self.track_box[iou_match_track_index]['position'] = np.append(
                        self.track_box[iou_match_track_index]['position'], current_position, axis=0)
                    self.track_box[iou_match_track_index]['scores'] = np.append(
                        self.track_box[iou_match_track_index]['scores'], score)
                    self.track_box[iou_match_track_index]['frame_index'] = np.append(
                        self.track_box[iou_match_track_index]['frame_index'], self.current_frame_index)
                    self.track_box[iou_match_track_index]['last_miss'] = False
                    self.track_box[iou_match_track_index]['last_iou'] = food_box_iou
                else:
                    self.logger['logger'].debug(
                        f'Add new waiting track target at position [ {current_position.tolist()} ]')
                    # 沒有匹配到正在等待追蹤的目標，自立一個新的等待追蹤目標
                    waiting_track_info = {
                        'position': current_position, 'scores': np.array([score]),
                        'label': self.yolox_classes_name[label],
                        'frame_index': np.array([self.current_frame_index])
                    }
                    # 添加到等待追蹤的列表當中
                    self.waiting_track_box.append(waiting_track_info)
        # 移除原先正在追蹤但是現在跟丟的目標
        self.remove_tracking_box()
        # 移除等待加入追蹤的標註框
        self.remove_wait_tracking_box()
        # 將已經追蹤到足夠次數的標註匡加入到正在追蹤標註匡當中
        self.waiting_to_tracking()
        # 將已經過時資料刪除，超過average_time_check部分刪除
        self.filter_frame()
        # 準備要輸出的結果
        results = self.prepare_output()
        # 更新目前最新的frame索引
        self.current_frame_index = (self.current_frame_index + 1) % self.mod_frame_index
        image = dict(rgb_image=image, deep_image=deep_image, deep_draw=deep_draw)
        if force_get_detect:
            return image, results, detect_results
        else:
            return image, results

    @staticmethod
    def change_to_ndarray(image, image_type):
        if image_type == 'ndarray':
            return image
        elif image_type == 'PIL' or image_type == 'Image':
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            return image
        else:
            raise ValueError(f'目前沒有實作{image_type}的轉換方式')

    def mark_tracking_miss_to_true(self):
        """ 先將當前所有正在追蹤對象設定成未追蹤到
        """
        for track_index, track_box_info in self.track_box.items():
            track_box_info['last_miss'] = True

    def get_food_box_iou(self, image, box, label):
        """
        Args:
            image: 原始圖像
            box: 當前目標位置
            label: 目標檢測出來的類別
        Returns:
            標註框與主要對象間的交並比
        """
        segformer_label = self.yolox_to_segformer_class[label]
        if self.segformer_modules[segformer_label] is None:
            return -1
        device = torch.device(self.device)
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        clip_image = image[ymin:ymax, xmin:xmax, :]
        segformer_pred = segformer_detect_single_picture(model=self.segformer_modules[segformer_label],
                                                         device=device, image_info=clip_image, with_draw=False)
        mark_pixels = (segformer_pred == self.main_classes_idx).sum()
        image_size = (xmax - xmin) * (ymax - ymin)
        return mark_pixels / (image_size + 1)

    def matching_tracking_box(self, box, label_name):
        """ 讓當前檢測到的目標與已經正在追蹤的目標進行匹配
        Args:
            box: 當前目標匡座標
            label_name: 當前目標的類別名稱
        Returns:
            return: 如果有配對上正在追蹤目標就會是track_id，沒有匹配上回傳-1
        """
        for track_index, track_box_info in self.track_box.items():
            label = track_box_info['label']
            if label != label_name:
                continue
            position = track_box_info['position']
            average_intersection = self.compute_intersection(box, position)
            if average_intersection >= self.intersection_area_threshold:
                return track_index
        return -1

    def match_waiting_track(self, box, label_name):
        """ 當前目標嘗試配對等待追蹤的目標
        Args:
            box: 當前目標的標註匡位置
            label_name: 當前目標的類別名稱
        Returns:
            return: 如果有配對上就會是等待追蹤的目標index，沒有匹配上回傳-1
        """
        for index, track_box_info in enumerate(self.waiting_track_box):
            label = track_box_info['label']
            if label != label_name:
                continue
            position = track_box_info['position']
            average_intersection = self.compute_intersection(box, position)
            if average_intersection >= self.intersection_area_threshold:
                return index
        return -1

    def iou_match_track(self, current_iou, current_label):
        """
        Args:
            current_iou: 當前目標的iou值
            current_label: 當前的類別
        Returns:
            回傳對應到當前正在追蹤的對象id，如果沒有匹配的會是-1
        """
        match_id = -1
        min_diff_iou = 101
        for track_index, track_box_info in self.track_box.items():
            last_miss = track_box_info['last_miss']
            if not last_miss:
                continue
            label = track_box_info['label']
            if current_label != label:
                continue
            iou = track_box_info['last_iou']
            if abs(iou - current_iou) < min_diff_iou:
                match_id = track_index
                min_diff_iou = abs(iou - current_iou)
        if min_diff_iou < self.iou_diff_threshold:
            return match_id
        return -1

    def compute_intersection(self, box, position):
        """ 計算交並比
        Args:
            box: 當前目標匡資料，傳入的會是list或是tuple
            position: 要嘗試匹配上的目標，傳入的會是ndarray
        Returns:
            return: 交集與並集的比例
        """
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
        """ 將規定時間內沒有再次追蹤到的正在追蹤資料刪除
        """
        # track_box，目前有被追蹤的標註匡，track_index會作為key，其餘作為value
        # 需要保存資料[position(ndarray), scores(ndarray), label(str), frame_index(list)]
        last_frame_index = {k: v['frame_index'][-1] for k, v in self.track_box.items()}
        delete_key = [k for k, v in last_frame_index.items()
                      if (self.current_frame_index - v +
                          self.mod_frame_index) % self.mod_frame_index >= self.tracking_keep_period]
        [self.logger['logger'].info(f'Tracking ID [ {k} ] delete') for k in delete_key]
        [self.track_box.pop(k) for k in delete_key]

    def remove_wait_tracking_box(self):
        """ 將等待加入追蹤的標註匡進行過濾，過久沒有追蹤到的就會取消
        """
        [self.logger['logger'].debug(f'Waiting track box index {idx} delete')
         for idx, wait_box in enumerate(self.waiting_track_box)
         if ((self.current_frame_index - wait_box['frame_index'][-1] + self.mod_frame_index) %
             self.mod_frame_index) > self.new_track_box_max_interval]
        self.waiting_track_box = [wait_box for wait_box in self.waiting_track_box
                                  if ((self.current_frame_index - wait_box['frame_index'][-1] + self.mod_frame_index) %
                                      self.mod_frame_index) <= self.new_track_box_max_interval]

    def waiting_to_tracking(self):
        """ 將已經追蹤夠久的預備等待追蹤目標轉到追蹤目標
        """
        for track_info in self.waiting_track_box:
            keep_tracking_length = track_info['frame_index'].shape[0]
            if keep_tracking_length > self.new_track_box:
                # 如果追蹤長度已經到達設定值就將等待追蹤對象放到正在追蹤對象當中
                self.logger['logger'].info(f'Success start tracking [ {str(self.track_index)} ]')
                self.track_box[str(self.track_index)] = track_info
                self.track_index = (self.track_index + 1) % self.mod_track_index
        # 將已經放到正在追蹤對象的資料重等待追中資料當中去除
        self.waiting_track_box = [wait_box for wait_box in self.waiting_track_box
                                  if wait_box['frame_index'].shape[0] <= self.new_track_box]

    def filter_frame(self):
        """ 將過時的資料刪除，我們只會取距離當前時間內的資料，超過時間的資料需要釋放掉
        """
        valid_right = self.current_frame_index
        valid_left = self.current_frame_index - self.tracking_keep_period + 1
        outer = False
        if valid_left < 0:
            valid_left = self.mod_frame_index + valid_left
            outer = True
        for track_index, track_info in self.track_box.items():
            frame_index = track_info['frame_index']
            valid_left_index = frame_index >= valid_left
            valid_right_index = frame_index <= valid_right
            if outer:
                valid_index = valid_left_index | valid_right_index
            else:
                valid_index = valid_left_index == valid_right_index
            # 將過時的幀去除
            track_info['position'] = track_info['position'][valid_index]
            track_info['scores'] = track_info['scores'][valid_index]
            track_info['frame_index'] = track_info['frame_index'][valid_index]

    def prepare_output(self):
        """ 最後要輸出的資料
        """
        # 最終要往下層傳送的資料
        results = list()
        # 在指定時間內至少要追蹤到多少次才可以進行輸出
        threshold_output = int(self.average_time_check * self.track_percentage)
        for track_index, track_info in self.track_box.items():
            tracking_length = len(track_info['frame_index'])
            if tracking_length < threshold_output or \
                    (self.current_frame_index - track_info['frame_index'][-1] + self.mod_frame_index) % \
                    self.mod_frame_index >= self.last_track_send:
                continue
            self.logger['logger'].info(f'Track ID {track_index} send to next stage')
            data = dict(position=track_info['position'][-1], category_from_object_detection=track_info['label'],
                        object_score=round(track_info['scores'].mean() * 100, 2), track_id=track_index,
                        using_last=(self.output_countdown != 0))
            results.append(data)
        self.output_countdown -= 1
        if self.output_countdown < 0:
            self.output_countdown = self.average_time_output - 1
        return results

    def get_num_wait_tracking_object(self):
        return len(self.waiting_track_box)

    def get_num_tracking_object(self):
        return len(self.track_box)


def test():
    # self, phi, yolox_pretrained, yolox_classes_path, segformer_classes_path, confidence, nms, filter_edge, yolox_cfg,
    # remain_module_file, save_last_period = 60, iou_diff_threshold = 5, main_classes_idx = 0,
    # inference_image_size = (640, 640), average_time_check = 30, track_percentage = 0.5, tracking_keep_period = 120,
    # last_track_send = 5, average_time_output = 1, intersection_area_threshold = 0.2, new_track_box = 60,
    # new_track_box_max_interval = 3, device = 'auto', yolox_label_to_segformer_json = 'transform.json'
    import logging
    import time
    yolox_pretrained = r'C:\DeepLearning\SpecialTopic\WorkingFlow\checkpoint\object_detection\yolox_l_fake_rice.pth'
    yolox_class = r'C:\DeepLearning\SpecialTopic\WorkingFlow\prepare\object_detection_classes.txt'
    segformer_module_file = r'C:\DeepLearning\SpecialTopic\WorkingFlow\prepare\remain_segformer_module_cfg.json'
    segformer_class = r'C:\DeepLearning\SpecialTopic\WorkingFlow\prepare\remain_segformer_detection_classes.txt'
    transform_json = r'C:\DeepLearning\SpecialTopic\WorkingFlow\config\ObjectClassifyToRemainClassify\object_' \
                     r'classify_to_remain_classify_cfg.json'
    model_cfg = dict(phi='l', yolox_pretrained=yolox_pretrained, yolox_classes_path=yolox_class,
                     segformer_classes_path=segformer_class, confidence=0.7, nms=0.3,
                     filter_edge=False, yolox_cfg='auto', new_track_box=10, remain_module_file=segformer_module_file,
                     yolox_label_to_segformer_json=transform_json)
    module = YoloxWithSegmentationToTrack(**model_cfg)
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger = dict(logger=logger, sub_log=None)
    module.logger = logger
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 0.5)
    while True:
        ret, image = cap.read()
        if ret:
            time.sleep(1)
            image_height, image_width = image.shape[:2]
            _, results, detect_results = module(call_api='detect_single_picture',
                                                input=dict(image=image, image_type='ndarray', force_get_detect=True))
            for result in results:
                position = result['position']
                object_detection_label = result['category_from_object_detection']
                track_id = result['track_id']
                using_last = result['using_last']
                position = position.tolist()
                xmin, ymin, xmax, ymax = position
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                score = str(result['object_score'])
                info = object_detection_label + '||' + score + '||' + str(track_id) + '||' + str(using_last)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                cv2.putText(image, info, (xmin + 30, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (89, 214, 210), 2, cv2.LINE_AA)
            labels, scores, boxes = detect_results
            for label, score, box in zip(labels, scores, boxes):
                ymin, xmin, ymax, xmax = box
                ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
                ymin, xmin = max(0, ymin), max(0, xmin)
                ymax, xmax = min(image_height, ymax), min(image_width, xmax)
                label_name = module.yolox_classes_name[label]
                score = str(round(score * 100, 2))
                info = label_name + '||' + score
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
                cv2.putText(image, info, (xmin + 30, ymin + 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
            num_wait_tracking_object = module(call_api='get_num_wait_tracking_object', input=None)
            num_tracking_object = module(call_api='get_num_tracking_object', input=None)
            info = f"wait: {num_wait_tracking_object}, tracking: {num_tracking_object}"
            cv2.putText(image, info, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (89, 214, 210), 2, cv2.LINE_AA)

            cv2.imshow('img', image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    test()
