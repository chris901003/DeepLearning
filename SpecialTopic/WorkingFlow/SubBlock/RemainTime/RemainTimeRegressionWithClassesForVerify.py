import os
import copy
import json
import torch
from functools import partial
from typing import Union
from SpecialTopic.ST.utils import get_cls_from_dict
from SpecialTopic.RemainEatingTime.RegressionModel.api import init_model as regression_model_init
from SpecialTopic.RemainEatingTime.RegressionModel.api import predict_remain_time as regression_model_predict
from SpecialTopic.YoloxObjectDetection.api import init_model as time_detect_init
from SpecialTopic.YoloxObjectDetection.api import detect_image as time_image_detect


class RemainTimeRegressionWithClassForVerify:
    def __init__(self, regression_cfg_file, input_reduce_mode: Union[str, dict] = 'Default',
                 output_reduce_mode: Union[str, dict] = 'Default', remain_project_remain_time=None,
                 time_detect_pretrained=None, time_detect_classes=11, screen_xmin=0, screen_ymin=0,
                 screen_height=200, screen_width=200):
        """ 用來驗證使用，會根據碼錶的時間預測剩餘時間，並且將碼表資訊傳回，根據不同類別會使用不同的回歸模型權重
        此模塊將不會丟棄任何長時間沒有追蹤到的對象，因為在驗證模式下默認只會有一個食物，並且需要保持追蹤
        Args:
            regression_cfg_file: 回歸模型初始化設定資料
            input_reduce_mode: 一段時間內的剩餘量輸入的緩衝方式
            output_reduce_mode: 輸出剩餘時間的緩衝方式
            remain_project_remain_time: 可以設定將哪些類別映射到哪個回歸模型，如果不設定就是依照原始值
            time_detect_pretrained: 碼表數字辨識權重路徑
            time_detect_classes: 碼表辨識類別數
            screen_xmin: 碼表截圖左上角點座標
            screen_ymin: 碼表截圖左上角點座標
            screen_height: 截圖高度
            screen_width: 截圖寬度
        """
        support_input_reduce_mode = {
            'mean': self.input_reduce_mean,
            'topk': self.input_reduce_topk,
            'range': self.input_reduce_range
        }
        support_output_reduce_mode = {
            'momentum': self.output_reduce_momentum
        }
        if input_reduce_mode == 'Default':
            input_reduce_mode = dict(type='mean')
        else:
            assert isinstance(input_reduce_mode, dict), '傳入的資料需要是字典格式'
        if output_reduce_mode == 'Default':
            output_reduce_mode = dict(type='momentum', alpha=0.2)
        else:
            assert isinstance(output_reduce_mode, dict), '傳入的資料需要是字典格式'
        input_reduce_func = get_cls_from_dict(support_input_reduce_mode, input_reduce_mode)
        self.input_reduce = partial(input_reduce_func, **input_reduce_mode)
        output_reduce_func = get_cls_from_dict(support_output_reduce_mode, output_reduce_mode)
        self.output_reduce = partial(output_reduce_func, **output_reduce_mode)
        self.regression_cfg_file = regression_cfg_file
        self.models = self.create_regression_models()
        self.remain_project_remain_time = remain_project_remain_time
        self.keep_data = dict()
        # keep_data = {
        #   'remain_buffer': list()，一段時間內獲取的剩餘量保存地方
        #   'remain_time': 上次輸出的剩餘時間，有可能會是字串(表示還在初始化當中)
        #   'stopwatch_time': 畫面上碼錶的時間，時間單位為(秒)
        #   'stopwatch_detail': 碼表的詳細內容[stopwatch_labels, stopwatch_boxes]
        #   'last_predict_time': 最後一次預估剩餘時間時碼表的秒數
        #   'record_remain': list()，紀錄每次進行預測剩餘時間的剩餘量
        # }
        self.support_api = {
            'remain_time_detection': self.remain_time_detection
        }
        self.logger = None

        # 辨識畫面上碼表時間
        # 碼表辨識的11種類別分別為[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, :]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.time_detect_classes = time_detect_classes
        self.stopwatch_detection_model = time_detect_init(pretrained=time_detect_pretrained,
                                                          num_classes=time_detect_classes)
        self.stopwatch_detection_model = self.stopwatch_detection_model.to(self.device)
        self.stopwatch_detection_model.eval()
        # 碼表在畫面上的位置
        self.screen_xmin = screen_xmin
        self.screen_ymin = screen_ymin
        self.screen_width = screen_width
        self.screen_height = screen_height

    def create_regression_models(self):
        assert os.path.exists(self.regression_cfg_file), '指定的回歸模型設定檔案不存在'
        regression_models_info = self.parse_json_file(self.regression_cfg_file)
        models_dict = dict()
        for idx, model_dict in regression_models_info.items():
            models_dict[idx] = None
            model_cfg = model_dict.get('model_cfg', None)
            pretrained_path = model_dict.get('pretrained_path', None)
            setting_path = model_dict.get('setting_path', None)
            if model_cfg is None:
                model_cfg = 'Default'
            if pretrained_path is None or (not os.path.exists(pretrained_path)):
                print(f'Remain Time model id: {idx}, 沒有訓練權重資料，將無法使用')
                continue
            if setting_path is None or (not os.path.exists(setting_path)):
                print(f'Remain Time model id: {idx}, 沒有提供setting資料，無法使用')
                continue
            model = regression_model_init(cfg=model_cfg, setting_path=setting_path, pretrained=pretrained_path)
            models_dict[idx] = model
        return models_dict

    @staticmethod
    def parse_json_file(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def __call__(self, call_api, inputs=None):
        func = self.support_api.get(call_api, None)
        assert func is not None, '尚未提供該函數'
        results = func(**inputs)
        return results

    def remain_time_detection(self, image, track_object_info):
        if len(track_object_info) == 0:
            return image, track_object_info
        assert len(track_object_info) == 1, f'在RemainTimeRegression中有多個追蹤對象，請將桌面保持只有一個食物'
        # 將單一個追蹤對象取出
        track_object = track_object_info[0]
        track_id = track_object.get('track_id', None)
        category_from_remain = track_object.get('category_from_remain', None)
        using_last = track_object.get('using_last', None)
        remain_category_id = track_object.get('remain_category_id', None)
        assert track_id is not None and category_from_remain is not None and \
               remain_category_id is not None, '傳入到RemainTime的資料有缺少'
        if isinstance(category_from_remain, str):
            self.upper_layer_init(track_id)
        elif using_last:
            raise AssertionError('在驗證模式下，每一幀都需要進行推理，請關閉加速方式')
        else:
            self.update_detection(track_id, category_from_remain, remain_category_id, image)
        # remain_time = 模型預測出來的剩餘時間
        # stopwatch_time = 畫面上碼錶的時間
        # 這裡的時間統一使用(秒)作為單位
        track_object['remain_time'] = self.keep_data[track_id]['remain_time']
        track_object['stopwatch_time'] = self.keep_data[track_id]['stopwatch_time']
        track_object['stopwatch_detail'] = self.keep_data[track_id]['stopwatch_detail']
        if isinstance(track_object['remain_time'], float):
            track_object['remain_time'] = round(track_object['remain_time'], 2)
        # 最終回傳時需要將追蹤對象轉回list型態
        return image, [track_object]

    def upper_layer_init(self, track_id):
        if track_id in self.keep_data.keys():
            assert isinstance(self.keep_data[track_id]['remain_time'], str), \
                f'Track ID: {track_id}, 程序錯誤，理論上不應該出現非字串資料'
        else:
            new_data = self.create_new_track_object()
            new_data['remain_time'] = 'Upper layer init...'
            self.keep_data[track_id] = new_data

    def update_detection(self, track_id, food_remain, remain_category_id, image):
        if track_id not in self.keep_data.keys():
            new_data = self.create_new_track_object()
            self.keep_data[track_id] = new_data
            self.keep_data[track_id]['remain_time'] = 'Waiting init...'
            self.keep_data[track_id]['stopwatch_time'] = 'Waiting init...'
        self.keep_data[track_id]['remain_buffer'].append(food_remain)
        # stopwatch_time=目標檢測出來的時間
        # stopwatch_detail=目標檢測的詳細內容[stopwatch_labels, stopwatch_boxes]
        stopwatch_time, stopwatch_detail = self.detect_stopwatch_time(image)
        self.keep_data[track_id]['stopwatch_time'] = stopwatch_time
        self.keep_data[track_id]['stopwatch_detail'] = stopwatch_detail
        self.predict_remain_time(track_id, remain_category_id)

    def detect_stopwatch_time(self, image):
        """
        傳入圖像資料，對指定地方進行目標檢測，主要是獲取畫面上碼表的數字
        最後回傳當前秒數，會將畫面上的時間轉換成秒數
        """
        rgb_image = image.get('rgb_image', None)
        assert rgb_image is not None, '無法取的彩色圖像資料'
        # 只使用有效的畫面
        rgb_image = rgb_image[self.screen_ymin:self.screen_ymin + self.screen_height,
                    self.screen_xmin:self.screen_xmin + self.screen_width, :3]
        stopwatch_result = time_image_detect(self.stopwatch_detection_model, self.device, rgb_image,
                                             input_shape=(640, 640),
                                             num_classes=self.time_detect_classes, confidence=0.8)
        stopwatch_labels, _, stopwatch_boxes = stopwatch_result
        time_info = list()
        for stopwatch_label, stopwatch_box in zip(stopwatch_labels, stopwatch_boxes):
            data = stopwatch_box.copy()
            data.append(stopwatch_label)
            time_info.append(data)
        time_info = sorted(time_info, key=lambda s: s[1])
        # 碼表辨識的10種類別分別為[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, :]
        time_cls_record = list()
        for info in time_info:
            time_cls_record.append(info[-1])
        time_record = list()
        current = 0
        for idx, info in enumerate(time_cls_record):
            if info == 10 or (idx == len(time_cls_record) - 1):
                if idx == len(time_cls_record) - 1:
                    current *= 10
                    current += info
                time_record.append(current)
                current = 0
            else:
                current *= 10
                current += info
        assert len(time_record) <= 3, '碼表的規格應該是有[時分秒]或是[分秒]或是[秒]的類型'
        stopwatch_time = 0
        for idx, info in enumerate(time_record):
            stopwatch_time *= 60
            stopwatch_time += info
        stopwatch_detail = dict(stopwatch_labels=stopwatch_labels, stopwatch_boxes=stopwatch_boxes)
        return stopwatch_time, stopwatch_detail

    def predict_remain_time(self, track_id, remain_category_id):
        stopwatch_time = self.keep_data[track_id].get('stopwatch_time', None)
        assert stopwatch_time is not None, '須提供當前碼表時間才有辦法決定是否進行預測剩餘時間'
        if self.keep_data[track_id]['last_predict_time'] == -1:
            self.keep_data[track_id]['last_predict_time'] = stopwatch_time
        if self.remain_project_remain_time is not None:
            category_id = self.remain_project_remain_time[remain_category_id]
        else:
            category_id = remain_category_id
        if self.models.get(category_id, None) is None:
            raise RuntimeError('驗證過程必須有對應的回歸模型權重')
        model = self.models[category_id]
        settings = model.settings
        time_gap = settings['time_gap']
        if stopwatch_time - self.keep_data[track_id]['last_predict_time'] < time_gap:
            return
        assert len(self.keep_data[track_id]['remain_buffer']) != 0, '一段時間內沒有獲取到任何剩餘量資訊，請檢查流程是否有錯誤'
        # 將一段時間的剩餘量進行壓縮後得得到一段時間的剩餘量
        current_remain = self.input_reduce(track_id=track_id)
        self.keep_data[track_id]['remain_buffer'] = list()
        current_index = len(self.keep_data[track_id]['record_remain'])
        # 在驗證時，如果超過最大時間長度就會直接報錯
        max_len = settings['max_length'] - 2
        if current_index == max_len:
            raise RuntimeError('驗證時請勿吃超過最大時間，無法進行預估')
        self.keep_data[track_id]['record_remain'].append(current_remain)
        # 進行預測
        predict_remain_time = regression_model_predict(model, self.keep_data[track_id]['record_remain'])
        current_remain_time = int(predict_remain_time[current_index])
        if current_remain_time >= settings['remain_time_start_value']:
            print('Remain Time Detection超出可預期範圍，請查看情況')
        keep_current_index = current_index
        while current_index >= 0 and predict_remain_time[current_index] >= settings['remain_time_start_value']:
            current_index -= 1
        current_remain_time = int(predict_remain_time[current_index])
        if current_remain_time >= settings['remain_time_start_value']:
            current_index = keep_current_index
            while current_index < len(predict_remain_time) and \
                    predict_remain_time[current_index] >= settings['remain_time_start_value']:
                current_index += 1
            current_remain_time = int(predict_remain_time[current_index])
        assert current_index < settings['remain_time_start_value'], '預測剩餘時間超出範圍，請查看資料是否錯誤'
        # 將time_gap係數與預測出的剩餘時間進行相乘，將時間尺度進行縮放
        current_remain_time *= time_gap
        last_remain_time = self.keep_data[track_id]['remain_time']
        if not isinstance(last_remain_time, str):
            current_remain_time = self.output_reduce(old_value=last_remain_time, new_value=current_remain_time)
        # 保存預測剩餘時間以及重置最後預測時間
        self.keep_data[track_id]['remain_time'] = current_remain_time
        self.keep_data[track_id]['last_predict_time'] = stopwatch_time

    def input_reduce_mean(self, track_id):
        tot = sum(self.keep_data[track_id]['remain_buffer'])
        record_len = len(self.keep_data[track_id]['remain_buffer'])
        avg = tot / record_len
        return avg

    def input_reduce_topk(self, topk, track_id):
        remain_buffer = copy.deepcopy(self.keep_data[track_id]['remain_buffer'])
        remain_buffer = sorted(remain_buffer)
        remain_len = len(remain_buffer)
        topk = min(topk, remain_len)
        remain_buffer = remain_buffer[-topk:]
        avg = sum(remain_buffer) / len(remain_buffer)
        return avg

    def input_reduce_range(self, scope, track_id):
        assert scope[0] <= scope[1] and len(scope) == 2, '給定範圍錯誤'
        remain_buffer = copy.deepcopy(self.keep_data[track_id]['remain_buffer'])
        remain_buffer = sorted(remain_buffer)
        remain_len = len(remain_buffer)
        left_idx, right_idx = int(remain_len * scope[0]), int(remain_len * scope[1])
        if left_idx == right_idx:
            right_idx += 1
        left_idx = min(max(0, left_idx), remain_len - 1)
        right_idx = min(max(1, right_idx), remain_len)
        remain_buffer = remain_buffer[left_idx:right_idx]
        avg = sum(remain_buffer) / len(remain_buffer)
        return avg

    @staticmethod
    def output_reduce_momentum(alpha, old_value, new_value):
        new_value = old_value * alpha + new_value * (1 - alpha)
        return new_value

    @staticmethod
    def create_new_track_object():
        data = dict(remain_buffer=list(), remain_time='New Remain Time track object',
                    stopwatch_time='New Remain Time track object', last_predict_time=-1, record_remain=list([100]),
                    stopwatch_detail=None)
        return data


def test():
    print('Testing Remain Time Regression With Classes For Verify')
    print('此模塊是專給驗證剩餘時間所使用')


if __name__ == '__main__':
    test()
