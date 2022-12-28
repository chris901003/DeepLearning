try:
    import tensorrt
except ImportError:
    raise ImportError('無法使用TensorRT模組，需先安裝完TensorRT才可以使用該模塊')
import time
import os
import copy
import json
from functools import partial
from typing import Union
from SpecialTopic.ST.utils import get_cls_from_dict
from SpecialTopic.Deploy.RemainEatingTimeRegression.api import create_tensorrt_engine as regression_model_init
from SpecialTopic.Deploy.RemainEatingTimeRegression.api import tensorrt_engine_detect_remain_time as \
    regression_trt_predict_remain_time


class RemainTimeRegressionWithClassTensorRT:
    def __init__(self, regression_cfg_file, keep_frame=200, input_reduce_mode: Union[str, dict] = 'Default',
                 output_reduce_mode: Union[dict, str] = 'Default', remain_project_remain_time=None):
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
        # models = dict('idx': dict('trt_model': tensorRT Model, 'settings': Setting))
        self.remain_project_remain_time = remain_project_remain_time
        self.keep_frame = keep_frame
        self.keep_data = dict()
        # keep_data = {
        #   'remain_buffer': list()，一段時間內獲取的剩餘量保存地方
        #   'remain_time': 上次輸出的剩餘時間，有可能會是字串(表示還在初始化當中)
        #   'record_remain': list()，紀錄每次進行預測剩餘時間的剩餘量
        #   'last_track_frame': 最後一次有追蹤到的幀數
        #   'last_predict_time': 最後一次有進行預測剩餘時間的時間
        # }
        self.frame = 0
        self.frame_mod = keep_frame * 10
        self.current_time = time.time()
        self.support_api = {
            'remain_time_detection': self.remain_time_detection
        }
        self.logger = None

    @staticmethod
    def parse_setting(setting_path):
        with open(setting_path, 'r') as f:
            return json.load(f)

    def create_regression_models(self):
        assert os.path.exists(self.regression_cfg_file), '指定的回歸模型設定檔案不存在'
        with open(self.regression_cfg_file, 'r') as f:
            regression_model_info = json.load(f)
        models_dict = dict()
        for idx, model_dict in regression_model_info.items():
            # 先把model_dict當中的資料清空
            model_dict[idx] = None
            onnx_file_path = model_dict.get('onnx_file_path', None)
            trt_file_path = model_dict.get('trt_file_path', None)
            save_trt_file_path = model_dict.get('save_trt_file_path', None)
            setting_file_path = model_dict.get('setting_path', None)
            if not os.path.exists(onnx_file_path) and not os.path.exists(trt_file_path):
                # 如果沒有onnx且沒有trt檔案就會跳過該模型的建置
                continue
            # 如果沒有獲取到setting會報錯
            assert os.path.exists(setting_file_path), f'{idx}未獲取setting檔案，路徑錯誤或是沒有收集齊'
            settings = self.parse_setting(setting_file_path)
            try:
                trt_engine = regression_model_init(onnx_file_path=onnx_file_path, trt_engine_path=trt_file_path,
                                                   save_trt_engine_path=save_trt_file_path)
                models_dict[idx] = dict(trt_model=trt_engine, settings=settings)
            except RuntimeError:
                print(f'Create TensorRT Error at {idx}')
        return models_dict

    def __call__(self, call_api, inputs=None):
        self.current_time = time.time()
        func = self.support_api.get(call_api, None)
        assert func is not None, self.logger['logger'].critical(f'Remain time regression未提供{call_api}函數')
        results = func(**inputs)
        self.remove_miss_object()
        self.frame = (self.frame + 1) % self.frame_mod
        return results

    def remain_time_detection(self, image, track_object_info):
        for track_object in track_object_info:
            track_id = track_object.get('track_id', None)
            category_from_remain = track_object.get('category_from_remain', None)
            using_last = track_object.get('using_last', None)
            remain_category_id = track_object.get('remain_category_id', None)
            assert track_id is not None and category_from_remain is not None and using_last is not None and \
                   remain_category_id is not None, self.logger['logger'].critical('傳入到RemainTime的資料有缺少')
            if isinstance(category_from_remain, str):
                self.upper_layer_init(track_id)
            elif using_last:
                self.get_last_detection(track_id)
            else:
                self.update_detection(track_id, category_from_remain, remain_category_id)
            track_object['remain_time'] = self.keep_data[track_id]['remain_time']
            if isinstance(track_object['remain_time'], float):
                track_object['remain_time'] = round(track_object['remain_time'], 2)
        return image, track_object_info

    def upper_layer_init(self, track_id):
        if track_id in self.keep_data.keys():
            assert isinstance(self.keep_data[track_id]['remain_time'], str), \
                self.logger['logger'].critical(f'Track ID: {track_id}, 程序錯誤，理論上不應該出現非字串資料')
            self.keep_data[track_id]['last_remain_frame'] = self.frame
        else:
            self.logger['logger'].info(f'Track ID: {track_id}, Waiting init ...')
            new_data = self.create_new_track_object()
            new_data['remain_time'] = 'Upper layer init ...'
            self.keep_data[track_id] = new_data

    def get_last_detection(self, track_id):
        if track_id not in self.keep_data.keys():
            new_data = self.create_new_track_object()
            new_data['remain_time'] = 'Upper layer get last waiting init'
            self.keep_data[track_id] = new_data
        else:
            self.keep_data[track_id]['last_track_frame'] = self.frame

    def update_detection(self, track_id, food_remain, remain_category_id):
        if track_id not in self.keep_data.keys():
            new_data = self.create_new_track_object()
            self.keep_data[track_id] = new_data
            self.keep_data[track_id]['remain_time'] = 'Waiting init ...'
            self.logger['logger'].info(f'Track ID: {track_id}, Wait init ...')
        self.keep_data[track_id]['remain_buffer'].append(food_remain)
        self.keep_data[track_id]['last_track_frame'] = self.frame
        self.predict_remain_time(track_id, remain_category_id)

    def predict_remain_time(self, track_id, remain_category_id):
        # 檢查是否有開始計算蒐集資料
        if self.keep_data[track_id]['last_predict_time'] == 0:
            self.keep_data[track_id]['last_predict_time'] = self.current_time
        # 如果有設定剩餘量類別映射到剩餘時間就會進行映射轉換
        if self.remain_project_remain_time is not None:
            category_id = self.remain_project_remain_time[remain_category_id]
        else:
            category_id = remain_category_id
        # 檢查是否有該模型被載入
        if self.models.get(category_id, None) is None:
            # 如果沒有載入會在remain_time上面顯示未載入模型，無法預測
            self.keep_data[track_id]['remain_time'] = 'Remain Time model is None'
            self.logger['logger'].warning(f'Track ID: {track_id}, 對應上使用的{category_id}剩餘量模型不存在')
            self.keep_data[track_id]['remain_buffer'] = list()
            return
        # 獲取對應模型以及該模型的setting資料
        model_info = self.models[category_id]
        model, settings = model_info['trt_model'], model_info['settings']
        time_gap = settings['time_gap']
        # 檢查是否有到該模型的time-gap時間
        if self.current_time - self.keep_data[track_id]['last_predict_time'] < time_gap:
            return
        assert len(self.keep_data[track_id]['remain_buffer']) != 0, \
            self.logger['logger'].critical(f'Track ID: {track_id}, 一段時間內沒有新的剩餘量資料，看是否中途有出問題或是時間衝突')
        # 獲取當前的剩餘量
        current_remain = self.input_reduce(track_id=track_id)
        self.keep_data[track_id]['remain_buffer'] = list()
        current_index = len(self.keep_data[track_id]['record_remain'])
        max_len = settings['max_length'] - 2
        # 當吃的時間已經超過可預期上限時會直接告知，已超出正常時間範圍
        if current_index == max_len:
            self.keep_data[track_id]['remain_time'] = 'Out of normal eating time...'
            self.keep_data[track_id]['last_predict_time'] = self.current_time
            return
        self.keep_data[track_id]['record_remain'].append(current_remain)
        # 進行預測
        predict_remain_time = regression_trt_predict_remain_time(
            tensorrt_engine=model, food_remain=self.keep_data[track_id]['record_remain'], settings=settings)
        current_remain_time = int(predict_remain_time[current_index])
        if current_remain_time >= settings['remain_time_start_value']:
            self.logger['logger'].warning(f'Track ID: {track_id}, Remain Time Detection超出可預期範圍，請查看情況')
        keep_current_index = current_index
        while current_index >= 0 and \
                predict_remain_time[current_index] >= settings['remain_time_start_value']:
            current_index -= 1
        # 獲取當前剩餘時間資料
        current_remain_time = int(predict_remain_time[current_index])
        if current_remain_time >= settings['remain_time_start_value']:
            current_index = keep_current_index
            while current_index < len(predict_remain_time) and \
                    predict_remain_time[current_index] >= settings['remain_time_start_value']:
                current_index += 1
            # 獲取當前剩餘時間資料
            current_remain_time = int(predict_remain_time[current_index])
        assert current_remain_time < settings['remain_time_start_value'], \
            self.logger['logger'].critical(f'Track ID: {track_id}, Remain Time Detection發生嚴重錯誤，'
                                           f'所有預測結果都沒有在合法範圍內')
        # 將time_gap係數與預測出的剩餘時間進行相乘，將時間尺度進行縮放
        current_remain_time *= time_gap
        # 獲取上次最後的剩餘時間，避免突發值
        last_remain_time = self.keep_data[track_id]['remain_time']
        if not isinstance(last_remain_time, str):
            current_remain_time = self.output_reduce(old_value=last_remain_time, new_value=current_remain_time)
        # 保存預測剩餘時間以及重置最後預測時間
        self.keep_data[track_id]['remain_time'] = current_remain_time
        self.keep_data[track_id]['last_predict_time'] = self.current_time

    def remove_miss_object(self):
        remove_keys = [track_id for track_id, track_info in self.keep_data.items()
                       if ((self.frame - track_info['last_track_frame'] + self.frame_mod)
                           % self.frame_mod) > self.keep_frame]
        [self.logger['logger'].info(f'Track ID: {k} remove from RemainTimeCalculate') for k in remove_keys]
        [self.keep_data.pop(k) for k in remove_keys]

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

    def create_new_track_object(self):
        data = dict(remain_buffer=list(), remain_time='New Remain Time track object', last_track_frame=self.frame,
                    last_predict_time=0, record_remain=list([100]))
        return data


def test():
    import numpy as np
    import logging
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger = dict(logger=logger, sub_log=None)
    input_reduce_cfg = dict(type='range', scope=(0.4, 0.7))
    model = RemainTimeRegressionWithClassTensorRT(
        regression_cfg_file='./prepare/remain_time/regression/regression_model_trt_cfg.json',
        input_reduce_mode=input_reduce_cfg
    )
    model.logger = logger
    remains = [100, 99, 98, 97, 96, 94, 93, 92, 92, 90, 90, 90, 89, 89, 89, 88, 88, 87, 87, 83, 82, 82, 82, 80, 79, 78,
              76, 76, 76, 75, 75, 74, 74, 72, 72, 69, 68, 67, 66, 66, 65, 64, 64, 62, 62, 60, 60, 59, 56, 55, 55, 55,
              54, 54, 52, 51, 50, 47, 46, 43, 43, 42, 41, 40, 40, 39, 38, 38, 36, 35, 34, 34, 34, 33, 32, 32, 31, 30,
              29, 26, 24, 23, 23, 23, 23, 21, 16, 16, 15, 14, 13, 11, 11, 9, 9, 7, 7, 6, 5]
    remains = sorted(np.array(remains))[::-1]
    track_object_info = [{
        'position': [],
        'category_from_object_detection': 0,
        'object_score': 100,
        'track_id': 0,
        'using_last': False,
        'remain_category_id': '0',
        'category_from_remain': 'Init standard remain ratio ...'
    }]
    image = []
    inputs = {'image': image, 'track_object_info': track_object_info}
    image, track_object_info = model(call_api='remain_time_detection', inputs=inputs)
    remain_time = track_object_info[0]['remain_time']
    print(remain_time)
    print(remains)
    for remain in remains:
        track_object_info[0]['category_from_remain'] = remain
        image, track_object_info = model(call_api='remain_time_detection', inputs=inputs)
        remain_time = track_object_info[0]['remain_time']
        print(remain_time)
        time.sleep(1)


if __name__ == '__main__':
    test()
