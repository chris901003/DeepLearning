import time
import copy
from functools import partial
from typing import Union
from SpecialTopic.ST.utils import get_cls_from_dict
from SpecialTopic.RemainEatingTime.RegressionModel.api import init_model as regression_model_init
from SpecialTopic.RemainEatingTime.RegressionModel.api import predict_remain_time as regression_predict_remain_time


class RemainTimeRegression:
    def __init__(self, model_cfg: Union[str, dict] = 'Default', setting_path=None, pretrained_path=None, time_gap=30,
                 keep_frame=200, input_reduce_mode: Union[str, dict] = 'Default',
                 output_reduce_mode: Union[dict, str] = 'Default'):
        """ 透過回歸方式預測剩餘時間
        Args:
            model_cfg: 模型設定參數
            setting_path: 初始化模型所需要的參數設定資料
            pretrained_path: 訓練權重位置
            time_gap: 多少秒會進行一次檢測
            keep_frame: 一個追蹤對象在多少幀沒有被追蹤到會自動刪除
            input_reduce_mode: 一段時間內的剩餘量平均方式
            output_reduce_mode: 輸出剩餘時間的平緩方式
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
            assert isinstance(input_reduce_mode, dict), '傳入的設定檔需要是字典型態'
        if output_reduce_mode == 'Default':
            output_reduce_mode = dict(type='momentum', alpha=0.2)
        else:
            assert isinstance(output_reduce_mode, dict), '傳入的設定檔需要是字典型態'
        input_reduce_func = get_cls_from_dict(support_input_reduce_mode, input_reduce_mode)
        self.input_reduce = partial(input_reduce_func, **input_reduce_mode)
        output_reduce_func = get_cls_from_dict(support_output_reduce_mode, output_reduce_mode)
        self.output_reduce = partial(output_reduce_func, **output_reduce_mode)
        self.setting_path = setting_path
        self.model = regression_model_init(cfg=model_cfg, setting_path=setting_path, pretrained=pretrained_path)
        self.time_gap = time_gap
        self.keep_frame = keep_frame
        self.keep_data = dict()
        # keep_data = {
        #   'remain_buffer': list()，一段時間內獲取的剩餘量保存地方
        #   'remain_time': 上次輸出的剩餘時間，有可能會是字串(表示還在初始化當中)
        #   'record_remain_time': list()，紀錄每次進行預測剩餘時間的剩餘量
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
            assert track_id is not None and category_from_remain is not None and using_last is not None, '傳入的資料有缺'
            if isinstance(category_from_remain, str):
                self.upper_layer_init(track_id)
            elif using_last:
                self.get_last_detection(track_id)
            else:
                self.update_detection(track_id, category_from_remain)
            assert self.keep_data.get(track_id, None) is not None, \
                self.logger['logger'].critical('程序錯誤，不應該找不到追蹤對象的資料')
            track_object['remain_time'] = self.keep_data[track_id]['remain_time']
            if isinstance(track_object['remain_time'], float):
                track_object['remain_time'] = round(track_object['remain_time'], 2)
        return image, track_object_info

    def upper_layer_init(self, track_id):
        if track_id in self.keep_data.keys():
            assert isinstance(self.keep_data[track_id]['remain_time'], str), \
                self.logger['logger'].critical(f'Track ID: {track_id}, 程序錯誤，理論上不應該出現非字串資料')
            self.keep_data[track_id]['last_track_frame'] = self.frame
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

    def update_detection(self, track_id, food_remain):
        if track_id not in self.keep_data.keys():
            new_data = self.create_new_track_object()
            self.keep_data[track_id] = new_data
            self.keep_data[track_id]['remain_time'] = 'Waiting init ...'
            self.logger['logger'].info(f'Track ID: {track_id}, Wait init ...')
        self.keep_data[track_id]['remain_buffer'].append(food_remain)
        self.keep_data[track_id]['last_track_frame'] = self.frame
        self.predict_remain_time(track_id)

    def predict_remain_time(self, track_id):
        if self.current_time - self.keep_data[track_id]['last_predict_time'] < self.time_gap:
            return
        assert len(self.keep_data[track_id]['remain_buffer']) != 0, \
            self.logger['logger'].critical(f'Track ID: {track_id}, 一段時間內沒有新的剩餘量資料，看是否中途有出問題或是時間衝突')
        # 獲取本段時間的剩餘量平均值
        current_remain = self.input_reduce(track_id=track_id)
        # 清空本段時間的剩餘量資料
        self.keep_data[track_id]['remain_buffer'] = list()
        # current_index = 本次要存取predict_remain_time的index資料
        current_index = len(self.keep_data[track_id]['record_remain_time'])
        # 將本段的剩餘量資料保存
        self.keep_data[track_id]['record_remain_time'].append(current_remain)
        # 透過前面多段的剩餘量計算剩餘時間
        predict_remain_time = regression_predict_remain_time(self.model, self.keep_data[track_id]['record_remain_time'])
        # 獲取本次的剩餘時間
        current_remain_time = predict_remain_time[current_index]
        # 如果獲取本次的剩餘時間的值有發生越界就會進入，進行處理
        if current_remain_time >= self.model.settings['remain_time_start_value']:
            self.logger['logger'].warning(f'Track ID: {track_id}, Remain Time Detection超出可預期範圍，請查看情況')
        while current_index >= 0 and \
                predict_remain_time[current_index] >= self.model.settings['remain_time_start_value']:
            current_index -= 1
        current_remain_time = predict_remain_time[current_index]
        assert current_remain_time < self.model.settings['remain_time_start_value'], \
            self.logger['logger'].critical(f'Track ID: {track_id}, Remain Time Detection發生嚴重錯誤，'
                                           f'所有預測結果都沒有在合法範圍內')
        # 更新本次預測的剩餘時間
        last_remain_time = self.keep_data[track_id]['remain_time']
        if not isinstance(last_remain_time, str):
            current_remain_time = self.output_reduce(old_value=last_remain_time, new_value=current_remain_time)
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
                    last_predict_time=self.current_time, record_remain_time=list([100]))
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
    input_reduce_mode_cfg = dict(type='range', scope=(0.4, 0.7))
    model = RemainTimeRegression(setting_path='/Users/huanghongyan/Documents/DeepLearning/SpecialTopic/Remain'
                                              'EatingTime/RegressionModel/setting.json',
                                 pretrained_path='/Users/huanghongyan/Documents/DeepLearning/SpecialTopic/Remain'
                                                 'EatingTime/RegressionModel/regression_model.pth',
                                 time_gap=0, input_reduce_mode=input_reduce_mode_cfg)
    model.logger = logger
    remains = np.random.randint(low=0, high=100, size=100)
    remains = sorted(np.array(remains))[::-1]
    track_object_info = [{
        'position': [],
        'category_from_object_detection': 0,
        'object_score': 100,
        'track_id': 0,
        'using_last': False,
        'remain_category_id': 0,
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
        time.sleep(0.01)


if __name__ == '__main__':
    test()
