import time
import copy
from functools import partial
from typing import Union
from SpecialTopic.ST.utils import get_cls_from_dict


class RemainTimeCalculate:
    def __init__(self, time_gap=5, keep_frame=200, input_reduce_mode: Union[dict, str] = 'Default',
                 output_reduce_mode: Union[dict, str] = 'Default', slope_reduce_mode: Union[dict, str] = 'Default'):
        """ 通過計算下降率獲取剩餘時間
        Args:
            time_gap: 幾秒會進行一次剩餘時間計算，會將上次檢測到這次檢測所獲取的資料進行預測
            keep_frame: 一個追蹤目標丟失多少個frame後會刪除
            input_reduce_mode: 將buffer資料進行過濾，有提供幾種方式可以過濾
            output_reduce_mode: 輸出剩餘時間的處理方式，主要是避免突發的情況，使得剩餘時間大幅跳動
            slope_reduce_mode: 斜率處理方式，也是要避免突發的情況
        """
        support_input_reduce_mode = {
            'mean': self.input_reduce_mean,
            'topk': self.input_reduce_topk,
            'range': self.input_reduce_range
        }
        support_output_reduce_mode = {
            'momentum': self.output_reduce_momentum
        }
        support_slope_reduce_mode = {
            'momentum': self.output_reduce_momentum
        }
        if input_reduce_mode == 'Default':
            input_reduce_mode = {'type': 'mean'}
        else:
            assert isinstance(input_reduce_mode, dict), '傳入的設定檔需要是dict格式'
        if output_reduce_mode == 'Default':
            output_reduce_mode = {'type': 'momentum', 'alpha': 0.2}
        else:
            assert isinstance(output_reduce_mode, dict), '傳入的設定檔需要是dict格式'
        if slope_reduce_mode == 'Default':
            slope_reduce_mode = {'type': 'momentum', 'alpha': 0.9}
        else:
            assert isinstance(slope_reduce_mode, dict), '傳入的設定檔需要是dict格式'
        input_reduce_func = get_cls_from_dict(support_input_reduce_mode, input_reduce_mode)
        self.input_reduce = partial(input_reduce_func, **input_reduce_mode)
        output_reduce_func = get_cls_from_dict(support_output_reduce_mode, output_reduce_mode)
        self.output_reduce = partial(output_reduce_func, **output_reduce_mode)
        slope_reduce_func = get_cls_from_dict(support_slope_reduce_mode, slope_reduce_mode)
        self.slope_reduce = partial(slope_reduce_func, **slope_reduce_mode)
        self.time_gap = time_gap
        self.keep_frame = keep_frame
        self.keep_data = dict()
        # keep_data = {
        #   'remain_buffer': list()，一段時間內獲取的剩餘量
        #   'remain_time':
        #       1. int，上次輸出的剩餘時間
        #       2. str，還沒有上次輸出的剩餘時間
        #   'last_remain': 上次的剩餘量
        #   'last_slope': 上次的下降率
        #   'last_track_frame': 上次有追蹤到目標frame
        #   'last_predict_time': 上次預測剩餘時間的時間
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
        assert func is not None, self.logger['logger'].critical(f'Remain time calculate未提供{call_api}函數')
        results = func(**inputs)
        self.remove_miss_object()
        self.frame = (self.frame + 1) % self.frame_mod
        return results

    def remain_time_detection(self, image, track_object_info):
        for track_object in track_object_info:
            track_id = track_object.get('track_id', None)
            category_from_remain = track_object.get('category_from_remain', None)
            using_last = track_object.get('using_last', None)
            assert track_id is not None and category_from_remain is not None and using_last is not None, \
                self.logger['logger'].critical('傳送到Remain time calculate中的追蹤對象缺少資料')
            if isinstance(category_from_remain, str):
                self.upper_layer_init(track_id)
            elif using_last:
                self.get_last_detection(track_id)
            else:
                self.update_detection(track_id, category_from_remain)
            assert self.keep_data.get(track_id, None) is not None, self.logger['logger'].critical('程序有問題請排查')
            track_object['remain_time'] = self.keep_data[track_id]['remain_time']
            if isinstance(track_object['remain_time'], float):
                track_object['remain_time'] = round(track_object['remain_time'], 2)
        return image, track_object_info

    def upper_layer_init(self, track_id):
        if track_id in self.keep_data.keys():
            assert isinstance(self.keep_data[track_id]['remain_time'], str), \
                self.logger['logger'].critical(f'Track ID: {track_id}, 應該要是字串格式，程序發生錯誤')
            self.keep_data[track_id]['last_track_frame'] = self.frame
        else:
            self.logger['logger'].info(f'Track ID: {track_id}, Waiting init ...')
            new_data = self.create_new_track_object()
            new_data['remain_time'] = 'Upper layer init ...'
            self.keep_data[track_id] = new_data

    def get_last_detection(self, track_id):
        if track_id not in self.keep_data.keys():
            new_data = self.create_new_track_object()
            new_data['remain_time'] = 'Upper layer get last waiting init ...'
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
        """ 嘗試更新剩餘時間
        """
        if self.current_time - self.keep_data[track_id]['last_predict_time'] < self.time_gap:
            return
        assert len(self.keep_data[track_id]['remain_buffer']) != 0, \
            self.logger['logger'].critical(f'Track id: {track_id}, 一段時間內收到的剩餘量為空，無法檢測剩餘時間')
        current_remain = self.input_reduce(track_id=track_id)
        self.keep_data[track_id]['remain_buffer'] = list()
        last_remain = self.keep_data[track_id]['last_remain']
        last_slope = self.keep_data[track_id]['last_slope']
        last_remain_time = self.keep_data[track_id]['remain_time']
        current_slope = (last_remain - current_remain) / (self.time_gap + 1e-8)
        if last_slope is not None:
            current_slope = self.slope_reduce(old_value=last_slope, new_value=current_slope)
        if current_slope < 0:
            if current_slope < -5:
                self.logger['logger'].warning(f'Track ID: {track_id}, '
                                              f'RaminTimeCalculate突發剩餘量上升問題，會直接將斜率調整回0')
            self.logger['logger'].info(f'Track ID: {track_id}, RemainTimeCalculate下降率小於0，直接將其調整成0')
            current_slope = 0
        print(f'Track ID: {track_id}, current slope: {current_slope}')
        current_remain_time = current_remain / (current_slope + (1e-2 if current_slope == 0 else 0))
        if isinstance(last_remain_time, (int, float)):
            current_remain_time = self.output_reduce(old_value=last_remain_time, new_value=current_remain_time)
        self.keep_data[track_id]['remain_time'] = current_remain_time
        self.keep_data[track_id]['last_remain'] = current_remain
        self.keep_data[track_id]['last_slope'] = current_slope
        self.keep_data[track_id]['last_predict_time'] = self.current_time

    def remove_miss_object(self):
        """ 移除過久沒有出現的追蹤對象
        """
        remove_keys = [track_id for track_id, track_info in self.keep_data.items()
                       if ((self.frame - track_info['last_track_frame'] + self.frame_mod)
                           % self.frame_mod) > self.keep_frame]
        [self.logger['logger'].info(f'Track ID: {k} remove from RemainTimeCalculate') for k in remove_keys]
        [self.keep_data.pop(k) for k in remove_keys]

    def input_reduce_mean(self, track_id):
        """ 將一段時間內的剩餘量資料平均下來
        """
        tot = sum(self.keep_data[track_id]['remain_buffer'])
        record_len = len(self.keep_data[track_id]['remain_buffer'])
        avg = tot / record_len
        return avg

    def input_reduce_topk(self, topk, track_id):
        """ 在buffer中選出前k大的剩餘量值作為結果
        """
        remain_buffer = copy.deepcopy(self.keep_data[track_id]['remain_buffer'])
        remain_buffer = sorted(remain_buffer)
        remain_len = len(remain_buffer)
        topk = min(topk, remain_len)
        remain_buffer = remain_buffer[-topk:]
        avg = sum(remain_buffer) / len(remain_buffer)
        return avg

    def input_reduce_range(self, scope, track_id):
        """ 將buffer中選出指定範圍的資料
        """
        assert scope[0] <= scope[1] and len(scope) == 2, 'topk長度需要是2'
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
        """ 構建新的追蹤對象資料
        """
        data = dict(remain_buffer=list(), remain_time='New track object', last_remain=100, last_slope=None,
                    last_track_frame=self.frame, last_predict_time=self.current_time)
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
    input_reduce_mode_cfg = {
        'type': 'range', 'scope': (0.4, 0.7)
    }
    model = RemainTimeCalculate(time_gap=1, input_reduce_mode=input_reduce_mode_cfg)
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
    for remain in remains:
        track_object_info[0]['category_from_remain'] = remain
        image, track_object_info = model(call_api='remain_time_detection', inputs=inputs)
        remain_time = track_object_info[0]['remain_time']
        print(remain_time)
        time.sleep(0.2)


if __name__ == '__main__':
    test()
