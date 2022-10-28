import torch
import numpy as np
from functools import partial
import time
from SpecialTopic.RemainEatingTime.api import init_model, detect_single_remain_time
from SpecialTopic.ST.utils import get_cls_from_dict


class RemainTimeTransformerDetection:
    def __init__(self, model_cfg, time_gap, min_remain_detection, reduce_mode_buffer, reduce_mode_output, keep_time):
        """
        Args:
            model_cfg: 設定模型相關參數
                phi: 模型大小
                setting_file_path: 模型當中會使用到的參數值
                pretrained: 模型訓練權重位置
            time_gap: 搜集多少時間的剩餘量會放到剩餘流當中
            min_remain_detection: 最少需要收集多少剩餘量檢測才會放到剩餘量留當中
                這裡可以與time_gap混用，需兩者條件滿足才會將資料往下一個地方推進
            reduce_mode_buffer: 一段時間內的剩餘量統合方式
            reduce_mode_output: 輸出的剩餘時間緩衝方式
            keep_time: 一個目標ID最多保存多久
        """
        self.model_cfg = model_cfg
        self.time_gap = time_gap
        self.min_remain_detection = min_remain_detection
        self.reduce_mode_buffer = reduce_mode_buffer
        self.reduce_mode_output = reduce_mode_output
        self.keep_time = keep_time
        self.model = init_model(**model_cfg)
        support_reduce_buffer_mode = {
            'mean': self.reduce_mean,
            'maximum': self.reduce_maximum,
            'minimum': self.reduce_minimum,
            'reduce_filter_maximum_and_minimum_mean': self.reduce_filter_maximum_and_minimum_mean,
        }
        support_reduce_output_mode = {
            'momentum': self.reduce_momentum
        }
        reduce_buffer_func = get_cls_from_dict(support_reduce_buffer_mode, reduce_mode_buffer)
        self.reduce_buffer_func = partial(reduce_buffer_func, **reduce_mode_buffer)
        reduce_output_func = get_cls_from_dict(support_reduce_output_mode, reduce_mode_output)
        self.reduce_output_func = partial(reduce_output_func, **reduce_mode_output)
        # keep_data = {
        #   'remain_buffer': list()，一段時間內從剩餘量檢測模型獲取的剩餘量值
        #   'remain_time_input': list()，從開始追蹤到當前的剩餘量計算
        #   'remain_time_output': list()，經過模型判斷剩餘時間的輸出
        #   'status': (int, str)，當前要輸出的狀態
        #       str = 'Waiting init ...'
        #       int = 剩餘時間值
        #   'last_track_time': float，最後一次獲取到該ID資料的時間
        #   'last_input_time': float，最後一次將buffer資料放到input當中的時間
        # }
        self.keep_data = dict()
        self.current_time = time.time()
        self.last_time = time.time()
        self.support_api = {
            'remain_time_detection': self.remain_time_detection
        }
        self.logger = None

    def __call__(self, call_api, inputs=None):
        self.current_time = time.time()
        func = self.support_api.get(call_api, None)
        assert func is not None, f'Transformer Nlp未提供{call_api}函數'
        results = func(**inputs)
        self.remove_miss_object()
        return results

    def remain_time_detection(self, image, track_object_info):
        """ 根據時間的推移以及剩餘量推算還需多少時間
        Args:
            image: 圖像相關資料
            track_object_info: 正在追蹤對象的資料
        """
        for track_object in track_object_info:
            track_id = track_object.get('track_id', None)
            category_from_remain = track_object.get('category_from_remain', None)
            using_last = track_object.get('using_last', None)
            assert track_id is not None and category_from_remain is not None and using_last is not None, \
                self.logger['logger'].critical('傳送到Remain time transformer detection中的追蹤對象缺少資料')
            if isinstance(category_from_remain, str):
                # 如果傳送過來的剩餘量資料是字串型態表示上層剩餘量檢測模型正在初始化
                self.upper_layer_init(track_id)
            elif using_last:
                # 如果上層是使用之前的結果
                self.get_last_detection(track_id)
            else:
                # 上層是本次判斷的結果
                self.update_detection(track_id, category_from_remain)
            # 檢查當前track_id應該已經被放到keep_data當中
            assert self.keep_data.get(track_id, None) is not None, self.logger['logger'].critical('程序有問題請排查')
            # 獲取模型輸出的結果作為預測結果
            track_object['remain_time'] = self.keep_data[track_id]['status']
        return image, track_object_info

    def upper_layer_init(self, track_id):
        """ 如果上層模型正在初始化會到這裡處理
        Args:
            track_id: 追蹤對象的ID
        """
        if track_id in self.keep_data.keys():
            # 如果該追蹤ID已經有出現在keep_data當中，那麼它的狀態應該要是str型態否則表示過程有問題
            assert isinstance(self.keep_data[track_id]['status'], str), \
                self.logger['logger'].critical('程序過程有問題')
            self.keep_data[track_id]['last_track_time'] = self.current_time
        else:
            # 如果該追蹤ID是第一次出現就會需要建立保存資料的字典
            self.logger['logger'].info(f'Track ID: {track_id}, Waiting init ...')
            new_data = dict(remain_buffer=list(), remain_time_input=list(), remain_time_output=list(), status='',
                            last_track_time=self.current_time, last_input_time=0)
            # 將狀態設定成[Upper layer init ...]
            new_data['status'] = 'Upper layer init ...'
            self.keep_data[track_id] = new_data

    def get_last_detection(self, track_id):
        """ 如果上層傳下的資料是之前的預測結果
        Args:
            track_id: 追蹤對象ID
        """
        if track_id not in self.keep_data.keys():
            new_data = dict(remain_buffer=list(), remain_time_input=list(), remain_time_output=list(), status='',
                            last_track_time=self.current_time, last_input_time=0)
            new_data['status'] = 'Upper layer get last waiting init ...'
            self.keep_data[track_id] = new_data
        else:
            self.keep_data[track_id]['last_track_time'] = self.current_time

    def update_detection(self, track_id, food_remain):
        """ 上層是本次的預測結果
        Args:
            track_id: 追蹤對象ID
            food_remain: 食物剩餘量
        """
        if track_id not in self.keep_data.keys():
            new_data = dict(remain_buffer=list(), remain_time_input=list(), remain_time_output=list(), status='',
                            last_track_time=self.current_time, last_input_time=0)
            self.keep_data[track_id] = new_data
            self.keep_data[track_id]['status'] = 'Waiting init ...'
            self.logger['logger'].info(f'Track ID: {track_id}, Wait init ...')
        # self.keep_data[track_id]['status'] = 'Wait init ...'
        # 將當前檢測出的剩餘量放到remain_buffer當中
        self.keep_data[track_id]['remain_buffer'].append(food_remain)
        self.keep_data[track_id]['last_track_time'] = self.current_time
        # 嘗試更新要放到剩餘時間檢測的輸入
        self.update_input_data(track_id)

    def update_input_data(self, track_id):
        """ 嘗試更新要放到剩餘時間檢測的輸入資料
        Args:
            track_id: 正在追蹤的ID
        """
        last_input_time = self.keep_data[track_id]['last_input_time']
        if self.current_time - last_input_time < self.time_gap:
            return
        if len(self.keep_data[track_id]['remain_buffer']) < self.min_remain_detection:
            return
        remain_buffer = self.keep_data[track_id]['remain_buffer']
        self.keep_data[track_id]['remain_buffer'] = list()
        remain_buffer_reduce = self.reduce_buffer_func(remain_buffer)
        self.keep_data[track_id]['remain_time_input'].append(remain_buffer_reduce)
        self.keep_data[track_id]['last_input_time'] = self.current_time
        self.update_output_data(track_id)

    def update_output_data(self, track_id):
        """ 預測還需多少時間
        """
        # 這裡需要保留<SOS>與<EOS>的兩位
        max_len = self.model.variables['max_len'] - 2
        food_remain = self.keep_data[track_id]['remain_time_input'][:max_len]
        predict_sequence = detect_single_remain_time(model=self.model, food_remain=food_remain)
        idx = len(food_remain)
        EOS_index = predict_sequence.index(self.model.variables['time_EOS_val'])
        predict_sequence = predict_sequence[:EOS_index]
        result = predict_sequence[idx] if idx < len(predict_sequence) else predict_sequence[-1]
        if result == self.model.variables['time_SOS_val']:
            return
        if len(self.keep_data[track_id]['remain_time_output']) > 0:
            reduce_result = self.reduce_output_func(self.keep_data[track_id]['remain_time_output'][-1], result)
        else:
            reduce_result = result
        self.keep_data[track_id]['remain_time_output'].append(reduce_result)
        self.keep_data[track_id]['status'] = reduce_result

    def remove_miss_object(self):
        """ 移除過久沒有出線的追蹤對象
        """
        remove_keys = [track_id for track_id, track_info in self.keep_data.items()
                       if (self.current_time - track_info['last_track_time']) > self.keep_time]
        [self.logger['logger'].info(f'Track ID: {k} remove') for k in remove_keys]
        [self.keep_data.pop(k) for k in remove_keys]

    @staticmethod
    def reduce_momentum(old_value, new_value, alpha):
        # 傳入的data需要是list，會自動將倒數第二個作為舊的值，將倒數第一個作為新的值
        assert isinstance(old_value, (int, float, list)) and isinstance(new_value, (int, float))
        if isinstance(old_value, list):
            pre_value = old_value[-1]
        else:
            pre_value = old_value
        results = pre_value * alpha + new_value * (1 - alpha)
        return results

    @staticmethod
    def reduce_mean(data):
        if isinstance(data, np.ndarray):
            results = data.mean()
        elif isinstance(data, list):
            results = sum(data) / len(data)
        elif isinstance(data, torch.Tensor):
            results = data.mean().item()
        else:
            raise NotImplementedError('尚未實作該類型的平均方式')
        return results

    @staticmethod
    def reduce_maximum(data):
        if isinstance(data, list):
            results = max(data)
        elif isinstance(data, np.ndarray):
            results = np.max(data)
        else:
            raise NotImplementedError('尚未實作該類型的最大值方式')
        return results

    @staticmethod
    def reduce_minimum(data):
        if isinstance(data, list):
            results = min(data)
        elif isinstance(data, np.ndarray):
            results = np.min(data)
        else:
            raise NotImplementedError('尚未實作該類型的最小值方式')
        return results

    @staticmethod
    def reduce_filter_maximum_and_minimum_mean(data):
        if isinstance(data, np.ndarray):
            assert len(data) >= 3, '長度至少需要為3'
            mx = np.max(data)
            mn = np.min(data)
            total = np.sum(data)
        elif isinstance(data, list):
            assert len(data) >= 3, '長度至少需要為3'
            mx = max(data)
            mn = min(data)
            total = sum(data)
        else:
            raise NotImplementedError('尚未實作該類型')
        total = total - mx - mn
        average = total / (len(data) - 2)
        return average


def test():
    import logging
    import time
    model_cfg = {
        'phi': 'm',
        'setting_file_path': '/Users/huanghongyan/Documents/DeepLearning/SpecialTopic/RemainEatingTime/train_an'
                             'notation.pickle',
        'pretrained': '/Users/huanghongyan/Documents/DeepLearning/SpecialTopic/RemainEatingTime/save/auto_eval.pth'
    }
    TransformerNlp_cfg = {
        'model_cfg': model_cfg,
        'time_gap': 1,
        'min_remain_detection': 1,
        'reduce_mode_buffer': {'type': 'mean'},
        'reduce_mode_output': {'type': 'momentum', 'alpha': 0.7},
        'keep_time': 60
    }
    remain_time_detection = RemainTimeTransformerDetection(**TransformerNlp_cfg)
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger = dict(logger=logger, sub_log=None)
    remain_time_detection.logger = logger
    sleep_time = 1
    food_remain_list = [100, 97, 93, 90, 90, 89, 86, 85, 85, 84, 82, 81, 79, 76, 76, 75, 75, 73, 73, 72, 71, 67, 61, 59,
                        58, 58, 55, 49, 49, 45, 44, 42, 34, 28, 24, 24, 20, 16, 14, 11, 7, 4, 3, 1, 0]
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
    image, track_object_info = remain_time_detection(call_api='remain_time_detection', inputs=inputs)
    remain_time = track_object_info[0]['remain_time']
    print(remain_time)
    for food_remain in food_remain_list:
        track_object_info[0]['category_from_remain'] = food_remain
        inputs = {'image': image, 'track_object_info': track_object_info}
        image, track_object_info = remain_time_detection(call_api='remain_time_detection', inputs=inputs)
        remain_time = track_object_info[0]['remain_time']
        print(remain_time)
        time.sleep(sleep_time)


if __name__ == '__main__':
    print('Testing Transformer Nlp module')
    test()
