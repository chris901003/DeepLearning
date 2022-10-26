import torch
import json
import os
import numpy as np
import math
from functools import partial
from typing import Union
from SpecialTopic.SegmentationNet.api import init_module as init_segmentation_module
from SpecialTopic.SegmentationNet.api import detect_single_picture as segmentation_detect_single_picture
from SpecialTopic.ST.utils import get_cls_from_dict, get_classes


class SegformerWithDeepRemainDetection:
    def __init__(self, remain_module_file, classes_path, with_color_platte='FoodAndNotFood', save_last_period=60,
                 strict_down=False, reduce_mode: Union[dict, str] = 'Default', area_mode: Union[dict, str] = 'Default',
                 init_deep_mode: Union[dict, str] = 'Default', dynamic_init_deep_mode: Union[dict] = None,
                 check_init_ratio_frame=5, std_error=0.5, with_seg_draw=False,
                 with_depth_draw=False):
        """
        Args:
            remain_module_file: 配置剩餘量模型的config資料，目前是根據不同類別會啟用不同的分割權重模型
            classes_path: 分割網路的類別檔案
            with_color_platte: 使用的調色盤
            save_last_period: 最多可以保存多少幀沒有獲取到該id的圖像
            strict_down: 強制剩餘量只會越來越少
            reduce_mode: 對於當前檢測剩餘量的平均方式，可以減少預測結果突然暴增
            area_mode: 計算當前剩餘量的方式
            init_deep_mode: 初始化深度的方式，也就是定義原始底層深度的方式
            dynamic_init_deep_mode: 動態調整底層深度的方式，如果沒有打算動態調整就設定成None，或是初始化時不要傳入相關資料
            check_init_ratio_frame: 有新目標要檢測剩餘量時需要以前多少幀作為100%的比例
            std_error: 在確認表準比例時的最大容忍標準差
            with_seg_draw: 將分割的顏色標註圖回傳
            with_depth_draw: 將深度圖像顏色回傳
        """
        if reduce_mode == 'Default':
            reduce_mode = dict(type='momentum', alpha=0.7)
        if area_mode == 'Default':
            area_mode = dict(type='volume_change', main_classes_idx=0)
        if init_deep_mode == 'Default':
            init_deep_mode = dict(type='target_seg_idx_mean', target_seg_idx=[1, 2])
        assert isinstance(reduce_mode, dict), 'reduce mode 需要提供字典格式才有辦法初始化'
        assert isinstance(area_mode, dict) and isinstance(init_deep_mode, dict), '需要是字典格式才可以初始化'
        assert isinstance(dynamic_init_deep_mode, dict) or dynamic_init_deep_mode is None, 'dynamic_init_deep_mode需要是' \
                                                                                           '字典格式或是None'
        support_area_mode = {
            # 透過體積比例變化量得知剩餘量
            'volume_change': self.volume_change
        }
        support_reduce_mode = {
            # 透過動量方式減少波動
            'momentum': self.momentum_reduce_mode
        }
        support_init_deep_mode = {
            # 根據一開始圖像的指定地方的平均深度作為基底深度
            'target_seg_idx_mean': self.target_seg_idx_mean
        }
        support_dynamic_init_deep_mode = {
            # 在過程中可以根據當前還是目標但周遭已經是盤子的部分動態調整深度基底
            'around_food': self.around_food
        }
        area_func = get_cls_from_dict(support_area_mode, area_mode)
        area_func = partial(area_func, **area_mode)
        reduce_func = get_cls_from_dict(support_reduce_mode, reduce_mode)
        reduce_func = partial(reduce_func, **reduce_mode)
        init_deep_func = get_cls_from_dict(support_init_deep_mode, **init_deep_mode)
        init_deep_func = partial(init_deep_func, **init_deep_mode)
        if dynamic_init_deep_mode is not None:
            dynamic_init_deep_func = get_cls_from_dict(support_dynamic_init_deep_mode, **dynamic_init_deep_mode)
            dynamic_init_deep_func = partial(dynamic_init_deep_func, **dynamic_init_deep_mode)
        else:
            dynamic_init_deep_func = None
        self.area_func = area_func
        self.reduce_func = reduce_func
        self.init_deep_func = init_deep_func
        self.dynamic_init_deep_func = dynamic_init_deep_func
        self.remain_module_file = remain_module_file
        self.classes_path = classes_path
        self.save_last_period = save_last_period
        self.strict_downs = strict_down
        self.with_color_platte = with_color_platte
        self.check_init_ratio_frame = check_init_ratio_frame
        self.std_error = std_error
        self.with_seg_draw = with_seg_draw
        self.with_depth_draw = with_depth_draw
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.classes, self.num_classes = get_classes(classes_path)
        self.segformer_modules = self.build_modules()
        # keep_data = {'track_id': track_info}
        # track_id = 由目標檢測追蹤出的對象ID
        # track_info = {
        #   'remain': 原先會是上次的輸出，經過本次更新後就會是本次的輸出
        #   'last_frame': 最後一次追蹤到的幀數，超過save_last_period沒有追蹤到就會拋棄
        #   'standard_remain': 剩餘量標準，認定為100%時的食物體積
        #   'standard_remain_record': 剩餘量標準的參數保存，如果有使用動態基底深度調整就一定需要該資料
        #       這裡的每一個值表示的當初有效像素點的深度值，型態應為list(dict)
        #       dict當中會有['target_depth_info', 'basic_deep']
        #       target_depth_info = ndarray會是所有目標的深度值
        #       basic_deep = 基礎深度，透過init_deep_func獲取到的深度值
        #   'remain_seg_picture': 最後一次預測的分割色圖
        #   'remain_depth_picture': 最後一次的深度圖
        #   'basic_deep': 基底深度
        # }
        self.keep_data = dict()
        self.frame = 0
        self.mod_frame = save_last_period * 10
        self.support_api = {
            'remain_detection': self.remain_detection
        }

    def build_modules(self):
        modules_dict = dict()
        with open(self.remain_module_file, 'r') as f:
            module_config = json.load(f)
        for module_name, module_info in module_config.items():
            phi = module_info.get('phi', None)
            assert phi is not None, f'Segformer with deep remain detection {module_name}需要提供phi來指定模型大小'
            pretrained = module_info.get('pretrained', 'none')
            if not os.path.exists(pretrained):
                model = None
                print(f'Segformer with deep remain detection中{module_name}未加載預訓練權重')
            else:
                model = init_segmentation_module(model_type='Segformer', phi=phi, pretrained=pretrained,
                                                 num_classes=self.num_classes, with_color_platte=self.with_color_platte)
            modules_dict[module_name] = model
        return modules_dict

    def __call__(self, call_api, inputs=None):
        func = self.support_api.get(call_api, None)
        assert func is not None, f'Segformer with deep remain detection沒有提供{call_api}的函數'
        results = func(**inputs)
        self.remove_miss_object()
        self.frame = (self.frame + 1) % self.mod_frame
        return results

    def remain_detection(self, image, track_object_info):
        with_seg_draw = self.with_seg_draw
        with_depth_draw = self.with_depth_draw
        assert 'rgb_image' in image.keys(), '缺少rgb_image資料'
        assert 'deep_image' in image.keys(), '缺少深度deep_image資料'
        if with_depth_draw:
            assert 'deep_draw' in image.keys(), '如需畫出深度彩圖就需要提供deep_draw'
        for track_object in track_object_info:
            position = track_object.get('position', None)
            track_id = track_object.get('track_id', None)
            using_last = track_object.get('using_last', None)
            remain_category_id = track_object.get('remain_category_id', None)
            assert position is not None and track_id is not None and using_last is not None and \
                   remain_category_id is not None, '傳送到segformer with deep remain detection資料有缺少'
            results = -1
            # results = {
            #   'remain': 剩餘量的值
            #   'rgb_draw': 分割標註圖
            #   'depth_draw': 深度標註圖
            # }
            if using_last:
                results = self.get_last_detection(track_id)
            if results == -1:
                results = self.update_detection(image, position, track_id, remain_category_id,
                                                with_seg_draw=with_seg_draw, with_depth_draw=with_depth_draw)
            if isinstance(results['remain'], int):
                self.keep_data[track_id]['remain'] = results['remain']
            self.keep_data[track_id]['remain_seg_picture'] = results['rgb_draw']
            self.keep_data[track_id]['remain_depth_picture'] = results['depth_draw']
            track_object['category_from_remain'] = results['remain']
            track_object['remain_color_picture'] = results['rgb_draw']
            track_object['remain_deep_picture'] = results['depth_draw']
        return image, track_object_info

    def get_last_detection(self, track_id):
        if track_id not in self.keep_data.keys():
            return -1
        standard_remain = self.keep_data[track_id]['standard_remain']
        if standard_remain == -1:
            # 初始化食物量尚未完成
            return -1
        remain = self.keep_data[track_id]['remain']
        if remain == -1:
            return -1
        self.keep_data[track_id]['last_frame'] = self.frame
        # remain = self.get_remain_through_standard_remain(standard_remain, remain)
        rgb_draw = self.keep_data[track_id].get('remain_seg_picture', None)
        depth_draw = self.keep_data[track_id].get('remain_depth_picture', None)
        results = dict(remain=remain, rgb_draw=rgb_draw, depth_draw=depth_draw)
        return results

    def update_detection(self, image, position, track_id, remain_category_id, with_seg_draw=False,
                         with_depth_draw=False):
        # 彩色圖像
        rgb_image = image['rgb_image']
        # 深度值圖像
        depth_image = image['deep_image']
        # 根據深度值轉成顏色圖
        depth_color = image.get('deep_color', None)
        image_height, image_width = rgb_image.shape[:2]
        xmin, ymin, xmax, ymax = position
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        xmax, ymax = min(image_width, xmax), min(image_height, ymax)
        xmin, ymin = max(0, xmin), max(0, ymin)
        rgb_picture = rgb_image[xmin:xmax, ymin:ymax, :]
        if self.segformer_modules[remain_category_id] is None:
            pred = np.full((image_height, image_width), 0, dtype=np.int)
        else:
            pred = segmentation_detect_single_picture(model=self.segformer_modules[remain_category_id],
                                                      device=self.device, image_info=rgb_picture,
                                                      with_draw=with_seg_draw)
        depth_data = depth_image[xmin:xmax, ymin:ymax]
        if with_depth_draw:
            depth_color_picture = depth_color[xmin:xmax, ymin:ymax]
        else:
            depth_color_picture = None
        if with_seg_draw:
            remain = self.save_to_keep_last(track_id, pred[2], depth_data)
            results = dict(remain=remain, rgb_draw=pred[1], depth_draw=depth_color_picture)
        else:
            remain = self.save_to_keep_last(track_id, pred, depth_data)
            results = dict(remain=remain, rgb_draw=None, depth_draw=depth_color_picture)
        return results

    def save_to_keep_last(self, track_id, seg_pred, depth_data):
        assert isinstance(seg_pred, np.ndarray), '分割網路出來的資料需要是ndarray格式'
        if seg_pred.ndim == 3 and seg_pred.shape[2] != 1:
            raise ValueError('預測出來的圖樣需要是單同道圖像')
        if seg_pred.ndim == 3:
            seg_pred = seg_pred.squeeze(axis=-1)
        if track_id not in self.keep_data.keys():
            data = self.get_empty_data()
            self.keep_data[track_id] = data
        if self.keep_data[track_id]['standard_remain'] == -1:
            # 將有標註到的區域的深度值抓取出來，這裡的返回值會需要是一個維度的ndarray
            depth_info = self.area_func(seg_pred, depth_data)
            if isinstance(depth_info, list):
                depth_info = np.array(depth_info)
            assert isinstance(depth_info, np.ndarray), '需要是ndarray型態'
            assert depth_info.ndim == 1, '這裡接收的值只能是一維的ndarray'
            # 這裡回傳的會是一個感興趣區域的平均深度
            basic_deep = self.init_deep_func(seg_pred, depth_data)
            standard_remain_record_data = dict(target_depth_info=depth_info, basic_deep=basic_deep)
            self.keep_data[track_id]['standard_remain_record'].append(standard_remain_record_data)
            result = self.get_standard_remain(track_id)
        else:
            last_remain = self.keep_data[track_id]['remain']
            standard_remain = self.keep_data[track_id]['standard_remain']
            assert standard_remain != -1, 'standard_remain在-1時跳進來了，這裡發生重大錯誤'
            total_volume = self.get_total_volume(track_id, seg_pred, depth_data)
            remain = self.get_remain_through_standard_remain(standard_remain, total_volume)
            result = self.reduce_func(last_remain, remain) if last_remain != -1 else remain
            if self.dynamic_init_deep_func is not None:
                self.dynamic_init_deep_func(track_id, seg_pred, depth_data)
        return result

    def get_standard_remain(self, track_id):
        if len(self.keep_data[track_id]['standard_remain_record']) < self.check_init_ratio_frame:
            return 'Init standard remain ratio ...'
        average_basic_depth = 0
        for init_info in self.keep_data[track_id]['standard_remain_record']:
            average_basic_depth += init_info['basic_deep']
        average_basic_depth /= self.check_init_ratio_frame
        target_volume = list()
        for init_info in self.keep_data[track_id]['standard_remain_record']:
            depth_info = init_info['target_depth_info'] - average_basic_depth
            total_volume = depth_info.sum()
            target_volume.append(total_volume)
        target_volume = np.array(target_volume)
        std = target_volume.std()
        avg = target_volume.mean()
        if std > self.std_error:
            self.keep_data[track_id]['standard_remain_record'] = list()
            return f'Standard deviation is {std} lager then setting std value: {self.std_error}'
        else:
            self.keep_data[track_id]['basic_deep'] = average_basic_depth
            self.keep_data[track_id]['standard_remain'] = avg
            return f'Standard remain volume {avg}'

    def get_total_volume(self, track_id, seg_pred, depth_data):
        depth_info = self.area_func(seg_pred, depth_data)
        basic_deep = self.keep_data[track_id]['basic_deep']
        depth_info = depth_info - basic_deep
        total_volume = depth_info.sum()
        return total_volume

    def around_food(self, track_id, seg_pred, depth_data, main_classes_idx, sub_classes_idx, range_distance,
                    sample_points, momentum_alpha=0.5):
        assert isinstance(seg_pred, np.ndarray), '分割圖需要是ndarray型態'
        assert isinstance(depth_data, np.ndarray), '深度圖資料需要是ndarray型態'
        assert isinstance(main_classes_idx, int), '只能有單一一個主要目標'
        assert isinstance(range_distance, list), '檢測範圍需要是list'
        assert len(range_distance) == 2, '範圍需要剛好兩個數'
        assert range_distance[0] < range_distance[1], '範圍數值要從小到大'
        if isinstance(sub_classes_idx, int):
            sub_classes_idx = [sub_classes_idx]
        if isinstance(sample_points, int):
            sample_points = [sample_points, sample_points + 1]
        image_height, image_width = seg_pred.shape[:2]
        # 獲取主要對象(飯)的座標位置
        valid_choice_index = np.where(seg_pred == main_classes_idx)
        # 將座標位置整理成list((x, y))型態
        valid_choice_index = [(i, j) for i, j in zip(valid_choice_index[0], valid_choice_index[1])]
        valid_choice_len = len(valid_choice_index)
        valid_mask = np.full_like(seg_pred, 0, dtype=bool)
        for idx in sub_classes_idx:
            mask = seg_pred == idx
            valid_mask = np.logical_or(valid_mask, mask)
        random_sample_points = np.random.randint(sample_points[0], sample_points[1])
        deep_candidate = list()
        for _ in range(random_sample_points):
            center_index = np.random.randint(low=0, high=valid_choice_len)
            center_y, center_x = valid_choice_index[center_index]
            for _ in range(10):
                offset_x = np.random.randint(low=range_distance[0], high=range_distance[1])
                offset_y = np.random.randint(low=range_distance[0], high=range_distance[1])
                point_x, point_y = center_x + offset_x, center_y + offset_y
                if (0 < point_x < image_width) and (0 < point_y < image_height):
                    if valid_mask[point_y][point_x]:
                        deep_candidate.append(depth_data[point_y][point_x])
                        break
        if len(deep_candidate) >= (random_sample_points / 2):
            avg_deep_candidate = np.array(deep_candidate).mean()
            old_basic_deep = self.keep_data[track_id]['basic_deep']
            basic_deep = self.momentum_reduce_mode(old_basic_deep, avg_deep_candidate, alpha=momentum_alpha)
            standard_remain = self.recalculate_standard_remain(track_id, basic_deep)
            if standard_remain == -1:
                # 由於根據新的基底深度計算的體積標準差大於設定值，所以不更新基底深度
                pass
            else:
                self.keep_data[track_id]['basic_deep'] = basic_deep
                self.keep_data[track_id]['standard_remain'] = standard_remain

    def recalculate_standard_remain(self, track_id, basic_deep):
        # 根據新的基礎深度計算標準食物體積
        # 這裡會是保存list
        total_volume = list()
        for standard_info in self.keep_data[track_id]['standard_remain_record']:
            depth_info = standard_info['target_depth_info']
            depth_info = depth_info - basic_deep
            total_volume.append(depth_info.sum())
        total_volume = np.array(total_volume)
        std = total_volume.std()
        avg = total_volume.mean()
        if std > self.std_error:
            # 透過新的基底深度計算的體積差距過大時就會不更新基底深度
            return -1
        else:
            return avg

    def remove_miss_object(self):
        remove_keys = [track_id for track_id, track_info in self.keep_data.itmes()
                       if (self.frame - track_info['last_frame'] + self.mod_frame)
                       % self.mod_frame > self.save_last_period]
        [self.keep_data.pop(k) for k in remove_keys]

    @staticmethod
    def volume_change(seg_pred, depth_data, main_classes_idx):
        main_target_mask = seg_pred == main_classes_idx
        depth_data = depth_data[main_target_mask]
        return depth_data

    @staticmethod
    def target_seg_idx_mean(seg_pred, depth_data, target_seg_idx):
        if isinstance(target_seg_idx, int):
            # 將target_seg_idx統一變成list，這樣後續統一操作
            target_seg_idx = [target_seg_idx]
        # 構建一個最後我們感興趣的mask，只有為True的地方才會被算到深度平均值
        target_mask = np.full_like(seg_pred, 0, dtype=bool)
        for idx in target_seg_idx:
            # 遍歷所有我們感興趣的標籤值
            mask = seg_pred == idx
            # 將該區塊設定成True，這樣接下來就會被計算上去
            target_mask = np.logical_or(target_mask, mask)
        # 將需要的地方提取出來
        depth = depth_data[target_mask]
        # 計算平均值
        depth = depth.mean()
        return depth

    @staticmethod
    def get_remain_through_standard_remain(standard_remain, remain):
        scale = 100 / standard_remain
        result = remain * scale
        result = min(100, result)
        return result

    @staticmethod
    def momentum_reduce_mode(old_pred, new_pred, alpha=0.7):
        if math.isnan(old_pred):
            result = new_pred
        else:
            result = new_pred * (1 - alpha) + old_pred * alpha
        return result

    def get_empty_data(self):
        data = dict(remain=-1, last_frame=self.frame, standard_remain=-1, standard_remain_record=list(),
                    remain_seg_picture=None, remain_depth_picture=None, basic_deep=-1)
        return data
