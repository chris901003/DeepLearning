import json
import os
import torch
import numpy as np
from typing import Union
import math
from functools import partial
from SpecialTopic.ST.utils import get_classes, get_cls_from_dict
from SpecialTopic.SegmentationNet.api import init_module, detect_single_picture


class SegformerRemainDetection:
    def __init__(self, remain_module_file, classes_path, save_last_period=60, strict_down=False,
                 reduce_mode: Union[str, dict] = 'Default', area_mode: Union[str, dict] = 'Default',
                 with_color_platte='ADE20KDataset', check_init_ratio_frame=10, std_error=0.5, with_draw=False):
        """
        Args:
            remain_module_file: 配置剩餘量模型的config資料，目前是根據不同類別會啟用不同的分割權重模型
            classes_path: 分割網路的類別檔案
            save_last_period: 最多可以保存多少幀沒有獲取到該id的圖像
            strict_down: 強制剩餘量只會越來越少
            reduce_mode: 對於當前檢測剩餘量的平均方式，可以減少預測結果突然暴增
            area_mode: 計算當前剩餘量的方式
            with_color_platte: 使用的調色盤
            check_init_ratio_frame: 有新目標要檢測剩餘量時需要以前多少幀作為100%的比例
            std_error: 在確認表準比例時的最大容忍標準差
            with_draw: 將分割的顏色標註圖回傳
        """
        if reduce_mode == 'Default':
            reduce_mode = dict(type='momentum', alpha=0.9)
        if area_mode == 'Default':
            area_mode = dict(type='pixel_mode', main_classes_idx=0, sub_classes_idx=1)
        assert isinstance(reduce_mode, dict), 'reduce mode需要是dict型態'
        support_area_mode = {
            'area_mode': self.remain_area_mode,
            'pixel_mode': self.remain_pixel_mode,
            'bbox_mode': self.remain_bbox_mode
        }
        support_reduce_mode = {
            'momentum': self.momentum_reduce_mode
        }
        area_func = get_cls_from_dict(support_area_mode, area_mode)
        area_func = partial(area_func, **area_mode)
        reduce_func = get_cls_from_dict(support_reduce_mode, reduce_mode)
        reduce_func = partial(reduce_func, **reduce_mode)
        self.area_func = area_func
        self.reduce_func = reduce_func
        self.remain_module_file = remain_module_file
        self.classes_path = classes_path
        self.save_last_period = save_last_period
        self.strict_down = strict_down
        self.reduce_mode = reduce_mode
        self.area_mode = area_mode
        self.with_color_platte = with_color_platte
        self.check_init_ratio_frame = check_init_ratio_frame
        self.std_error = std_error
        self.with_draw = with_draw
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.classes, self.num_classes = get_classes(classes_path)
        self.segformer_modules = self.build_modules()
        # keep_last = {'track_id': track_info}
        # track_info = {
        #   'remain': 最後一次檢測出來的剩餘量
        #   'last_frame': 最後一次檢測時的幀數
        #   'standard_remain': 剩餘量標準，認定為100%時的食物佔比
        #   'standard_remain_record': 保存認定標準的空間
        #   'remain_color_picture': 最後一次預測的色圖
        # }
        self.keep_last = dict()
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
            phi = module_info['phi']
            pretrained = module_info['pretrained']
            if not os.path.exists(pretrained):
                model = None
                print(f'Segformer remain detection中{module_name}未加載預訓練權重')
            else:
                model = init_module(model_type='Segformer', phi=phi, pretrained=pretrained,
                                    num_classes=self.num_classes, with_color_platte=self.with_color_platte)
            modules_dict[module_name] = model
        return modules_dict

    def __call__(self, call_api, inputs):
        func = self.support_api.get(call_api, None)
        assert func is not None, f'Segformer reman detection未提供{call_api}函數'
        results = func(**inputs)
        self.remove_miss_object()
        self.frame = (self.frame + 1) % self.mod_frame
        return results

    def remain_detection(self, image, track_object_info):
        with_draw = self.with_draw
        for track_object in track_object_info:
            position = track_object.get('position', None)
            track_id = track_object.get('track_id', None)
            using_last = track_object.get('using_last', None)
            remain_category_id = track_object.get('remain_category_id', None)
            assert position is not None and track_id is not None and using_last is not None and \
                   remain_category_id is not None, '傳送到segformer remain detection資料有缺少'
            results = -1
            if using_last:
                results = self.get_last_detection(track_id, with_draw=with_draw)
            if results == -1:
                results = self.update_detection(image, position, track_id, remain_category_id, with_draw=with_draw)
            if isinstance(results, (list, tuple)):
                results = list(results)
                if isinstance(results[0], (int, float)):
                    results[0] = round(results[0] * 100, 2)
            else:
                if isinstance(results, (int, float)):
                    results = round(results * 100, 2)
            if with_draw:
                track_object['category_from_remain'] = results[0]
                track_object['remain_color_picture'] = results[1]
            else:
                track_object['category_from_remain'] = results
        return image, track_object_info

    def get_last_detection(self, track_id, with_draw=False):
        if track_id not in self.keep_last.keys():
            return -1
        standard_remain = self.keep_last[track_id]['standard_remain']
        if standard_remain == -1:
            # 尚未擁有標準剩餘量，需要直接拿去檢測當前剩餘量
            return -1
        remain = self.keep_last[track_id]['remain']
        if remain == -1:
            return -1
        self.keep_last[track_id]['last_frame'] = self.frame
        remain = self.get_remain_through_standard_remain(standard_remain, remain)
        if with_draw:
            return remain, self.keep_last[track_id]['remain_color_picture']
        else:
            return remain

    def update_detection(self, image, position, track_id, remain_category_id, with_draw=False):
        image_height, image_width = image['rgb_image'].shape[:2]
        xmin, ymin, xmax, ymax = position
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(image_width, xmax), min(image_height, ymax)
        picture = image['rgb_image'][ymin:ymax, xmin:xmax, :]
        if self.segformer_modules[remain_category_id] is None:
            pred = np.full((image_height, image_width), 0, dtype=np.int)
        else:
            pred = detect_single_picture(model=self.segformer_modules[remain_category_id], device=self.device,
                                         image_info=picture, with_draw=with_draw)
        if with_draw:
            result = self.save_to_keep_last(track_id, pred[2])
            self.keep_last[track_id]['remain_color_picture'] = pred[1]
            return result, pred[1]
        else:
            result = self.save_to_keep_last(track_id, pred)
            return result

    def save_to_keep_last(self, track_id, pred):
        assert isinstance(pred, np.ndarray), 'pred資料需要是ndarray類型'
        if pred.ndim == 3 and pred.shape[2]:
            raise ValueError('預測出來的圖像需要是單通道')
        if pred.ndim == 3:
            pred = pred.squeeze(axis=-1)
        result = self.area_func(pred)
        print(f'Track id: {track_id}, pred: {result}')
        if track_id not in self.keep_last.keys():
            data = dict(remain=-1, last_frame=self.frame, standard_remain=-1, standard_remain_record=list())
            self.keep_last[track_id] = data
        if self.keep_last[track_id]['standard_remain'] == -1:
            self.keep_last[track_id]['standard_remain_record'].append(result)
            # 查看是否可以獲取標準剩餘量判斷比例，result會表示當前狀態
            result = self.get_standard_remain(track_id)
        else:
            last_remain = self.keep_last[track_id]['remain']
            result = self.reduce_func(result, last_remain) if last_remain != -1 else result
            if self.strict_down:
                result = min(result, self.keep_last[track_id]['remain'])
            self.keep_last[track_id]['remain'] = result
            self.keep_last[track_id]['last_frame'] = self.frame
            standard_remain = self.keep_last[track_id]['standard_remain']
            result = self.get_remain_through_standard_remain(standard_remain, result)
        return result

    def remove_miss_object(self):
        remove_keys = [track_id for track_id, track_info in self.keep_last.items()
                       if (self.frame - track_info['last_frame'] + self.mod_frame)
                       % self.mod_frame > self.save_last_period]
        [self.keep_last.pop(k) for k in remove_keys]

    def get_standard_remain(self, track_id):
        if len(self.keep_last[track_id]['standard_remain_record']) < self.check_init_ratio_frame:
            return 'Init standard remain ratio ...'
        standard_remain_record = np.array(self.keep_last[track_id]['standard_remain_record'])
        std = standard_remain_record.std()
        avg = standard_remain_record.mean()
        if std > self.std_error:
            self.keep_last[track_id]['standard_remain_record'] = list()
            return f'Standard deviation is {std} lager then setting need recollect'
        else:
            self.keep_last[track_id]['standard_remain'] = avg
            return f'Standard remain ratio {avg}'

    @staticmethod
    def get_remain_through_standard_remain(standard_remain, remain):
        scale = 1 / standard_remain
        remain = remain * scale
        remain = min(1, remain)
        return remain

    @staticmethod
    def momentum_reduce_mode(new_pred, old_pred, alpha=0.9):
        if math.isnan(old_pred):
            result = new_pred
        else:
            result = new_pred * (1 - alpha) + old_pred * alpha
        return result

    @staticmethod
    def remain_pixel_mode(pred, main_classes_idx, sub_classes_idx):
        num_main_pixel = (pred == main_classes_idx).sum()
        num_sub_pixel = (pred == sub_classes_idx).sum()
        result = num_main_pixel / (num_main_pixel + num_sub_pixel + 1)
        return result

    @staticmethod
    def remain_bbox_mode(pred, main_classes_idx):
        num_main_pixel = (pred == main_classes_idx).sum()
        num_sub_pixel = pred.shape[0] * pred.shape[1]
        result = num_main_pixel / num_sub_pixel
        return result

    @staticmethod
    def remain_area_mode(pred, main_classes_idx, sub_classes_idx):
        raise NotImplemented('目前尚未支持area_mode')


def test():
    import cv2
    import torch
    from SpecialTopic.YoloxObjectDetection.api import init_model as init_object_detection
    from SpecialTopic.YoloxObjectDetection.api import detect_image as detect_object_detection_image
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    object_detection_model = init_object_detection(pretrained='/Users/huanghongyan/Downloads/900_yolox_850.25.pth',
                                                   num_classes=9)
    module = SegformerRemainDetection(remain_module_file='./prepare/remain_segformer_module_cfg.json',
                                      classes_path='./prepare/remain_segformer_detection_classes.txt',
                                      with_color_platte='FoodAndNotFood', reduce_mode=dict(type='momentum', alpha=0),
                                      check_init_ratio_frame=5, with_draw=True)
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        # image = cv2.imread('/Users/huanghongyan/Downloads/Donburi/100/1.jpg')
        if ret:
            image_height, image_width = image.shape[:2]
            results = detect_object_detection_image(object_detection_model, device, image, (640, 640), 9,
                                                    confidence=0.7)
            labels, scores, boxes = results
            data = list()
            for index, (label, score, box) in enumerate(zip(labels, scores, boxes)):
                ymin, xmin, ymax, xmax = box
                if ymin < 0 or xmin < 0 or ymax >= image_height or xmax >= image_width:
                    continue
                box = xmin, ymin, xmax, ymax
                info = dict(position=box, category_from_object_detection='Donburi', object_score=score, track_id=index,
                            using_last=False, remain_category_id='0', label=label)
                data.append(info)
            inputs = dict(image=image, track_object_info=data)
            image, results = module(call_api='remain_detection', inputs=inputs)
            for result in results:
                position = result['position']
                category_from_remain = result['category_from_remain']
                remain_color_picture = result['remain_color_picture']
                label = result['label']
                xmin, ymin, xmax, ymax = position
                ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
                ymin, xmin = max(0, ymin), max(0, xmin)
                ymax, xmax = min(image_height, ymax), min(image_width, xmax)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                info = str(label) + '||' + str(category_from_remain)
                cv2.putText(image, info, (xmin + 30, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                image['rgb_image'][ymin:ymax, xmin:xmax] = image['rgb_image'][ymin:ymax, xmin:xmax] * (1 - 0.5) + \
                                                           remain_color_picture * 0.5
            cv2.imshow('img', image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    print('Testing segformer remain detection')
    test()
