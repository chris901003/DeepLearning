import json
import os
import torch
import numpy as np
from SpecialTopic.ST.utils import get_classes
from SpecialTopic.SegmentationNet.api import init_module, detect_single_picture


class SegformerRemainDetection:
    def __init__(self, remain_module_file, classes_path, save_last_period=60, strict_down=False, reduce_mode='Default',
                 area_mode='Default', with_color_platte='ADE20KDataset'):
        if reduce_mode == 'Default':
            reduce_mode = dict(type='momentum', alpha=0.9)
        if area_mode == 'Default':
            area_mode = 'inside'
        support_remain_cal = {
            'area_mode': self.remain_area_mode,
            'pixel_mode': self.remain_pixel_mode
        }
        area_func = support_remain_cal.get(area_mode, None)
        assert area_func is not None, f'Segformer remain detection當中沒有{area_mode}辨識剩餘量的方式'
        self.area_func = area_func
        self.remain_module_file = remain_module_file
        self.classes_path = classes_path
        self.save_last_period = save_last_period
        self.strict_down = strict_down
        self.reduce_mode = reduce_mode
        self.area_mode = area_mode
        self.with_color_platte = with_color_platte
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.classes, self.num_classes = get_classes(classes_path)
        self.segformer_modules = self.build_modules()
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
                print(f'Segformer remain detection中{module_name}未加載譽訓練權重')
            else:
                model = init_module(model_type='Segformer', phi=phi, pretrained=pretrained, num_classes=self.num_classes,
                                    with_pipeline=self.with_color_platte)
            modules_dict[module_name] = model
        return modules_dict

    def __call__(self, call_api, inputs):
        func = self.support_api.get(call_api, None)
        assert func is not None, f'Segformer reman detection未提供{call_api}函數'
        results = func(**inputs)
        return results

    def remain_detection(self, image, track_object_info):
        for track_object in track_object_info:
            position = track_object.get('position', None)
            track_id = track_object.get('track_id', None)
            using_last = track_object.get('using_last', None)
            remain_category_id = track_object.get('remain_category_id', None)
            assert position is not None and track_id is not None and using_last is not None and \
                   remain_category_id is not None, '傳送到segformer remain detection資料有缺少'
            results = -1
            if using_last:
                results = self.get_last_detection(track_id)
            if results == -1:
                results = self.update_detection(image, position, track_id, remain_category_id)
            track_object['category_from_remain'] = results
        return image, track_object_info

    def get_last_detection(self, track_id):
        if track_id not in self.keep_last.keys():
            return -1
        remain = self.keep_last[track_id]['remain']
        self.keep_last[track_id]['last_frame'] = self.frame
        return remain

    def update_detection(self, image, position, track_id, remain_category_id):
        image_height, image_width = image.shape[:2]
        xmin, ymin, xmax, ymax = position
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(image_width, xmax), min(image_height, ymax)
        picture = image[ymin:ymax, xmin:xmax, :]
        if self.segformer_modules[remain_category_id] is None:
            pred = np.full((image_height, image_width), 0, dtype=np.int)
        else:
            pred = detect_single_picture(model=self.segformer_modules[remain_category_id], device=self.device,
                                         image_info=picture, with_draw=False)
        result = self.save_to_keep_last(track_id, pred)
        return result

    def save_to_keep_last(self, track_id, pred):
        assert isinstance(pred, np.ndarray), 'pred資料需要是ndarray類型'
        if pred.ndim == 3 and pred.shape[2]:
            raise ValueError('預測出來的圖像需要是單通道')
        if pred.ndim == 3:
            pred = pred.squeeze(axis=-1)
        result = self.area_func(pred)
        if track_id not in self.keep_last.keys():
            self.keep_last[track_id]['remain'] = result
            self.keep_last[track_id]['last_frame'] = self.frame
        else:
            pass
        return result
