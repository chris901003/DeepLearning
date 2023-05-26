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


class SegformerWithDeepV2:
    def __init__(self, remain_module_file, classes_path, with_color_platte="FoodAndNotFood", save_last_period=60,
                 reduce_mode: Union[dict, str] = "Default",
                 initial_depth_info=None, check_init_ratio_frame=30, standard_remain_error="Default",
                 with_seg_draw=False, with_depth_draw=False, remain_filter_std=100,
                 remain_filter_stable_check_period=30, remain_filter_max_len=40):
        if reduce_mode == "Default":
            reduce_mode = dict(type="momentum", alpha=0.7)
        if standard_remain_error == "Default":
            standard_remain_error = [0.95, 1.05]
        assert isinstance(reduce_mode, dict)
        assert isinstance(standard_remain_error, (tuple, list))
        assert len(standard_remain_error) == 2
        assert standard_remain_error[0] < standard_remain_error[1]
        support_reduce_mode = {
            'momentum': self.momentum_reduce_mode
        }
        reduce_func = get_cls_from_dict(support_reduce_mode, reduce_mode)
        reduce_func = partial(reduce_func, **reduce_mode)
        self.initial_depth = self.read_numpy_file(initial_depth_info)
        self.reduce_func = reduce_func
        self.remain_module_file = remain_module_file
        self.classes_path = classes_path
        self.save_last_period = save_last_period
        self.with_color_platte = with_color_platte
        self.check_init_ratio_frame = check_init_ratio_frame
        self.standard_remain_error = standard_remain_error
        self.with_seg_draw = with_seg_draw
        self.with_depth_draw = with_depth_draw
        self.remain_filter_std = remain_filter_std
        self.remain_filter_stable_check_period = remain_filter_stable_check_period
        self.remain_filter_max_len = remain_filter_max_len
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.classes, self.num_classes = get_classes(classes_path)
        self.segformer_modules = self.build_modules()
        self.keep_data = dict()
        self.frame = 0
        self.mod_frame = save_last_period * 10
        self.support_api = {
            'remain_detection': self.remain_detection
        }
        self.logger = None

    def build_modules(self):
        modules_dict = dict()
        with open(self.remain_module_file, 'r') as f:
            modules_config = json.load(f)
        for module_name, module_info in modules_config.items():
            phi = module_info.get('phi', None)
            assert phi is not None
            pretrained = module_info.get('pretrained', None)
            if pretrained is None or not os.path.exists(pretrained):
                model = None
                print(f'Segformer with deep remain detection中{module_name}未加載預訓練權重')
            else:
                model = init_segmentation_module(model_type='Segformer', phi=phi, pretrained=pretrained,
                                                 num_classes=self.num_classes, with_color_platte=self.with_color_platte)
            modules_dict[module_name] = model
        return modules_dict

    @staticmethod
    def read_numpy_file(file_path):
        result = np.load(file_path)
        return result

    def __call__(self, call_api, inputs=None):
        func = self.support_api.get(call_api, None)
        assert func is not None
        results = func(**inputs)
        self.remove_miss_object()
        self.frame = (self.frame + 1) % self.mod_frame
        return results

    def remain_detection(self, image, track_object_info):
        assert 'rgb_image' in image.keys()
        assert 'deep_image' in image.keys()
        if self.with_depth_draw:
            assert 'deep_draw' in image.keys()
        for track_object in track_object_info:
            position = track_object.get('position', None)
            track_id = track_object.get('track_id', None)
            using_last = track_object.get('using_last', None)
            remain_category_id = track_object.get('remain_category_id', None)
            assert position is not None and track_id is not None and using_last is not None and \
                   remain_category_id is not None
            results = -1
            if using_last:
                results = self.get_last_detection(track_id)
            if results == -1:
                results = self.update_detection(image, position, track_id, remain_category_id,
                                                with_seg_draw=self.with_seg_draw, with_depth_draw=self.with_depth_draw)
            self.keep_data[track_id]["remain_seg_picture"] = results["rgb_draw"]
            self.keep_data[track_id]["remain_depth_picture"] = results["depth_draw"]
            self.keep_data[track_id]["last_frame"] = self.frame
            track_object["category_from_remain"] = results["remain"]
            track_object["remain_color_picture"] = results["rgb_draw"]
            track_object["remain_deep_picture"] = results["depth_draw"]
            if isinstance(track_object["category_from_remain"], (int, float)):
                self.keep_data[track_id]["remain"] = track_object["category_from_remain"]
        return image, track_object_info

    def get_last_detection(self, track_id):
        if track_id not in self.keep_data.keys():
            return -1
        standard_remain = self.keep_data[track_id]["standard_remain"]
        if standard_remain == -1:
            return -1
        remain = self.keep_data[track_id]["remain"]
        if remain == -1:
            return -1
        rgb_draw = self.keep_data[track_id].get('remain_seg_picture', None)
        depth_draw = self.keep_data[track_id].get('remain_depth_picture', None)
        results = dict(remain=remain, rgb_draw=rgb_draw, depth_draw=depth_draw)
        return results

    def update_detection(self, image, position, track_id, remain_category_id, with_seg_draw=False,
                         with_depth_draw=False):
        rgb_image = image["rgb_image"]
        depth_image = image["deep_image"]
        depth_color = image.get("deep_draw", None)
        image_height, image_width = rgb_image.shape[:2]
        xmin, ymin, xmax, ymax = position
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(image_width, xmax), min(image_height, ymax)
        rgb_picture = rgb_image[ymin:ymax, xmin:xmax, :]
        if self.segformer_modules[remain_category_id] is None:
            seg_height, seg_width = ymax - ymin, xmax - xmin
            if self.with_seg_draw:
                draw_image_mix = np.zeros((seg_height, seg_width, 3), dtype=np.uint8)
                draw_image = np.zeros((seg_height, seg_width, 3), dtype=np.uint8)
                seg_pred = np.zeros((seg_height, seg_width), dtype=np.int)
                pred = (draw_image_mix, draw_image, seg_pred)
            else:
                pred = np.full((seg_height, seg_width), 0, dtype=np.int)
        else:
            pred = segmentation_detect_single_picture(model=self.segformer_modules[remain_category_id],
                                                      device=self.device, image_info=rgb_picture,
                                                      with_draw=with_seg_draw, threshold=0.7)
        depth_data = depth_image[ymin:ymax, xmin:xmax]
        initial_depth_info = self.initial_depth[ymin:ymax, xmin:xmax]
        if with_depth_draw:
            depth_color_picture = depth_color[ymin:ymax, xmin:xmax]
        else:
            depth_color_picture = None
        if with_seg_draw:
            remain = self.save_to_keep_last(track_id, pred[2], depth_data, initial_depth_info)
            results = dict(remain=remain, rgb_draw=pred[1], depth_draw=depth_color_picture)
        else:
            remain = self.save_to_keep_last(track_id, pred, depth_data, initial_depth_info)
            results = dict(remain=remain, rgb_draw=None, depth_draw=depth_color_picture)
        return results

    def save_to_keep_last(self, track_id, seg_pred, depth_data, initial_depth_info):
        assert isinstance(seg_pred, np.ndarray)
        if seg_pred.ndim == 3 and seg_pred.shape[2] != 1:
            assert False
        if seg_pred.ndim == 3:
            seg_pred = seg_pred.squeeze(axis=-1)
        if track_id not in self.keep_data.keys():
            data = self.get_empty_data()
            self.keep_data[track_id] = data
        if self.keep_data[track_id]["initial_volume"] == -1:
            basic_volume = self.get_volume(seg_pred, depth_data, initial_depth_info)
            self.keep_data[track_id]["record_initial_volume"].append(basic_volume)
            return self.get_standard_volume(track_id)
        else:
            last_remain = self.keep_data[track_id]["remain"]
            current_volume = self.get_volume(seg_pred, depth_data, initial_depth_info)
            remain = self.get_current_remain(current_volume, self.keep_data[track_id]["initial_volume"])
            result = self.reduce_func(last_remain, remain) if last_remain != -1 else remain
            return result

    # 獲取原始食物體積資料
    def get_standard_volume(self, track_id):
        if len(self.keep_data[track_id]["record_initial_volume"]) < self.check_init_ratio_frame:
            return "Init standard remain ratio ..."
        average_basic_volume = 0
        for init_info in self.keep_data[track_id]["record_initial_volume"]:
            average_basic_volume += init_info
        average_basic_volume /= self.check_init_ratio_frame
        self.keep_data[track_id]["initial_volume"] = average_basic_volume
        return f'Initial volume = {average_basic_volume}'

    def remove_miss_object(self):
        remove_keys = [track_id for track_id, track_info in self.keep_data.items()
                       if ((self.frame - track_info['last_frame'] + self.mod_frame)
                       % self.mod_frame) > self.save_last_period]
        [self.logger['logger'].info(f'Track ID: {k}, Delete') for k in remove_keys]
        [self.keep_data.pop(k) for k in remove_keys]

    # 將seg_pred中為0的部分提取出來，計算出目前的體積
    @staticmethod
    def get_volume(seg_pred, depth_data, initial_depth_info):
        main_target_mask = seg_pred == 0
        current_depth_data = depth_data[main_target_mask]
        initial_depth_data = initial_depth_info[main_target_mask]
        volume_diff = initial_depth_data - current_depth_data
        volume = np.sum(volume_diff)
        return volume

    # 獲取當前剩餘量比例
    @staticmethod
    def get_current_remain(current_volume, init_volume):
        return current_volume / init_volume * 100

    @staticmethod
    def momentum_reduce_mode(old_pred, new_pred, alpha=0.7):
        if math.isnan(old_pred):
            return new_pred
        else:
            return new_pred * (1 - alpha) + old_pred * alpha

    def get_empty_data(self):
        data = dict(remain=-1, last_frame=self.frame, initial_volume=-1, record_initial_volume=list())
        return data


def test():
    print("Now you are testing SegformerWithDeepV2")
    import time
    import cv2
    import logging
    from SpecialTopic.WorkingFlow.SubBlock.ReadPicture.ReadPictureFromD435 import ReadPictureFromD435
    # from SpecialTopic.WorkingFlow.SubBlock.ObjectDetection.YoloxObjectDetection import YoloxObjectDetection
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logging.Formatter = logging.Formatter('%(asctime)s - %(name)s - %(lineno)d - %(message)s')
    logger = dict(logger=logger, sub_log=None)
    picture_reader = ReadPictureFromD435(deep_color_range=[700, 1200])
    # picture_reader = ReadPictureFromD435(rgb_image_width=960, rgb_image_height=540, deep_image_width=960,
    #                                          deep_image_height=540, deep_color_range=[700, 800])
    # picture_reader = ReadPictureFromKinectV2(deep_color_range=[500, 1500])
    picture_reader.logger = logger
    # object_detection_config = {
    #     "phi": "l",
    #     "pretrained": "./checkpoint/object_detection/yolox_l.pth",
    #     "classes_path": "./prepare/object_detection_classes.txt",
    #     "confidence": 0.7,
    #     "nms": 0.3,
    #     "filter_edge": False,
    #     "cfg": "auto",
    #     "new_track_box": 10,
    #     "average_time_output": 1
    # }
    # object_detection_model = YoloxObjectDetection(**object_detection_config)
    # object_detection_model.logger = logger
    module_config = {
        'remain_module_file': './prepare/remain_segformer_module_cfg.json',
        'classes_path': './prepare/remain_segformer_detection_classes.txt',
        'with_color_platte': 'FoodAndNotFood',
        'save_last_period': 60,
        'reduce_mode': {
            'type': 'momentum', 'alpha': 0.5
        },
        'check_init_ratio_frame': 5,
        'standard_remain_error': [0.8, 1.2],
        'with_seg_draw': True,
        'with_depth_draw': True,
        'initial_depth_info': r'C:\DeepLearning\some_utils\initial_depth.npy'
    }
    remain_module = SegformerWithDeepV2(**module_config)
    remain_module.logger = logger
    pTime = 0
    while True:
        picture_info = picture_reader(api_call='get_single_picture')
        rgb_image, image_type, deep_image, deep_draw = picture_info
        image_height, image_width = rgb_image.shape[:2]
        # image = dict(image=rgb_image, image_type=image_type, deep_image=deep_image, deep_draw=deep_draw,
        #              force_get_detect=True)
        # object_detection_result = object_detection_model(call_api='detect_single_picture', input=image)

        image = dict(rgb_image=rgb_image, deep_image=deep_image, deep_draw=deep_draw)
        track_object_info = [dict(position=[368, 245, 550, 418], track_id=0, using_last=False,
                                  category_from_object_detection='Test')]
        detect_results = (list(), list(), list())
        object_detection_result = (
            image, track_object_info, detect_results
        )

        image, track_object_info, detect_results = object_detection_result
        for track_object in track_object_info:
            track_object['remain_category_id'] = '0'
        inputs = dict(image=image, track_object_info=track_object_info)
        image, results = remain_module(call_api='remain_detection', inputs=inputs)
        labels, scores, boxes = detect_results
        for label, score, box in zip(labels, scores, boxes):
            ymin, xmin, ymax, xmax = box
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(image_width, xmax), min(image_height, ymax)
            cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)

        for result in results:
            position = result['position']
            category_from_remain = result['category_from_remain']
            remain_color_picture = result['remain_color_picture']
            remain_deep_picture = result['remain_deep_picture']
            label = result['category_from_object_detection']
            xmin, ymin, xmax, ymax = position
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(image_width, xmax), min(image_height, ymax)
            cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.rectangle(deep_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            info = str(category_from_remain)
            cv2.putText(rgb_image, info, (xmin + 30, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            rgb_image[ymin:ymax, xmin:xmax] = rgb_image[ymin:ymax, xmin:xmax] * 0.5 + remain_color_picture * 0.5
            # deep_draw[ymin:ymax, xmin:xmax] = deep_draw[ymin:ymax, xmin:xmax] * 0.5 + remain_deep_picture * 0.5
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(rgb_image, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('rgb_image', rgb_image)
        cv2.imshow('deep_image', deep_draw)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    test()
