import json
import os
import torch
import numpy as np
from typing import Union
import math
from functools import partial
from SpecialTopic.ST.utils import get_classes, get_cls_from_dict
from SpecialTopic.SegmentationNet.api import init_module, detect_single_picture
from SpecialTopic.Deploy.SegmentationNet.api import create_tensorrt_engine


class SegformerRemainDetection:
    def __init__(self, remain_module_file, fp16=True, save_last_period=60, strict_down=False,
                 reduce_mode: Union[str, dict] = 'Default', area_mode: Union[str, dict] = 'Default',
                 with_color_platte: Union[str, dict] = 'FoodAndNotFood', check_init_ratio_frame=10, std_error=0.5,
                 with_draw=False):
        """
        Args:
            remain_module_file: 配置剩餘量模型的config資料，目前是根據不同類別會啟用不同的分割權重模型
            fp16: 在使用TensorRT推理時是否要啟用fp16模式
            save_last_period: 最多可以保存多少幀沒有獲取到該id的圖像
            strict_down: 強制剩餘量只會越來越少
            reduce_mode: 對於當前檢測剩餘量的平均方式，可以減少預測結果突然暴增
            area_mode: 計算當前剩餘量的方式
            with_color_platte: 使用的調色盤，可以使用以存在的調色盤名稱，或是用字典方式將資料傳入
                dict: {
                    'CLASSES': [], 'PALETTE': []
                }
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
        self.fp16 = fp16
        self.save_last_period = save_last_period
        self.strict_down = strict_down
        self.reduce_mode = reduce_mode
        self.area_mode = area_mode
        self.with_color_platte = with_color_platte
        self.check_init_ratio_frame = check_init_ratio_frame
        self.std_error = std_error
        self.with_draw = with_draw
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        self.logger = None

    def build_modules(self):
        """ 構建分割網路模型，如果該模型沒有提供權重就會用None取代
        """
        models_list = list()
        with open(self.remain_module_file, 'r') as f:
            module_config = json.load(f)
        for module_name, module_info in module_config.items():
            onnx_file_path = module_info.get('onnx_file_path', None)
            trt_file_path = module_info.get('trt_file_path', None)
            save_trt_file_path = module_info.get('save_trt_file_path', None)
            if trt_file_path is None or not os.path.exists(trt_file_path):
                if onnx_file_path is None or not os.path.exists(onnx_file_path):
                    tensorrt_engine = None
                    print(f'')
            tensorrt_engine = create_tensorrt_engine(onnx_file_path=onnx_file_path, fp16_mode=self.fp16,
                                                     max_batch_size=self.max_batch_size,
                                                     save_trt_engine_path=save_trt_file_path,
                                                     trt_engine_path=trt_file_path,
                                                     dynamic_shapes=self.dynamic_shapes)
            models_list.append(tensorrt_engine)
        return models_list

    def __call__(self, call_api, inputs):
        func = self.support_api.get(call_api, None)
        assert func is not None, self.logger['logger'].critical(f'Segformer reman detection未提供{call_api}函數')
        results = func(**inputs)
        self.remove_miss_object()
        self.frame = (self.frame + 1) % self.mod_frame
        return results

    def remain_detection(self, image, track_object_info):
        """
        Args:
            image: 圖像相關資料，包含彩色圖像以及深度圖像資料
            track_object_info: 當前正在追蹤目標的資料
        """
        with_draw = self.with_draw
        for track_object in track_object_info:
            position = track_object.get('position', None)
            track_id = track_object.get('track_id', None)
            using_last = track_object.get('using_last', None)
            remain_category_id = track_object.get('remain_category_id', None)
            assert position is not None and track_id is not None and using_last is not None and \
                   remain_category_id is not None, self.logger['logger'].critical('傳送到segformer '
                                                                                  'remain detection資料有缺少')
            results = -1
            if using_last:
                results = self.get_last_detection(track_id, with_draw=with_draw)
                if results != -1:
                    self.logger['logger'].debug(f'Track ID: [ {track_id} ]將使用上次結果')
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
                self.logger['logger'].debug(f'Track ID: {track_id}, Remain: {results[0]}')
                track_object['category_from_remain'] = results[0]
                self.logger['logger'].debug(f'Track ID: [ {track_id} ], 剩餘量結果: [ {results[0]} ]')
                track_object['remain_color_picture'] = results[1]
            else:
                self.logger['logger'].debug(f'Track ID: {track_id}, Remain: {results}')
                track_object['category_from_remain'] = results
                self.logger['logger'].debug(f'Track ID: [ {track_id} ], 剩餘量結果: [ {results} ]')
        return image, track_object_info

    def get_last_detection(self, track_id, with_draw=False):
        """ 根據track_id獲取最後一次的剩餘量結果
        Args:
            track_id: 追蹤對象ID
            with_draw: 是否需要分割圖的標註
        Returns:
            remain: 如果有最後剩餘量資料就直接回傳，否則就會是-1
        """
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
        """ 對指定追蹤對象偵測剩餘量
        Args:
            image: 圖像相關資料
            position: 追蹤目標在圖像當中的座標位置
            track_id: 追蹤對象的ID
            remain_category_id: 對應上剩餘量模型的類別，根據此參數會使用不同權重預測
            with_draw: 是否需要標註分割圖像
        """
        image_height, image_width = image['rgb_image'].shape[:2]
        xmin, ymin, xmax, ymax = position
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(image_width, xmax), min(image_height, ymax)
        picture = image['rgb_image'][ymin:ymax, xmin:xmax, :]
        if self.segformer_modules[remain_category_id] is None:
            self.logger['logger'].error(f'Track ID: {track_id}, Not have match remain detection model')
            seg_height, seg_width = ymax - ymin, xmax - xmin
            if with_draw:
                draw_image_mix = np.full((seg_height, seg_width, 3), 0, dtype=np.int)
                draw_image = np.full((seg_height, seg_width, 3), 0, dtype=np.int)
                seg_pred = np.full((seg_height, seg_width), 0, dtype=np.int)
                pred = (draw_image_mix, draw_image, seg_pred)
            else:
                pred = np.full((image_height, image_width), 0, dtype=np.int)
        else:
            pred = detect_single_picture(model=self.segformer_modules[remain_category_id], device=self.device,
                                         image_info=picture, with_draw=with_draw)
        if with_draw:
            # 將結果保存下來
            result = self.save_to_keep_last(track_id, pred[2])
            self.keep_last[track_id]['remain_color_picture'] = pred[1]
            return result, pred[1]
        else:
            # 將結果保存下來
            result = self.save_to_keep_last(track_id, pred)
            return result

    def save_to_keep_last(self, track_id, pred):
        """ 將剩餘量資料保存下來
        Args:
            track_id: 追蹤對象ID
            pred: 預測的分割圖
        """
        assert isinstance(pred, np.ndarray), self.logger['logger'].critical('pred資料需要是ndarray類型')
        if pred.ndim == 3 and pred.shape[2]:
            self.logger['logger'].critical('預測出來的圖像需要是單通道')
            raise ValueError('預測出來的圖像需要是單通道')
        if pred.ndim == 3:
            self.logger['logger'].debug('預測輸出通道多了channel維度，正常是不需要的')
            pred = pred.squeeze(axis=-1)
        result = self.area_func(pred)
        if track_id not in self.keep_last.keys():
            self.logger['logger'].info(f'Track ID: {track_id}, add in to segformer remain detection.')
            data = dict(remain=-1, last_frame=self.frame, standard_remain=-1, standard_remain_record=list())
            self.keep_last[track_id] = data
        if self.keep_last[track_id]['standard_remain'] == -1:
            self.keep_last[track_id]['standard_remain_record'].append(result)
            # 查看是否可以獲取標準剩餘量判斷比例，result會表示當前狀態
            result = self.get_standard_remain(track_id)
        else:
            last_remain = self.keep_last[track_id]['remain']
            result = self.reduce_func(result, last_remain) if last_remain != -1 else result
            if self.strict_down and last_remain != -1:
                result = min(result, self.keep_last[track_id]['remain'])
            self.keep_last[track_id]['remain'] = result
            self.keep_last[track_id]['last_frame'] = self.frame
            standard_remain = self.keep_last[track_id]['standard_remain']
            result = self.get_remain_through_standard_remain(standard_remain, result)
        return result

    def remove_miss_object(self):
        """ 將沒有追蹤到的目標拋棄
        """
        remove_keys = [track_id for track_id, track_info in self.keep_last.items()
                       if (self.frame - track_info['last_frame'] + self.mod_frame)
                       % self.mod_frame > self.save_last_period]
        [self.logger['logger'].debug(f'Track ID: {k}, Delete') for k in remove_keys]
        [self.keep_last.pop(k) for k in remove_keys]

    def get_standard_remain(self, track_id):
        """ 嘗試著根據一段時間的資料定義原始100%比例
        Args:
            track_id: 追蹤目標ID
        """
        if len(self.keep_last[track_id]['standard_remain_record']) < self.check_init_ratio_frame:
            return 'Init standard remain ratio ...'
        standard_remain_record = np.array(self.keep_last[track_id]['standard_remain_record'])
        std = standard_remain_record.std()
        avg = standard_remain_record.mean()
        if std > self.std_error:
            self.keep_last[track_id]['standard_remain_record'] = list()
            self.logger['logger'].info(f'Track ID: {track_id}, Init std too large')
            return f'Standard deviation is {std} lager then setting need recollect'
        else:
            self.keep_last[track_id]['standard_remain'] = avg
            self.logger['logger'].info(f'Track ID: {track_id}, Standard remain: {avg}')
            return f'Standard remain ratio {avg}'

    @staticmethod
    def get_remain_through_standard_remain(standard_remain, remain):
        """ 根據基礎值獲取當前應當的剩餘量
        """
        scale = 1 / standard_remain
        remain = remain * scale
        remain = min(1, remain)
        return remain

    @staticmethod
    def momentum_reduce_mode(new_pred, old_pred, alpha=0.9):
        """ 防止突發數據以及讓下降曲線平滑
        """
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
    import logging
    import cv2
    import torch
    import logging
    from SpecialTopic.YoloxObjectDetection.api import init_model as init_object_detection
    from SpecialTopic.YoloxObjectDetection.api import detect_image as detect_object_detection_image
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    object_detection_model = init_object_detection(pretrained='/Users/huanghongyan/Downloads/900_yolox_850.25.pth',
                                                   num_classes=9)
    module = SegformerRemainDetection(remain_module_file='./prepare/remain_segformer_module_cfg.json',
                                      classes_path='./prepare/remain_segformer_detection_classes.txt',
                                      with_color_platte='FoodAndNotFood', reduce_mode=dict(type='momentum', alpha=0),
                                      check_init_ratio_frame=5, with_draw=True)
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger = dict(logger=logger, sub_log=None)
    module.logger = logger
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
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
            image = dict(rgb_image=image)
            inputs = dict(image=image, track_object_info=data)
            image, results = module(call_api='remain_detection', inputs=inputs)
            image = image['rgb_image']
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
                image[ymin:ymax, xmin:xmax] = image[ymin:ymax, xmin:xmax] * (1 - 0.5) + remain_color_picture * 0.5
            cv2.imshow('img', image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    print('Testing segformer remain detection')
    test()
