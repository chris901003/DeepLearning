import os
from SpecialTopic.ClassifyNet.api import init_model, detect_single_picture
from SpecialTopic.ST.utils import get_classes
from SpecialTopic.WorkingFlow.utils import parser_cfg


class VitRemainDetection:
    def __init__(self, classes_path, remain_module_file, save_last_period=60, strict_down=False):
        """
        Args:
            save_last_period: 保存上次資料的時長，超過多久沒有獲取該ID的訊息就會刪除
            strict_down: 剩餘量強制下降
        """
        self.classes_path = classes_path
        self.remain_module_file = remain_module_file
        self.remain_module_cfg = parser_cfg(remain_module_file)
        assert os.path.isfile(classes_path), '給定的類別文件不存在'
        self.classes_name, self.num_classes = get_classes(classes_path)
        self.modules = self.build_modules()
        self.keep_last = dict()
        self.save_last_period = save_last_period
        # 用來看一個目標已經多久沒有偵測到，是否需要移除
        self.frame = 0
        self.mod_frame = save_last_period * 10
        self.strict_down = strict_down
        self.support_api = {
            'remain_detection': self.remain_detection,
            'get_num_keep_object': self.get_num_keep_object
        }

    def build_modules(self):
        module_dict = dict()
        for module_name, model_cfg in self.remain_module_cfg.items():
            phi = model_cfg.get('phi', None)
            pretrained_path = model_cfg.get('pretrained', None)
            assert phi is not None and pretrained_path is not None, 'VitRemainDetection初始化參數錯誤'
            pretrained = pretrained_path
            if not os.path.isfile(pretrained_path):
                print(f'Remain {module_name} 模型沒有加載預訓練權重，如果有需要請放到 {pretrained} 路徑上')
                pretrained = 'none'
            if pretrained == 'none':
                module = None
            else:
                module = init_model(cfg='none', model_type='VIT', phi=phi, num_classes=self.num_classes,
                                    pretrained=pretrained)
            module_dict[module_name] = module
        return module_dict

    def __call__(self, call_api, inputs):
        func = self.support_api.get(call_api, None)
        assert func is not None, f'Vit remain detection沒有提供{call_api}函數'
        results = func(**inputs)
        self.remove_miss_object()
        self.frame = (self.frame + 1) % self.mod_frame
        return results

    def remain_detection(self, image, track_object_info):
        for track_object in track_object_info:
            position = track_object.get('position', None)
            track_id = track_object.get('track_id', None)
            using_last = track_object.get('using_last', None)
            remain_category_id = track_object.get('remain_category_id', None)
            assert position is not None and track_id is not None and using_last is not None \
                   and remain_category_id is not None, '輸入到剩餘量檢測模塊資料有缺少，請確認變數名稱是否正確'
            results = -1
            if using_last:
                results = self.get_last_detection(track_id)
            if results == -1:
                results = self.update_detection(image, position, track_id, remain_category_id)
            track_object['category_from_remain'] = self.classes_name[results]
        return image, track_object_info

    def get_last_detection(self, track_id):
        if track_id not in self.keep_last.keys():
            return -1
        pred = self.keep_last[track_id]['remain']
        self.keep_last[track_id]['frame'] = self.frame
        return pred

    def update_detection(self, image, position, track_id, remain_category_id):
        image_height, image_width = image.shape[:2]
        xmin, ymin, xmax, ymax = position
        ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
        ymin, xmin = max(0, ymin), max(0, xmin)
        ymax, xmax = min(image_height, ymax), min(image_width, xmax)
        picture = image[ymin:ymax, xmin:xmax]
        if self.modules[remain_category_id] is None:
            pred = 0
        else:
            pred = detect_single_picture(model=self.modules[remain_category_id], image=picture)[0]
            pred = pred.argmax().item()
        pred = self.save_to_keep_last(track_id, pred)
        return pred

    def save_to_keep_last(self, track_id, pred):
        if track_id in self.keep_last.keys():
            if self.strict_down:
                last_pred = self.keep_last[track_id]['remain']
                if last_pred < pred:
                    pred = last_pred
            self.keep_last[track_id]['remain'] = pred
            self.keep_last[track_id]['frame'] = self.frame
        else:
            data = dict(remain=pred, frame=self.frame)
            self.keep_last[track_id] = data
        return pred

    def remove_miss_object(self):
        remove_keys = [track_id for track_id, track_info in self.keep_last.items()
                       if (self.frame - track_info['frame'] + self.mod_frame) % self.mod_frame >= self.save_last_period]
        [self.keep_last.pop(k) for k in remove_keys]

    def get_num_keep_object(self):
        return len(self.keep_last)


def test():
    import cv2
    import torch
    from SpecialTopic.YoloxObjectDetection.api import init_model as init_object_detection
    from SpecialTopic.YoloxObjectDetection.api import detect_image as object_detect_image
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    object_detection_model = init_object_detection(pretrained='/Users/huanghongyan/Downloads/900_yolox_850.25.pth',
                                                   num_classes=9)
    module = VitRemainDetection(classes_path="/Users/huanghongyan/Documents/DeepLearning/SpecialTopic/WorkingFlow/pr"
                                             "epare/remain_detection_classes.txt",
                                remain_module_file="/Users/huanghongyan/Documents/DeepLearning/SpecialTopic/Wor"
                                                   "kingFlow/prepare/remain_module_cfg.json")
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read(0)
        if ret:
            image_height, image_width = image.shape[:2]
            results = object_detect_image(object_detection_model, device, image, (640, 640), 9)
            labels, scores, boxes = results
            data = list()
            for index, (label, score, box) in enumerate(zip(labels, scores, boxes)):
                ymin, xmin, ymax, xmax = box
                if ymin < 0 or xmin < 0 or ymax >= image_height or xmax >= image_width:
                    continue
                box = xmin, ymin, xmax, ymax
                info = dict(position=box, category_from_object_detection='Noodle', object_score=score, track_id=index,
                            using_last=False, remain_category_id='5')
                data.append(info)
            inputs = dict(image=image, track_object_info=data)
            image, results = module(call_api='remain_detection', inputs=inputs)
            for result in results:
                position = result['position']
                category_from_remain = result['category_from_remain']
                xmin, ymin, xmax, ymax = position
                ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
                ymin, xmin = max(0, ymin), max(0, xmin)
                ymax, xmax = min(image_height, ymax), min(image_width, xmax)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                info = category_from_remain
                cv2.putText(image, info, (xmin + 30, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (89, 214, 210), 2, cv2.LINE_AA)
            cv2.imshow('img', image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    print('Test Vit remain detection')
    test()
