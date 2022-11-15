import copy
import cv2
import numpy as np
from SpecialTopic.ST.utils import get_cls_from_dict


class DrawResultsOnPicture:
    def __init__(self, triangles, texts, pictures):
        self.triangles = triangles
        self.texts = texts
        self.pictures = pictures
        self.support_position_type = {
            'xyxy': self.get_xyxy, 'yxyx': self.get_yxyx, 'xcycwh': self.get_xcycwh, 'xmymwh': self.get_xmymwh
        }
        self.support_api = {
            'show_results': self.show_results
        }

    def __call__(self, call_api, inputs):
        func = self.support_api.get(call_api, None)
        assert func is not None, f'Draw results on picture沒有{call_api}函數'
        results = func(**inputs)
        return results

    def show_results(self, image, track_object_info):
        result_image = copy.deepcopy(image['rgb_image'])
        result_image = self.draw_triangle(result_image, track_object_info)
        result_image = self.write_text(result_image, track_object_info)
        result_image = self.paste_picture(result_image, track_object_info)
        return result_image, track_object_info

    def draw_triangle(self, result_image, track_object_info):
        image_height, image_width = result_image.shape[:2]
        for triangle_info in self.triangles:
            triangle_info_ = copy.deepcopy(triangle_info)
            position_function = get_cls_from_dict(self.support_position_type, triangle_info_)
            for track_info in track_object_info:
                val_name = triangle_info_.get('val_name', None)
                assert val_name is not None, '需要指定用哪個資料畫圖'
                position = track_info.get(val_name, None)
                assert position is not None, f'傳入的資料當中沒有{val_name}資料，請確認是否正確'
                position = position_function(position)
                position = self.filter_box(position, image_height, image_width)
                color = triangle_info_.get('color', (0, 255, 0))
                thick = triangle_info_.get('thick', 3)
                cv2.rectangle(result_image, (position[0], position[1]), (position[2], position[3]), color, thick)
                track_info['draw_position'] = (position[0], position[1])
        return result_image

    def write_text(self, result_image, track_object_info):
        for index, text_info in enumerate(self.texts):
            prefix = text_info.get('prefix', '')
            suffix = text_info.get('suffix', '')
            vals_name = text_info.get('val_name', None)
            assert vals_name is not None, '需要指定文字的參數'
            sep = text_info.get('sep', ' ')
            color = text_info.get('color', (0, 0, 255))
            text_size = text_info.get('text_size', 1)
            thick = text_info.get('thick', 2)
            for track_info in track_object_info:
                info = prefix
                for val_name in vals_name:
                    val = track_info.get(val_name, None)
                    assert val is not None, f'無法獲取{val_name}資料'
                    info += self.get_string_type(val) + sep
                info += suffix
                draw_position = track_info['draw_position']
                cv2.putText(result_image, info,
                            (draw_position[0] + 30, draw_position[1] + 30 * (index + 1)),
                            cv2.FONT_HERSHEY_SIMPLEX, text_size, color, thick, cv2.LINE_AA)
        return result_image

    def paste_picture(self, result_image, track_object_info):
        image_height, image_width = result_image.shape[:2]
        for picture_info in self.pictures:
            val_name = picture_info.get('val_name', None)
            position_name = picture_info.get('position', None)
            opacity = picture_info.get('opacity', 0.5)
            assert val_name is not None and position_name is not None, '缺少圖像變數名稱以及座標變數名稱'
            for track_info in track_object_info:
                picture = track_info.get(val_name, None)
                position = track_info.get(position_name, None)
                assert picture is not None and position is not None, '缺少變數名稱'
                xmin, ymin, xmax, ymax = position
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(image_width, xmax), min(image_height, ymax)
                print(xmax, xmin, ymax, ymin, picture.shape)
                picture = cv2.resize(picture.astype(np.uint8), (xmax - xmin, ymax - ymin))
                result_image[ymin:ymax, xmin:xmax] = result_image[ymin:ymax, xmin:xmax] * \
                                                     (1 - opacity) + picture * opacity
        return result_image

    @staticmethod
    def get_string_type(val):
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, str):
            return val
        if isinstance(val, np.ndarray):
            if len(val) == 1:
                return str(int(val))
            else:
                raise ValueError('目前不支援多數值的ndarray轉換')

    @staticmethod
    def get_xyxy(position):
        xmin, ymin, xmax, ymax = position
        return xmin, ymin, xmax, ymax

    @staticmethod
    def get_yxyx(position):
        ymin, xmin, ymax, xmax = position
        return xmin, ymin, xmax, ymax

    @staticmethod
    def get_xcycwh(position):
        x_center, y_center, width, height = position
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2
        return xmin, ymin, xmax, ymax

    @staticmethod
    def get_xmymwh(position):
        xmin, ymin, width, height = position
        xmax = xmin + width
        ymax = ymin + height
        return xmin, ymin, xmax, ymax

    @staticmethod
    def filter_box(position, height, width):
        xmin, ymin, xmax, ymax = position
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(width, xmax), min(height, ymax)
        return xmin, ymin, xmax, ymax


def test():
    import torch
    from SpecialTopic.YoloxObjectDetection.api import init_model as init_object_detection
    from SpecialTopic.YoloxObjectDetection.api import detect_image as object_detect_image
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    object_detection_model = init_object_detection(pretrained='/Users/huanghongyan/Downloads/900_yolox_850.25.pth',
                                                   num_classes=9)
    triangles = [{"type": "yxyx", "val_name": "position", "color": [0, 255, 0], "thick": 3}]
    texts = [
        {"prefix": "Object: ", "val_name": ["object_score", "category_from_object_detection"], "suffix": "",
         "sep": "|"},
        {"prefix": "Remain: ", "val_name": ["track_id", "category_from_remain"], "suffix": "", "sep": "|"},
        {"prefix": "Other: ", "val_name": ["using_last"], "suffix": "", "sep": "|"}
    ]
    module = DrawResultsOnPicture(triangles=triangles, texts=texts)
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
                score = round(score * 100, 2)
                info = dict(position=box, category_from_object_detection='Noodle', object_score=score, track_id=index,
                            using_last=False, remain_category_id='5', category_from_remain='75')
                data.append(info)
            inputs = dict(image=image, track_object_info=data)
            image = module(call_api='show_results', inputs=inputs)
            cv2.imshow('img', image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    print('Test Draw results on picture')
    test()
