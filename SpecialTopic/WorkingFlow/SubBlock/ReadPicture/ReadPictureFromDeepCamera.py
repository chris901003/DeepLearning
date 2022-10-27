from openni import openni2
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import Union
from SpecialTopic.ST.utils import get_cls_from_dict
import cv2


class ReadPictureFromDeepCamera:
    def __init__(self, color_palette_type='jet_r', color_palette_range=255, deep_image_height=480, deep_image_width=640,
                 rgb_camera: Union[int, str] = 'Default', rgb_image_height=480, rgb_image_width=640, min_deep_value=300,
                 max_deep_value=6000, deep_match_rgb_cfg: Union[str, dict] = 'Default'):
        """
        Args:
            color_palette_type: 色彩條選擇
            color_palette_range: 色彩範圍，這裡需要根據使用的色彩條選擇範圍
            deep_image_height: 深度圖高度
            deep_image_width: 深度圖寬度
            rgb_camera: rgb圖像的攝影機，如果是Default就會是直接使用深度攝影機的rgb鏡頭
                如果是要額外使用其他攝影機就需要提供對應ID，同時需要處理RGB圖像與深度圖像的關係
            rgb_image_height: 彩色圖高度
            rgb_image_width: 彩色圖寬度
            min_deep_value: 深度最小值
            max_deep_value: 深度最大值
            deep_match_rgb_cfg: 深度圖像映射到rgb圖像的方式，這裡傳入的需要是一個dict格式
        """
        self.color_palette_type = color_palette_type
        self.color_palette_range = color_palette_range
        self.deep_image_height = deep_image_height
        self.deep_image_width = deep_image_width
        self.rgb_camera = rgb_camera
        self.rgb_image_height = rgb_image_height
        self.rgb_image_width = rgb_image_width
        self.min_deep_value = min_deep_value
        self.max_deep_value = max_deep_value
        if deep_match_rgb_cfg == 'Default':
            deep_match_rgb_cfg = {
                'type': 'brute_force_resize'
            }
        support_deep_match_rgb = {
            'brute_force_resize': self.brute_force_resize,
            'custom_match': self.custom_match
        }
        deep_match_rgb_func = get_cls_from_dict(support_deep_match_rgb, deep_match_rgb_cfg)
        self.deep_match_rgb_func = partial(deep_match_rgb_func, **deep_match_rgb_cfg)
        openni2.initialize()
        deep_camera_device = openni2.Device.open_any()
        self.deep_camera_device_info = deep_camera_device.get_device_info()
        self.depth_stream = deep_camera_device.create_depth_stream()
        self.depth_stream.start()
        if rgb_camera == 'Default':
            self.color_stream = deep_camera_device.create_color_stream()
            self.color_stream.start()
        else:
            assert isinstance(rgb_camera, int), '攝影機ID會是int格式'
            self.color_stream = cv2.VideoCapture(rgb_camera)
        self.color_palette = plt.cm.get_cmap(color_palette_type)(range(color_palette_range))
        self.support_api = {
            'get_single_picture': self.get_single_picture
        }
        self.logger = None

    def __call__(self, api_call, input=None):
        func = self.support_api.get(api_call, None)
        assert func is not None, f'Read picture from deep camera中沒有{api_call}可以使用'
        if input is not None:
            results = func(**input)
        else:
            results = func()
        return results

    def get_single_picture(self):
        """ 獲取一幀圖像
        Returns:
            rgb_image: 彩色圖像資料
            ndarray: 輸出的資料型態
            deep_image: 深度資料，每個點表示的就直接是該像素點的深度資料
            deep_draw: 透過深度配合上色盤讓深度更加可視化
        """
        self.logger['logger'].debug('get single picture')
        deep_frame = self.depth_stream.read_frame()
        deep_frame_data = np.array(deep_frame.get_buffer_as_triplet())
        deep_frame_data = deep_frame_data.reshape([self.deep_image_height, self.deep_image_width, 2])
        dpt1 = np.asarray(deep_frame_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(deep_frame_data[:, :, 1], dtype='float32')
        dpt2 *= 255
        # 最終深度圖資料
        dpt = dpt1 + dpt2
        # dpt = dpt2
        if isinstance(self.color_stream, cv2.VideoCapture):
            ret, rgb_image = self.color_stream.read()
            if not ret:
                raise RuntimeError('無法從攝影機獲取圖像資料')
        else:
            cframe = self.color_stream.read_frame()
            cframe_data = np.array(cframe.get_buffer_as_triplet())
            cframe_data = cframe_data.reshape([self.rgb_image_height, self.rgb_image_width, 3])
            rgb_image = cv2.cvtColor(cframe_data, cv2.COLOR_RGB2BGR)
        deep_image, rgb_image = self.deep_match_rgb_func(dpt, rgb_image)
        deep_draw = self.draw_deep_image(deep_image)
        return rgb_image, 'ndarray', deep_image, deep_draw

    def draw_deep_image(self, deep_image):
        """ 將深度資料用不同顏色來表示，增加可視化能力
        Args:
            deep_image: 深度圖像資料
        Returns:
            color_map: 根據不同深度以及色盤畫出的結果
        """
        dpt_clip = np.clip(deep_image, self.min_deep_value, self.max_deep_value)
        dpt_clip = (dpt_clip - self.min_deep_value) / (self.max_deep_value - self.min_deep_value) * \
                   (self.color_palette_range - 2)
        dpt_clip = dpt_clip.astype(int)
        image_height, image_width = deep_image.shape[:2]
        color_map = np.zeros((image_height, image_width, 3))
        for i in range(0, self.color_palette_range):
            color_map[dpt_clip == i] = self.color_palette[i][:3]
        color_map = (color_map * 255).astype(np.uint8)
        return color_map

    @staticmethod
    def brute_force_resize(deep_image, rgb_image, interpolate_mode='INTER_LINEAR'):
        support = {
            'INTER_NEAREST': cv2.INTER_NEAREST,
            'INTER_LINEAR': cv2.INTER_LINEAR,
            'INTER_AREA': cv2.INTER_AREA,
            'INTER_CUBIC': cv2.INTER_CUBIC,
            'INTER_LANCZOS4': cv2.INTER_LANCZOS4
        }
        rgb_image_height, rgb_image_width = rgb_image.shape[:2]
        interpolate_mode = support.get(interpolate_mode)
        resize_deep_image = cv2.resize(deep_image, (rgb_image_width, rgb_image_height),
                                       interpolation=interpolate_mode)
        return resize_deep_image, rgb_image

    @staticmethod
    def custom_match(deep_image, rgb_image, rgb_image_range, deep_image_range, horizon_flip=False, vertical_flip=False,
                     deep_image_resize=None, rgb_image_resize=None, deep_image_match_rgb=False,
                     rgb_image_match_deep=False):
        """ 預作順序[翻轉->擷取->重設大小->pad補齊->輸出]
        Args:
             deep_image: 深度圖像
             rgb_image: 彩色圖像
             rgb_image_range: 最後有與深度圖重疊到的範圍
                (xmin, xmax, ymin, ymax)，這裡使用的是左閉右閉
             deep_image_range: 最後有與彩色圖重疊到的範圍
                (xmin, xmax, ymin, ymax)，這裡使用的是左閉右閉
             horizon_flip: 是否需要對深度圖像進行水平翻轉
             vertical_flip: 是否需要對深度圖像進行上下翻轉
             deep_image_resize: 將擷取下來的深度圖像進行resize使得畫面中物體大小與彩色圖大小相同，shape (width, height)
             rgb_image_resize: 將擷取下來的彩色圖像進行resize使得畫面中物體大小與深度圖像大小相同，shape (width, height)
             deep_image_match_rgb: 在最後的時候讓深度圖像的大小透過均勻pad使得大小與彩色圖相同
             rgb_image_match_deep: 在最後的時候讓彩色圖像的大小透過均勻pad使得大小與深度圖相同
        """
        assert isinstance(rgb_image_range, (tuple, list)) and len(rgb_image_range) == 4, '彩色圖給定的範圍規範有錯誤'
        assert isinstance(deep_image_range, (tuple, list)) and len(deep_image_range) == 4, '深度圖給定的範圍規範有錯誤'
        rgb_image_height, rgb_image_width = rgb_image.shape[:2]
        deep_image_height, deep_image_width = deep_image.shape[:2]
        assert 0 <= rgb_image_range[0] < rgb_image_range[1] <= rgb_image_width and \
               0 <= rgb_image_range[2] < rgb_image_range[3] <= rgb_image_height, '裁切範圍有錯誤'
        assert 0 <= deep_image_range[0] < deep_image_range[1] <= deep_image_width and \
               0 <= deep_image_range[2] < deep_image_range[3] <= deep_image_height, '裁切範圍有錯誤'
        rgb_image_range = [int(info) for info in rgb_image_range]
        deep_image_range = [int(info) for info in deep_image_range]
        assert isinstance(horizon_flip, bool) and isinstance(vertical_flip, bool), '是否需要翻轉只能是bool型態'
        if horizon_flip:
            deep_image = cv2.flip(deep_image, 1)
        if vertical_flip:
            deep_image = cv2.flip(deep_image, 0)
        rgb_image_crop = rgb_image[rgb_image_range[2]: rgb_image_range[3],
                         rgb_image_range[0]: rgb_image_range[1]]
        deep_image_crop = deep_image[deep_image_range[2]: deep_image_range[3],
                          deep_image_range[0]: deep_image_range[1]]
        if deep_image_resize is not None:
            # deep_image_resize = (width, height)
            deep_image_crop = cv2.resize(deep_image_crop, deep_image_resize)
        if rgb_image_resize is not None:
            # rgb_image_resize = (width, height)
            rgb_image_crop = cv2.resize(rgb_image_crop, rgb_image_resize)
        rgb_crop_height, rgb_crop_width = rgb_image_crop.shape[:2]
        deep_crop_height, deep_crop_width = deep_image_crop.shape[:2]
        assert not (deep_image_match_rgb and rgb_image_match_deep), '不可以同時match到對方，這樣沒有基準點'
        if deep_image_match_rgb:
            assert (rgb_crop_height - deep_crop_height) >= 0 and (rgb_crop_width - deep_crop_width) >= 0
            x_offset = (rgb_crop_width - deep_crop_width) // 2
            y_offset = (rgb_crop_height - deep_crop_height) // 2
            # 將結果貼到新的圖上面
            deep_image_picture = np.zeros((rgb_crop_height, rgb_crop_width), dtype=np.uint8)
            deep_image_picture[y_offset:y_offset + deep_crop_height, x_offset: x_offset + deep_crop_width] = \
                deep_image_crop
            # 更新新的深度圖上去
            deep_image_crop = deep_image_picture
        if rgb_image_match_deep:
            assert (deep_crop_height - rgb_crop_height) >= 0 and (deep_crop_width - rgb_crop_width) >= 0
            x_offset = (deep_crop_width - rgb_crop_width) // 2
            y_offset = (deep_crop_height - rgb_crop_height) // 2
            # 將結果貼到新的圖上面
            rgb_image_picture = np.zeros((deep_crop_height, deep_crop_width, 3), dtype=np.uint8)
            rgb_image_picture[y_offset:y_offset + rgb_crop_height, x_offset:x_offset + rgb_crop_width] = rgb_image_crop
            # 更新新的彩色圖像
            rgb_image_crop = rgb_image_picture
        rgb_crop_height, rgb_crop_width = rgb_image_crop.shape[:2]
        deep_crop_height, deep_crop_width = deep_image_crop.shape[:2]
        assert rgb_crop_height == deep_crop_height, '彩色圖與深度圖需要有相同的高度'
        assert rgb_crop_width == deep_crop_width, '彩色圖與深度圖需要有相同的寬度'
        return deep_image_crop, rgb_image_crop


def test():
    import time
    deep_match_rgb_cfg = {
        'type': 'custom_match', 'rgb_image_range': (10, 640, 0, 460), 'deep_image_range': (0, 630, 20, 480),
        'horizon_flip': True, 'rgb_image_resize': (567, 414), 'rgb_image_match_deep': True
    }
    module = ReadPictureFromDeepCamera(
        rgb_camera=0, deep_match_rgb_cfg=deep_match_rgb_cfg, min_deep_value=300, max_deep_value=1000)
    pTime = 0
    while True:
        picture_info = module(api_call='get_single_picture')
        rgb_image, image_type, deep_image, deep_draw = picture_info
        mix_image = rgb_image * 0.5 + deep_draw * 0.5
        mix_image = mix_image.astype(np.uint8)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(rgb_image, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('rgb_image', rgb_image)
        cv2.imshow('deep_view', deep_draw)
        cv2.imshow('mix_view', mix_image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    # 測試通過
    print('Testing Read picture from deep camera')
    test()
