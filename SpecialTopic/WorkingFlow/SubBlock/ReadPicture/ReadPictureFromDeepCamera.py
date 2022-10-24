from openni import openni2
import numpy as np
import matplotlib.pyplot as plt
from SpecialTopic.ST.utils import get_cls_from_dict
import cv2


class ReadPictureFromDeepCamera:
    def __init__(self, color_palette_type='jet_r', color_palette_range=255, deep_image_height=480, deep_image_width=640,
                 rgb_camera='Default', rgb_image_height=480, rgb_image_width=640, min_deep_value=300,
                 max_deep_value=6000, deep_match_rgb_cfg='Default'):
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
                'type': 'BruteForceResize'
            }
        support_deep_match_rgb = {
            'BruteForceResize': self.BruteForceResize
        }
        self.deep_match_rgb_func = get_cls_from_dict(support_deep_match_rgb, deep_match_rgb_cfg)
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

    def __call__(self, api_call, input=None):
        func = self.support_api.get(api_call, None)
        assert func is not None, f'Read picture from deep camera中沒有{api_call}可以使用'
        if input is not None:
            results = func(**input)
        else:
            results = func()
        return results

    def get_single_picture(self):
        deep_frame = self.depth_stream.read_frame()
        deep_frame_data = np.array(deep_frame.get_buffer_as_triplet())
        deep_frame_data = deep_frame_data.reshape([self.deep_image_height, self.deep_image_width, 2])
        dpt1 = np.asarray(deep_frame_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(deep_frame_data[:, :, 1], dtype='float32')
        dpt2 *= 255
        # 最終深度圖資料
        dpt = dpt1 + dpt2
        if isinstance(self.color_stream, cv2.VideoCapture):
            ret, rgb_image = self.color_stream.read()
            if not ret:
                raise RuntimeError('無法從攝影機獲取圖像資料')
        else:
            cframe = self.color_stream.read_frame()
            cframe_data = np.array(cframe.get_buffer_as_triplet())
            cframe_data = cframe_data.reshape([self.rgb_image_height, self.rgb_image_width, 3])
            rgb_image = cv2.cvtColor(cv2.COLOR_RGB2BGR, cframe_data)
        deep_image = self.deep_match_rgb_func(dpt, rgb_image)
        deep_draw = self.draw_deep_image(deep_image)
        return rgb_image, 'ndarray', deep_image, deep_draw

    def draw_deep_image(self, deep_image):
        dpt_clip = np.clip(deep_image, self.min_deep_value, self.max_deep_value)
        dpt_clip = (dpt_clip - self.min_deep_value) / (self.max_deep_value - self.min_deep_value) * \
                   (self.color_palette_range - 2)
        dpt_clip = dpt_clip.astype(int)
        color_map = np.zeros((self.rgb_image_height, self.rgb_image_width, 3))
        for i in range(0, self.color_palette_range):
            color_map[dpt_clip == i] = self.color_palette[i][:3]
        return deep_image

    def BruteForceResize(self, deep_image, interpolate_mode='INTER_LINEAR'):
        support = {
            'INTER_NEAREST': cv2.INTER_NEAREST,
            'INTER_LINEAR': cv2.INTER_LINEAR,
            'INTER_AREA': cv2.INTER_AREA,
            'INTER_CUBIC': cv2.INTER_CUBIC,
            'INTER_LANCZOS4': cv2.INTER_LANCZOS4
        }
        interpolate_mode = support.get(interpolate_mode)
        resize_deep_image = cv2.resize(deep_image, (self.rgb_image_width, self.rgb_image_height),
                                       interpolation=interpolate_mode)
        return resize_deep_image
