import pyrealsense2 as rs
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt


class ReadPictureFromD435:
    def __init__(self, deep_color_palette='jet_r', deep_color_palette_range=255, deep_color_range=None,
                 rgb_height=480, rgb_width=640, deep_height=480, deep_width=640, fps=30):
        """ 專門從D435獲取RGB圖像以及深度圖像，同時會將兩個畫面進行對齊輸出
        Args:
            deep_color_palette: 對深度圖進行上色獲取可視化的結果
            deep_color_palette_range: 塗色板相關資料
            deep_color_range: 上色的深度範圍，大於或是小於的會以上界與下界取代
            rgb_height: 最大1080
            rgb_width: 最大1920
            deep_height: 最大960
            deep_width: 最大1280
            fps: 最大60，但是不可以將畫質開到最高
        """
        assert 0 < deep_color_palette_range, 'deep color palette range 至少需要大於0'
        assert isinstance(deep_color_range, list), '須給定一個範圍'
        assert deep_color_range[0] <= deep_color_range[1], '給定的顏色範圍有錯誤'
        self.deep_color_palette = plt.cm.get_cmap(deep_color_palette)(range(deep_color_palette_range))
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, deep_width, deep_height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, fps)
        self.pipeline.start(config)
        self.align_to_color = rs.align(rs.stream.color)
        self.deep_color_range = deep_color_range
        self.support_api = {
            'get_single_picture': self.get_single_picture
        }
        self.get_single_picture()
        self.logger = None

    def color_deep_info(self, deep_info):
        """ 生成深度可視化圖像
        Args:
            deep_info: 深度資料，當中每個像素點代表的值為深度，單位mm
        """
        min_deep, max_deep = self.deep_color_range
        dpt_clip = np.clip(deep_info, min_deep, max_deep)
        dpt_clip = (dpt_clip - min_deep) / (max_deep - min_deep) * 253
        dpt_clip = dpt_clip.astype(int)
        deep_color_map = self.deep_color_palette[dpt_clip, :].reshape([dpt_clip.shape[0], dpt_clip.shape[1],
                                                                       4])[..., :3]
        deep_color_map = (deep_color_map * 255).astype(np.uint8)
        return deep_color_map

    def __call__(self, api_call, input=None):
        func = self.support_api.get(api_call, None)
        assert func is not None, f'Read picture from D435沒有提供該api可以使用'
        results = func()
        return results

    def get_single_picture(self):
        pTime = time.time()
        while True:
            frames = self.pipeline.wait_for_frames()
            frames = self.align_to_color.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                if time.time() - pTime > 10:
                    self.logger['logger'].critical('無法從攝影機獲取資料，請確認是否有錯誤')
                    raise RuntimeError('無法獲取攝影機資，請確認是否有連接成功')
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_view = self.color_deep_info(depth_image)
            return color_image, 'ndarray', depth_image, depth_view


def test():
    import time
    import logging
    module = ReadPictureFromD435(deep_color_range=[600, 800],
                                 rgb_height=1080, rgb_width=1920, deep_height=720, deep_width=1280)
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger = dict(logger=logger, sub_log=None)
    module.logger = logger

    pTime = 0
    while True:
        picture_info = module(api_call='get_single_picture')
        rgb_image, image_type, deep_info, deep_color = picture_info
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(rgb_image, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('rgb_image', rgb_image)
        cv2.imshow('deep_view', deep_color)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    print('Testing Read picture from kinect v2')
    test()
