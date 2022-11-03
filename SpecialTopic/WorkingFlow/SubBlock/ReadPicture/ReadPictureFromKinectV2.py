from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import ctypes
import numpy as np
import matplotlib.pyplot as plt


class ReadPictureFromKinectV2:
    def __init__(self, deep_color_palette='jet_r', deep_color_palette_range=255, deep_color_range=None,
                 rgb_image_height=1080, rgb_image_width=1920, deep_image_height=1080, deep_image_width=1920):
        """ 專門從KinectV2獲取彩色圖像以及深度資料，同時這裡只支援彩色圖像與深度圖像都來自Kinect輸出
        Args:
            deep_color_palette: 對深度圖像可視化時，對於不同深度上色的色盤
            deep_color_palette_range: 生成色盤時需要指定色盤的顏色數量，建議不要超過255或是先自行測試，指定色盤的色彩量
            deep_color_range: 限制深度範圍，低於限制值得用最小表示，高於最大值的直接用最大表示
            rgb_image_height: 彩色圖像高度
            rgb_image_width: 彩色圖像寬度
            deep_image_height: 深度圖像的高度
            deep_image_width: 深度圖像的寬度
        """
        assert 0 < deep_color_palette_range, 'deep color palette range須至少大於0'
        if deep_color_range is None:
            deep_color_range = [500, 1500]
        assert deep_color_range[0] < deep_color_range[1], '取樣深度至少需要有1的差距'
        if rgb_image_width != deep_image_width or rgb_image_height != deep_image_height:
            print('如果想要直接對相同地方標註就需要兩者大小相同')
        self.deep_color_palette = plt.cm.get_cmap(deep_color_palette)(range(deep_color_palette_range))
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth |
                                                      PyKinectV2.FrameSourceTypes_Color)
        self.deep_color_range = deep_color_range
        self.rgb_image_height, self.rgb_image_width = rgb_image_height, rgb_image_width
        self.deep_image_height, self.deep_image_width = deep_image_height, deep_image_width
        self.support_api = {
            'get_single_picture': self.get_single_picture
        }
        # 處理第一張圖像會是全黑的情況
        self.get_single_picture()
        self.logger = None

    def depth_2_color_space(self, depth_space_point=_DepthSpacePoint, show=False, return_aligned_image=True):
        """ 主要是將深度圖像資料縮放到彩色圖像當中，且讓兩者可以匹配上
        Args:
            depth_space_point: 基本上不用改變
            show: 是否需要直接展示，在使用時基本關閉
            return_aligned_image: 是否需要將經過匹配後的深度圖回傳，基本上使用時直接開啟
        """
        color2depth_points_type = depth_space_point * np.int(1920 * 1080)
        color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(depth_space_point))
        self.kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), self.kinect._depth_frame_data,
                                                      ctypes.c_uint(1920 * 1080), color2depth_points)
        depthXYs = np.copy(np.ctypeslib.as_array(color2depth_points, shape=(self.kinect.color_frame_desc.Height *
                                                                            self.kinect.color_frame_desc.Width,)))
        depthXYs = depthXYs.view(np.float32).reshape(depthXYs.shape + (-1,))
        depthXYs += 0.5
        depthXYs = depthXYs.reshape(self.kinect.color_frame_desc.Height,
                                    self.kinect.color_frame_desc.Width, 2).astype(np.int)
        depthXs = np.clip(depthXYs[:, :, 0], 0, self.kinect.depth_frame_desc.Width - 1)
        depthYs = np.clip(depthXYs[:, :, 1], 0, self.kinect.depth_frame_desc.Height - 1)
        depth_frame = self.kinect.get_last_depth_frame()
        depth_img = depth_frame.reshape((self.kinect.depth_frame_desc.Height, self.kinect.depth_frame_desc.Width,
                                         1)).astype(np.uint16)
        align_depth_img = np.zeros((1080, 1920, 1), dtype=np.uint16)
        align_depth_img[:, :] = depth_img[depthYs, depthXs, :]
        if show:
            cv2.imshow('Aligned Image', cv2.resize(cv2.flip(align_depth_img, 1), (1920 // 2, 1080 // 2)))
            cv2.waitKey(1)
        if return_aligned_image:
            return align_depth_img
        return depthXs, depthYs

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
        assert func is not None, f'Read picture from kinect v2沒有提供 {api_call} 函數'
        if input is not None:
            results = func(**input)
        else:
            results = func()
        return results

    def get_single_picture(self):
        """ 獲取一次的彩色圖像與深度圖像資料
        Returns:
            rgb_image: 彩色圖像資料
            deep_image: 深度資料，單位為mm
            deep_color: 根據深度資料畫出顏色
        """
        # 因為每次不一定可以拿到資料，所以放while True
        while True:
            if self.kinect.has_new_depth_frame():
                color_frame = self.kinect.get_last_color_frame()
                colorImage = color_frame.reshape((self.kinect.color_frame_desc.Height,
                                                  self.kinect.color_frame_desc.Width, 4)).astype(np.uint8)
                colorImage = cv2.flip(colorImage, 1)
                img = self.depth_2_color_space()
                rgb_image = cv2.resize(colorImage, (self.rgb_image_width, self.rgb_image_height))[..., :3]
                rgb_image = rgb_image.copy()
                deep_image = cv2.resize(cv2.flip(img, 1), (self.deep_image_width, self.deep_image_height))
                deep_color = self.color_deep_info(deep_image)
                return rgb_image, 'ndarray', deep_image, deep_color


def test():
    import time
    import logging
    module = ReadPictureFromKinectV2(rgb_image_height=540, rgb_image_width=960, deep_image_height=540,
                                     deep_image_width=960, deep_color_range=[500, 1000])
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
        mix_image = rgb_image * 0.5 + deep_color * 0.5
        mix_image = mix_image.astype(np.uint8)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(rgb_image, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('rgb_image', rgb_image)
        cv2.imshow('deep_view', deep_color)
        cv2.imshow('mix_view', mix_image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    print('Testing Read picture from kinect v2')
    test()
