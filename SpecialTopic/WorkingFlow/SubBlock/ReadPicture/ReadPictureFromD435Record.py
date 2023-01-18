import numpy as np
import cv2
import json
import os
from matplotlib import pyplot as plt


class ReadPictureFromD435Record:
    def __init__(self, rgbd_record_config_path, deep_color_range=(300, 1000), fps=30):
        """
        Args:
            rgbd_record_config_path: 列出彩色以及深度保存資料的設定檔
            deep_color_range: 深度顯示範圍
            fps: frame
        """
        config_info = self.parse_json_file(rgbd_record_config_path)
        self.rgb_record_path = config_info.get('rgb_path')
        self.deep_record_path = config_info.get('deep_path')
        self.fps = fps
        self.deep_color_palette = plt.cm.get_cmap('jet_r')(range(255))
        self.deep_color_range = deep_color_range
        self.num_videos = len(self.rgb_record_path)
        self.current_frame_index = 0
        self.current_video_index = 0
        self.current_video = cv2.VideoCapture(self.rgb_record_path[0])
        self.current_deep = np.load(self.deep_record_path[0])
        self.isEnd = False
        self.support_api = {
            'get_single_picture': self.get_single_picture
        }
        self.logger = None

    def __call__(self, api_call, input=None):
        func = self.support_api.get(api_call, None)
        assert func is not None, f'指定的函數{api_call}不在服務內'
        result = func()
        return result

    def get_single_picture(self):
        """ 獲取單張圖像，如果沒有圖像可以獲取就會回傳一個全黑的畫面
        """
        if self.isEnd:
            return self.get_mock_image()
        rgb_ret, rgb_img = self.current_video.read()
        if not rgb_ret:
            raise RuntimeError('讀取影像資料錯誤')
        deep_info = self.current_deep[:, :, self.current_frame_index]
        deep_color_info = self.color_deep_info(deep_info=deep_info)
        self.update_new_frame()
        return rgb_img, 'ndarray', deep_info, deep_color_info

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

    def update_new_frame(self):
        """ 更新下張圖的位置，如果當前片段已經讀取完畢就會自動載入下一段
        """
        self.current_frame_index += 1
        if self.current_frame_index == self.current_deep.shape[2]:
            self.current_frame_index = 0
            self.current_video_index += 1
            if self.current_video_index == self.num_videos:
                self.isEnd = True
            else:
                self.current_video = cv2.VideoCapture(self.rgb_record_path[self.current_video_index])
                self.current_deep = np.load(self.deep_record_path[self.current_video_index])

    @staticmethod
    def parse_json_file(json_file_path):
        """ 讀取影片路徑資料
        """
        with open(json_file_path, 'r') as f:
            video_info = json.load(f)
        rgb_video_path_list = list()
        deep_info_path_list = list()
        rgb_video_folder_path = video_info.get('rgb_path')
        deep_info_folder_path = video_info.get('deep_path')
        for file_name in os.listdir(rgb_video_folder_path):
            if '_rgb.avi' in file_name and os.path.splitext(file_name)[1] == '.avi':
                rgb_video_path = os.path.join(rgb_video_folder_path, file_name)
                rgb_video_path_list.append(rgb_video_path)
        for file_name in os.listdir(deep_info_folder_path):
            if '_depth.npy' in file_name and os.path.splitext(file_name)[1] == '.npy':
                depth_info_path = os.path.join(deep_info_folder_path, file_name)
                deep_info_path_list.append(depth_info_path)
        assert len(rgb_video_path_list) == len(deep_info_path_list), '檔案中的彩色影片與深度資訊數量不對襯'
        result = dict(rgb_path=rgb_video_path_list, deep_path=deep_info_path_list)
        return result

    @staticmethod
    def get_mock_image():
        mock_rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_deep_image = np.zeros((480, 640), dtype=np.uint16)
        mock_deep_color_image = np.zeros((480, 640, 3), dtype=np.uint8)
        return mock_rgb_image, 'ndarray', mock_deep_image, mock_deep_color_image


def test():
    rgbd_record_config_path = r'C:\DeepLearning\SpecialTopic\WorkingFlow\prepare\read_picture\rgbd_record_config.json'
    model = ReadPictureFromD435Record(rgbd_record_config_path, fps=30)
    while True:
        rgb_image, image_type, deep_image, deep_color_image = model(api_call='get_single_picture')
        if np.min(rgb_image) == 0 and np.max(rgb_image) == 0:
            break
        cv2.imshow('RGB', rgb_image)
        cv2.imshow('Deep', deep_color_image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    print('Testing Read Picture From D435 Record')
    test()
