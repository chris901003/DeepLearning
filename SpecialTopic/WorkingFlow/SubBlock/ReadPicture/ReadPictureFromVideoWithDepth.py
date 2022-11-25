import cv2
import os


class ReadPictureFromVideoWithDepth:
    def __init__(self, rgb_video_path=None, depth_video_path=None, depth_view_video_path=None):
        assert rgb_video_path is not None, '須提供彩色影片路徑'
        assert depth_video_path is not None, '須提供深度影片路徑'
        assert os.path.exists(rgb_video_path), '提供的彩色影片不存在'
        assert os.path.exists(depth_video_path), '提供的深度影片不存在'
        self.rgb_video_path = rgb_video_path
        self.depth_video_path = depth_video_path
        self.rgb_cap = cv2.VideoCapture(rgb_video_path)
        self.depth_cap = cv2.VideoCapture(depth_video_path)
        if depth_view_video_path is not None and os.path.exists(depth_view_video_path):
            self.depth_view_cap = cv2.VideoCapture(depth_view_video_path)
        else:
            self.depth_view_cap = None
        self.support_api = {
            'get_single_picture': self.get_single_picture
        }
        self.logger = None

    def __call__(self, call_api, input=None):
        func = self.support_api.get(call_api, None)
        assert func is not None, f'ReadPictureFromVideoWithDepth尚未提供{func}函數'
        if input is None:
            results = func()
        else:
            results = func(**input)
        return results

    def get_single_picture(self):
        rgb_ret, rgb_image = self.rgb_cap.read()
        depth_ret, depth_image = self.depth_cap.read()
        if self.depth_view_cap is not None:
            depth_view_ret, depth_view_image = self.depth_view_cap.read()
            assert depth_view_ret, self.logger['logger'].critical('可視化深度圖讀取錯誤')
        else:
            depth_view_image = None
        assert rgb_ret, self.logger['logger'].critical('彩色圖讀取錯誤')
        assert depth_ret, self.logger['logger'].critical('深度資料圖取錯誤')
        return rgb_image, 'ndarray', depth_image, depth_view_image


def test():
    import logging
    logger = logging.getLogger('test')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger = dict(logger=logger, sub_log=None)

    rgb_video_path = r'C:\DeepLearning\some_utils\video_data\rgb.mp4'
    depth_video_path = r'C:\DeepLearning\some_utils\video_data\depth.mp4'
    depth_view_video_path = r'C:\DeepLearning\some_utils\video_data\depth_view.mp4'
    model = ReadPictureFromVideoWithDepth(rgb_video_path=rgb_video_path, depth_video_path=depth_video_path,
                                          depth_view_video_path=depth_view_video_path)
    model.logger = logger
    while True:
        rgb_image, image_type, depth_image, depth_view_image = model(call_api='get_single_picture')
        cv2.imshow('RGB', rgb_image)
        cv2.imshow('Depth', depth_image)
        cv2.imshow('Depth view', depth_view_image)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    print('Test ReadPictureFromVideoWithDepth sub block')
    test()
