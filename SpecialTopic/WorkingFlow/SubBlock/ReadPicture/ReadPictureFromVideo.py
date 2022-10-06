import cv2


class ReadPictureFromVideo:
    def __init__(self, video_path=None):
        assert video_path is not None, '需提供影片路徑以及api名稱'
        assert isinstance(video_path, str), '影片路徑需要是字串格式'
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.support_api = {'get_single_picture': self.get_single_picture, 'change_camera_id': self.change_camera_id}

    def __call__(self, api_call, input=None):
        func = self.support_api.get(api_call, None)
        assert func is not None, f'ReadPictureFromVideo當中沒有{api_call}功能'
        if input is None:
            results = func()
        else:
            results = func(**input)
        return results

    def get_single_picture(self):
        ret, image = self.cap.read()
        assert ret, '無法圖取圖像，get_single_picture發生錯誤'
        return image, 'ndarray'

    def change_camera_id(self, new_video_path, nothing='Change video'):
        self.cap = cv2.VideoCapture(new_video_path)
        self.video_path = new_video_path
        print(nothing)

    def __repr__(self):
        print('Read image from video')
        print(f'Current loading {self.video_path} video')


def test():
    module = ReadPictureFromVideo('./test.pm4')
    while True:
        img = module('get_single_picture')[0]
        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    print('Test Read Picture From Video class')
    test()
