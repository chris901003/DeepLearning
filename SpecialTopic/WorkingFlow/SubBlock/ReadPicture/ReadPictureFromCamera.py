import cv2


class ReadPictureFromCamera:
    def __init__(self, camera_id=0):
        assert isinstance(camera_id, int), '指定攝影機需要是正整數型態'
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        self.support_api = {'get_single_picture': self.get_single_picture}
        self.logger = None

    def __call__(self, api_call, input=None):
        func = self.support_api.get(api_call, None)
        assert func is not None, f'Read Picture From Camera當中沒有{api_call}函數可以使用'
        if input is not None:
            results = func(**input)
        else:
            results = func()
        return results

    def get_single_picture(self):
        """ 在當前指定攝影機上獲取一張圖像資料
        """
        self.logger['logger'].debug('get single picture')
        ret, image = self.cap.read()
        if not ret:
            self.logger['logger'].critical('無法獲取圖像資訊，程式即將終止')
        assert ret, '無法獲取攝影機圖像資訊'
        return image, 'ndarray'

    def change_camera(self, new_camera_id):
        """ 更換攝影機
        Args:
            new_camera_id: 新攝影機ID
        """
        self.logger['logger'].info(f'更新使用攝影機為{new_camera_id}')
        self.camera_id = new_camera_id
        self.cap.release()
        self.cap = cv2.VideoCapture(new_camera_id)

    def __repr__(self):
        print('Read image from camera')
        print(f'Current using {self.camera_id} to capture image')


def test():
    import logging
    logger = logging.getLogger('test')
    module = ReadPictureFromCamera(camera_id=0)
    module.logger = dict(logger=logger, sub_log=None)
    while True:
        img = module('get_single_picture')[0]
        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    print('Test Read picture from camera')
    test()
