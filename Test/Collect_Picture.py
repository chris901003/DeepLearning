import cv2
from matplotlib import pyplot as plt
import os


def main():
    # save_path = 要保存圖像的路徑
    save_path = '/Users/huanghongyan/Documents/DeepLearning/Test/PoseData/1'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # 檔案名稱會根據目前資料夾當中圖像數量決定
    cnt = len(os.listdir(save_path))
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if ret:
            k = cv2.waitKey(1)
            if k == ord('g'):
                # 按下g會進行截圖
                cv2.imwrite(f'{save_path}/{cnt}.jpg', img)
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 使用plt將截圖圖像暫時展示
                plt.imshow(imgRGB)
                plt.show(block=False)
                # 過1秒後會自動消失
                plt.pause(1)
                plt.close()
                cnt = cnt + 1
            elif k == ord('q'):
                # 按下q跳出
                break
            # 將當前畫面展示
            cv2.imshow('img', img)


if __name__ == '__main__':
    main()
