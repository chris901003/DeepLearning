import os

# 這部分只是用來解決一些模組上面的衝突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from shells.shell import Shell
import imutils
import cv2

# 輸入的影像
VIDEO_PATH = './video/traffic.mp4'
# VIDEO_PATH = './video/pedestrian.mp4'
# 輸出的影像
RESULT_PATH = './out/result.mp4'

# DeepSort配置文件位置
DEEPSORT_CONFIG_PATH = "./deep_sort/configs/deep_sort.yaml"
# Yolo_v5預訓練權重位置
YOLOV5_WEIGHT_PATH = './weights/yolov5m.pt'


def main():
    # 已看過
    # 構建Shell實例對象，傳入DeepSort配置文件以及Yolo_v5預訓練權重位置
    det = Shell(DEEPSORT_CONFIG_PATH, YOLOV5_WEIGHT_PATH)

    videoWriter = None
    # 開始讀取影片，這裡只要給的是檔案位置就會自動變成讀取影片或是照片
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(5))
    t = int(1000 / fps)
    while True:
        # 開始讀取直到影片結束
        # frame = numpy shape [height, width, channel] (1080, 1920, 3)
        _, frame = cap.read()
        if not _:
            break

        result = det.update(frame)
        result = result['frame']
        result = imutils.resize(result, height=500)
        # if videoWriter is None:
        #     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # opencv3.0
        #     videoWriter = cv2.VideoWriter(RESULT_PATH, fourcc, fps, (result.shape[1], result.shape[0]))
        # videoWriter.write(result)

        cv2.imshow("frame", result)
        key = cv2.waitKey(t)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    # videoWriter.release()
    cap.release()


if __name__ == '__main__':
    # 已看過
    main()
