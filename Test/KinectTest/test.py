from openni import openni2
import numpy as np
import cv2


def main():
    # 初始化openni2
    openni2.initialize()  # can also accept the path of the OpenNI redistribution

    # 獲取攝影機設備
    dev = openni2.Device.open_any()
    # 打印出設備資料
    print(dev.get_device_info())

    # 獲取深度流對象
    depth_stream = dev.create_depth_stream()
    # 獲取rgb圖像流
    color_stream = dev.create_color_stream()
    # 開始獲取深度資料
    depth_stream.start()
    # 開始獲取rgb圖像資料
    color_stream.start()

    while True:
        # 獲取深度圖像資料
        frame = depth_stream.read_frame()
        # 這裡獲取出來的資料會是，shape [204800, 3]所以需要進行reshape
        dframe_data = np.array(frame.get_buffer_as_triplet())
        # 將資料進行reshape，最後的shape [480, 640, 2]，這裡的深度資訊佔兩個channel深度
        dframe_data = dframe_data.reshape([480, 640, 2])
        # 將兩層資料拆分開來，這裡會是float32格式
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
        # 第二層資料需要x255才會是正確的值
        dpt2 *= 255
        # 將層資料相加獲取真實深度資訊，這裡的值就會是深度值，單位是mm
        dpt = dpt1 + dpt2
        max_depth = np.max(dpt)
        min_depth = np.min(dpt)
        mid_depth = dpt[240][320]
        print(f'Max Depth: {max_depth}, Min Depth: {min_depth}, Mid Depth: {mid_depth}')
        # 直接顯示，這裡其實應該要進行縮放
        cv2.imshow('dpt', dpt)

        # 獲取rgb圖像
        cframe = color_stream.read_frame()
        # 獲取彩色圖像並且同時進行reshape
        cframe_data = np.array(cframe.get_buffer_as_triplet()).reshape([480, 640, 3])
        # 這裡出來的排序就會是RGB，如果要使用cv2的顯示就會需要轉換通道排序
        R = cframe_data[:, :, 0]
        G = cframe_data[:, :, 1]
        B = cframe_data[:, :, 2]
        cframe_data = np.transpose(np.array([B, G, R]), [1, 2, 0])
        cv2.imshow('color', cframe_data)

        if cv2.waitKey(1) == ord('q'):
            break

    # 關閉設備流
    depth_stream.stop()
    color_stream.stop()
    # 關閉整個設備
    dev.close()


if __name__ == '__main__':
    # 使用該程式碼前需要先進行以下步驟
    # step1:
    # 確認有安裝過設備對應的SDK
    # XBOX 360 Kinect v1: https://www.microsoft.com/en-us/download/details.aspx?id=40278
    # step2:
    # 下載並安裝: https://structure.io/openni
    # step3:
    # 安裝openni對python的支援: pip install openni
    print('Testing Depth Camera')
    main()
