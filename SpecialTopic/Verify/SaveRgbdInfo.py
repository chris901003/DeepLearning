import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import mss
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    # 根目錄位置
    parser.add_argument('--root-folder-path', type=str, default='./VideoSave')
    # 資料夾名稱
    parser.add_argument('--folder-name', type=str, default='Donburi22')
    # 影片寬度(通常不用調整)
    parser.add_argument('--width', type=int, default=640)
    # 影片高度(通常不用調整)
    parser.add_argument('--height', type=int, default=480)
    # fps值(通常不用調整)
    parser.add_argument('--fps', type=int, default=30)
    # 可視化深度資訊的最小深度值(除非深度有問題否則基本不用調整)
    parser.add_argument('--depth-min-height', default=None)
    # 可視化深度資訊的最大深度值(除非深度有問題否則基本不用調整)
    parser.add_argument('--depth-max-height', default=None)

    # 碼表在螢幕上的哪個位置
    parser.add_argument('--stopwatch-xmin', type=int, default=122)
    parser.add_argument('--stopwatch-ymin', type=int, default=172)
    parser.add_argument('--stopwatch-xmax', type=int, default=396)
    parser.add_argument('--stopwatch-ymax', type=int, default=250)

    # 重量在螢幕上的位置
    parser.add_argument('--weight-xmin', type=int, default=1467)
    parser.add_argument('--weight-ymin', type=int, default=270)
    parser.add_argument('--weight-xmax', type=int, default=1801)
    parser.add_argument('--weight-ymax', type=int, default=444)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 獲取保存資料路徑，並且創建
    root_folder_path = args.root_folder_path
    folder_name = args.folder_name
    if not os.path.exists(root_folder_path):
        os.mkdir(root_folder_path)
    folder_path = os.path.join(root_folder_path, folder_name)
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # 初始化深度攝影機
    width = args.width
    height = args.height
    fps = args.fps
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    align_to_color = rs.align(rs.stream.color)

    # 其他零碎的東西
    stopwatch_xmin = args.stopwatch_xmin
    stopwatch_ymin = args.stopwatch_ymin
    stopwatch_width = args.stopwatch_xmax - stopwatch_xmin
    stopwatch_height = args.stopwatch_ymax - stopwatch_ymin
    stopwatch_monitor = {
        # top=ymin, left=xmin
        'top': stopwatch_ymin, 'left': stopwatch_xmin,
        'width': stopwatch_width, 'height': stopwatch_height
    }
    weight_xmin = args.weight_xmin
    weight_ymin = args.weight_ymin
    weight_width = args.weight_xmax - weight_xmin
    weight_height = args.weight_ymax - weight_ymin
    weight_monitor = {
        # top=ymin, left=xmin
        'top': weight_ymin, 'left': weight_xmin,
        'width': weight_width, 'height': weight_height
    }
    depth_min_height = args.depth_min_height
    depth_max_height = args.depth_max_height
    color_palette = plt.cm.get_cmap('jet_r')(range(255))
    pipeline.start(config)
    pTime = time.time()

    # 影片保存路徑，以及彩色圖保存實例化對象
    current_phase = 0
    color_path = os.path.join(folder_path, 'RgbView.avi')
    color_writer = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height), 1)
    stopwatch_path = os.path.join(folder_path, 'Stopwatch.mp4')
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    stopwatch_writer = cv2.VideoWriter(stopwatch_path, fourcc, fps, (stopwatch_width, stopwatch_height), 1)
    weight_path = os.path.join(folder_path, 'Weight.mp4')
    weight_writer = cv2.VideoWriter(weight_path, fourcc, fps, (weight_width, weight_height), 1)
    depth_path = os.path.join(folder_path, f'Depth_{current_phase}')

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align_to_color.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            sct = mss.mss()
            stopwatch_image = np.array(sct.grab(stopwatch_monitor))[..., :3]
            weight_image = np.array(sct.grab(weight_monitor))[..., :3]
            if not depth_frame or not color_frame:
                continue

            # 深度相關資料
            depth_info = np.asanyarray(depth_frame.get_data())
            np.save(depth_path, depth_info)
            current_phase += 1
            depth_path = os.path.join(folder_path, f'Depth_{current_phase}')
            depth_image = deep_color_image(depth_info, color_palette, depth_min_height, depth_max_height)

            # 相機彩色圖像
            color_image = np.asanyarray(color_frame.get_data())
            color_writer.write(color_image)

            # 螢幕碼表畫面
            stopwatch_writer.write(stopwatch_image)

            # 螢幕重量畫面
            weight_writer.write(weight_image)

            # 當前畫面顯示
            cTime = time.time()
            current_fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(color_image, f"FPS : {int(current_fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('Camera RGB', color_image)
            cv2.imshow('Camera Depth', depth_image)

            cv2.namedWindow("Stopwatch", 0)
            cv2.resizeWindow("Stopwatch", stopwatch_width, stopwatch_height)
            cv2.imshow('Stopwatch', stopwatch_image)

            cv2.namedWindow("Weight", 0)
            cv2.resizeWindow("Weight", weight_width, weight_height)
            cv2.imshow('Weight', weight_image)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        color_writer.release()
        stopwatch_writer.release()
        weight_writer.release()
        pipeline.stop()


def deep_color_image(deep_info, color_palette, min_value=None, max_value=None):
    if min_value is None:
        min_value = np.min(deep_info)
    if max_value is None:
        max_value = np.max(deep_info)
    dpt_clip = np.clip(deep_info, min_value, max_value)
    dpt_clip = (dpt_clip - min_value) / (max_value - min_value) * 253
    dpt_clip = dpt_clip.astype(int)
    color_map = color_palette[dpt_clip, :].reshape([dpt_clip.shape[0], dpt_clip.shape[1], 4])[..., :3]
    color_map = (color_map * 255).astype(np.uint8)
    return color_map


def get_mouse_place():
    """ 獲取當前滑鼠座標，可以快速獲取要擷取螢幕的位置
    """
    import pyautogui as pag
    screenWidth, screenHeight = pag.size()
    print(f'螢幕大小: 寬: {screenWidth}, 高: {screenHeight}')
    cv2.namedWindow('Space')
    try:
        while True:
            print('按下任意建獲取當前滑鼠座標，按下ESC退出')
            keycode = cv2.waitKey(0)
            if keycode == 27:
                break
            x, y = pag.position()
            print(f'滑鼠座標: x: {x}, y: {y}')
    except KeyboardInterrupt:
        print('Finish')


if __name__ == '__main__':
    main()
    # get_mouse_place()
