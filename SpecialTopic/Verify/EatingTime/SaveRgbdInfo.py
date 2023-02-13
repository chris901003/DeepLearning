import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import mss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder-name', '-f', type=str, default='./RgbdSave/Test')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--fps', type=int, default=30)

    # 設定螢幕擷取範圍資訊，這裡需要提供[左上角以及高寬資訊]
    parser.add_argument('--screen-xmin', '-x', type=int, default=100)
    parser.add_argument('--screen-ymin', '-y', type=int, default=100)
    parser.add_argument('--screen-height', type=int, default=150)
    parser.add_argument('--screen-width', type=int, default=200)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    folder_name = args.folder_name
    image_height = args.height
    image_width = args.width
    fps = args.fps
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, fps)
    align_to_color = rs.align(rs.stream.color)

    current_phase = 0
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    color_path = os.path.join(folder_name, 'RgbView.avi')
    depth_path = os.path.join(folder_name, f'Depth_{current_phase}')
    color_writer = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (image_width, image_height), 1)

    pipeline.start(config)
    pTime = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align_to_color.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            sct = mss.mss()
            monitor = {'top': args.screen_ymin, 'left': args.screen_xmin,
                       'width': args.screen_width, 'height': args.screen_height}
            screen_image = np.array(sct.grab(monitor))[..., :3]
            if not depth_frame or not color_frame:
                # 圖像資料收集不完全，先跳過此frame
                continue

            # 深度資訊保存方式
            depth_image = np.asanyarray(depth_frame.get_data())
            np.save(depth_path, depth_image)
            current_phase += 1
            depth_path = os.path.join(folder_name, f'Depth_{current_phase}')

            color_image = np.asanyarray(color_frame.get_data())
            color_image[args.screen_ymin: args.screen_ymin + args.screen_height,
            args.screen_xmin: args.screen_xmin + args.screen_width, :3] = screen_image
            color_writer.write(color_image)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(color_image, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('Stream', color_image)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        color_writer.release()
        pipeline.stop()


if __name__ == '__main__':
    main()
