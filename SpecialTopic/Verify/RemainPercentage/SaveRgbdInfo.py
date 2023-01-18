import pyrealsense2 as rs
import numpy as np
import cv2
import time
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-name', '-f', type=str, default='test')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    file_name = args.file_name
    image_height = args.height
    image_width = args.width
    fps = args.fps
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, fps)
    # 對齊深度與彩色圖實例化對象
    align_to_color = rs.align(rs.stream.color)

    current_phase = 0

    color_path = os.path.join('C:/Dataset/rgbd_video/', file_name + str(current_phase) + '_rgb.avi')
    depth_path = os.path.join('C:/Dataset/rgbd_video/', file_name + str(current_phase) + '_depth')
    color_writer = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (image_width, image_height), 1)

    pipeline.start(config)
    is_depth = 0
    depth_image = np.zeros((image_height, image_width, 1), dtype='uint16')
    pTime = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            # 將當前frame的彩色與深度圖對齊
            frames = align_to_color.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # convert images to numpy arrays
            if is_depth == 0:
                depth_image[:, :, 0] = np.asanyarray(depth_frame.get_data())
                is_depth = 1
            else:
                depth_image = np.append(depth_image, np.asanyarray(depth_frame.get_data()).reshape(
                    image_height, image_width, 1), axis=2)
            color_image = np.asanyarray(color_frame.get_data())
            color_writer.write(color_image)

            if depth_image.shape[2] == 900:
                np.save(depth_path, depth_image)
                color_writer.release()
                depth_image = np.zeros((image_height, image_width, 1), dtype='uint16')
                is_depth = 0
                current_phase += 1
                color_path = os.path.join('C:/Dataset/rgbd_video/', file_name + str(current_phase) + '_rgb.avi')
                depth_path = os.path.join('C:/Dataset/rgbd_video/', file_name + str(current_phase) + '_depth')
                color_writer = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), fps,
                                               (image_width, image_height), 1)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(color_image, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('Stream', color_image)
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        np.save(depth_path, depth_image)
        color_writer.release()
        pipeline.stop()


if __name__ == "__main__":
    main()
