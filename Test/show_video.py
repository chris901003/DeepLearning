import pyrealsense2 as rs
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--rgb-video-path', type=str, default=r'C:\Dataset\rgbd_video\test1_rgb.avi')
    parser.add_argument('--deep-record-path', type=str, default=r'C:\Dataset\rgbd_video\test1_depth.npy')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    image_height = args.height
    image_width = args.width
    fps = args.fps
    image_size = image_height * image_width
    rgb_video_path = args.rgb_video_path
    deep_record_path = args.deep_record_path
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, image_width, image_height, rs.format.bgr8, fps)
    align_to_color = rs.align(rs.stream.color)

    # Start streaming
    pipeline.start(config)
    depth_img = np.load(deep_record_path)
    cap = cv2.VideoCapture(rgb_video_path)

    color_palette = plt.cm.get_cmap('jet_r')(range(255))

    try:
        for i in range(depth_img.shape[2]):
            frames = pipeline.wait_for_frames()
            frames = align_to_color.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_camera = np.asanyarray(depth_frame.get_data())
            color_camera = np.asanyarray(color_frame.get_data())

            ret, rgb_img = cap.read()
            if not ret:
                print("rgb length != depth")
                break

            deep_different = np.sum(np.abs(depth_camera.astype(int) -
                                           depth_img[:, :, i].reshape(image_height,
                                                                      image_width).astype(int))) / image_size
            rgb_different = np.sum(np.abs(color_camera.astype(int) - rgb_img.astype(int))) / (image_size * 3)
            record_deep_view_image = deep_color_image(depth_img[:, :, i], color_palette)
            current_deep_view_image = deep_color_image(depth_camera, color_palette)

            cv2.putText(record_deep_view_image, f"Deep average different : {round(deep_different, 5)}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(rgb_img, f"Deep average different : {round(rgb_different, 5)}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Record Deep', record_deep_view_image)
            cv2.imshow('Current Deep', current_deep_view_image)
            cv2.imshow('Record RGB', rgb_img)
            cv2.imshow('Current RGB', color_camera)

            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()
