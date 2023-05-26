import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
from matplotlib import pyplot as plt


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-frame', type=int, default=120)
    parser.add_argument('--save-path', type=str, default='./initial_depth.npy')
    parser.add_argument('--visualize-min-depth', type=int, default=600)
    parser.add_argument('--visualize-max-depth', type=int, default=900)
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    total_frame = args.total_frame
    save_path = args.save_path
    visualize_min_depth = args.visualize_min_depth
    visualize_max_depth = args.visualize_max_depth
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align_to_color = rs.align(rs.stream.color)

    save_depth_info = np.zeros([480, 640])
    cnt = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align_to_color.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            save_depth_info += depth_image
            cnt += 1
            current_depth = deep_color_image(depth_image, visualize_min_depth, visualize_max_depth)
            save_depth = deep_color_image(save_depth_info / cnt, visualize_min_depth, visualize_max_depth)

            cv2.imshow('RGB', color_image)
            cv2.imshow("Current Depth", current_depth)
            cv2.imshow("Save Depth", save_depth)
            if cv2.waitKey(1) == ord('q') or cnt == total_frame:
                break
    finally:
        pipeline.stop()
    np.save(save_path, save_depth_info / cnt)


def deep_color_image(deep_info, min_value=None, max_value=None):
    color_palette = plt.cm.get_cmap('jet_r')(range(255))
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


if __name__ == "__main__":
    main()
