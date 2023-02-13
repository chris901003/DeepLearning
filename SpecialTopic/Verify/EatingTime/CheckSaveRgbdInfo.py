import argparse
import os
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt


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
    parser.add_argument('--target-folder', '-f', type=str, default='./RgbdSave/Test')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    target_folder = args.target_folder
    assert os.path.exists(target_folder), '指定檔案路徑不存在'

    # 指定RGB影片路徑，並且進行讀取
    rgb_path = os.path.join(target_folder, 'RgbView.avi')
    cap = cv2.VideoCapture(rgb_path)

    color_palette = plt.cm.get_cmap('jet_r')(range(255))
    current_frame = 0
    pTime = time.time()

    while True:
        ret, rgb_image = cap.read()
        if not ret:
            break
        depth_path = os.path.join(target_folder, f'Depth_{current_frame}.npy')
        assert os.path.exists(depth_path), f'無法獲取{depth_path}資料'
        depth_info = np.load(depth_path)
        current_frame += 1
        depth_color_image = deep_color_image(depth_info, color_palette)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(rgb_image, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('RGB', rgb_image)
        cv2.imshow('Depth', depth_color_image)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
