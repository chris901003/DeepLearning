import argparse
import os
import pyrealsense2 as rs
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-folder-path', type=str, default='video_data')
    parser.add_argument('--rgb-save-name', type=str, default='rgb.mp4')
    parser.add_argument('--depth-save-name', type=str, default='depth.mp4')
    parser.add_argument('--depth-color-save-name', type=str, default='depth_view.mp4')
    # 這裡給的順序是寬高
    parser.add_argument('--rgb-video-resolution', type=int, default=[1920, 1080])
    # 這裡給的順序是寬高，最終保存的大小會是以RGB的分辨率為主
    parser.add_argument('--depth-video-resolution', type=int, default=[1920, 960])
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--depth-view-range', type=int, default=[500, 1500])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    save_folder_path = args.save_folder_path
    rgb_save_name = args.rgb_save_name
    depth_save_name = args.depth_save_name


if __name__ == '__main__':
    print('保存影像以及深度資料')
    main()
