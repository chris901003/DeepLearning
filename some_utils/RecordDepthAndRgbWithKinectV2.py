import argparse
import os

from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from matplotlib import pyplot as plt
import cv2
import ctypes
import numpy as np
import time


def depth_2_color_space(kinect, depth_space_point, depth_frame_data, show=False, return_aligned_image=False):
    """ 主要是將深度圖像資料縮放到彩色圖像當中，且讓兩者可以匹配上
    """
    color2depth_points_type = depth_space_point * np.int(1920 * 1080)
    color2depth_points = ctypes.cast(color2depth_points_type(), ctypes.POINTER(depth_space_point))
    kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), depth_frame_data, ctypes.c_uint(1920 * 1080),
                                             color2depth_points)
    depthXYs = np.copy(np.ctypeslib.as_array(color2depth_points, shape=(kinect.color_frame_desc.Height *
                                                                        kinect.color_frame_desc.Width,)))
    depthXYs = depthXYs.view(np.float32).reshape(depthXYs.shape + (-1, ))
    depthXYs += 0.5
    depthXYs = depthXYs.reshape(kinect.color_frame_desc.Height, kinect.color_frame_desc.Width, 2).astype(np.int)
    depthXs = np.clip(depthXYs[:, :, 0], 0, kinect.depth_frame_desc.Width - 1)
    depthYs = np.clip(depthXYs[:, :, 1], 0, kinect.depth_frame_desc.Height - 1)
    depth_frame = kinect.get_last_depth_frame()
    depth_img = depth_frame.reshape((kinect.depth_frame_desc.Height, kinect.depth_frame_desc.Width,
                                     1)).astype(np.uint16)
    align_depth_img = np.zeros((1080, 1920, 1), dtype=np.uint16)
    align_depth_img[:, :] = depth_img[depthYs, depthXs, :]
    if show:
        cv2.imshow('Aligned Image', cv2.resize(cv2.flip(align_depth_img, 1), (1920 // 2, 1080 // 2)))
        cv2.waitKey(1)
    if return_aligned_image:
        return align_depth_img
    return depthXs, depthYs


def parse_args():
    parser = argparse.ArgumentParser()
    # 保存資料的資料夾路徑，如果不存在就會自動創建
    parser.add_argument('--save-folder-path', type=str, default='video_data')
    # 彩色圖像保存位置
    parser.add_argument('--rgb-save-name', type=str, default='rgb.mp4')
    # 深度資料保存位置
    parser.add_argument('--depth-save-name', type=str, default='depth.mp4')
    # 深度資料可視化保存位置
    parser.add_argument('--depth-view-name', type=str, default='depth_view.mp4')
    # 影片高寬，建議設定成[960, 540]或是[1920, 1080]剩下的都不推薦
    parser.add_argument('--video-resolution', type=int, default=[960, 540], nargs='+')
    # 設定保存影片的更新率，這裡因為kinectV2的更新率大致上都在10，所以這樣設定，如果使用30會有加速效果
    parser.add_argument('--fps', type=int, default=10)
    # 可視化深度資料的塗色範圍
    parser.add_argument('--depth-view-range', type=int, default=[600, 800], nargs='+')
    args = parser.parse_args()
    return args


def create_video_writer(save_path, save_name, video_fps, video_resolution):
    if save_name is None:
        return None
    save_path = os.path.join(save_path, save_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, video_fps, video_resolution)
    return video_writer


def deep_color_image(deep_info, color_palette, value_range=None):
    assert value_range is not None
    min_value, max_value = value_range
    dpt_clip = np.clip(deep_info, min_value, max_value)
    dpt_clip = (dpt_clip - min_value) / (max_value - min_value) * 253
    dpt_clip = dpt_clip.astype(int)
    color_map = color_palette[dpt_clip, :].reshape([dpt_clip.shape[0], dpt_clip.shape[1], 4])[..., :3]
    color_map = (color_map * 255).astype(np.uint8)
    return color_map


def main():
    args = parse_args()
    save_path = args.save_folder_path
    rgb_save_name = args.rgb_save_name
    depth_save_name = args.depth_save_name
    depth_view_name = args.depth_view_name
    video_resolution = args.video_resolution
    video_fps = args.fps
    depth_view_range = args.depth_view_range
    assert rgb_save_name is not None and depth_save_name is not None, '至少需要提供彩色圖像以及深度資料的保存名稱'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    rgb_video_writer = create_video_writer(save_path, rgb_save_name, video_fps, video_resolution)
    depth_video_writer = create_video_writer(save_path, depth_save_name, video_fps, video_resolution)
    depth_view_writer = create_video_writer(save_path, depth_view_name, video_fps, video_resolution)

    color_palette = plt.cm.get_cmap('jet_r')(range(255))
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
    sTime = time.time()
    while True:
        if kinect.has_new_depth_frame():
            color_frame = kinect.get_last_color_frame()
            colorImage = color_frame.reshape((kinect.color_frame_desc.Height,
                                              kinect.color_frame_desc.Width, 4)).astype(np.uint8)
            colorImage = cv2.flip(colorImage, 1)
            img = depth_2_color_space(kinect, _DepthSpacePoint, kinect._depth_frame_data, show=False,
                                      return_aligned_image=True)
            rgb_image = cv2.resize(colorImage, video_resolution)
            deep_image = cv2.resize(cv2.flip(img, 1), video_resolution)
            if depth_view_range is not None:
                deep_color = deep_color_image(deep_image, color_palette, depth_view_range)
            else:
                deep_color = None
            rgb_image = rgb_image[..., :3].copy()
            deep_image = np.expand_dims(deep_image, -1).repeat(3, axis=-1)
            rgb_video_writer.write(rgb_image)
            depth_video_writer.write(deep_image)
            if depth_view_writer is not None:
                assert deep_color is not None, '如果需要保存可視化後的深度資料，需要提供圖像'
                depth_view_writer.write(deep_color)

            eTime = time.time()
            fps = 1 / (eTime - sTime)
            sTime = eTime
            cv2.putText(rgb_image, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('RGB Video', rgb_image)
            if deep_color is not None:
                cv2.imshow('Deep view video', deep_color)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
