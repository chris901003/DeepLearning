from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import cv2
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import time
# 本測試檔案主要程式碼來自: https://github.com/NklausMikealson/Python-Kinectv2


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


def deep_color_image(deep_info, color_palette):
    min_value = 600
    max_value = 800
    dpt_clip = np.clip(deep_info, min_value, max_value)
    dpt_clip = (dpt_clip - min_value) / (max_value - min_value) * 253
    dpt_clip = dpt_clip.astype(int)
    color_map = color_palette[dpt_clip, :].reshape([dpt_clip.shape[0], dpt_clip.shape[1], 4])[..., :3]
    color_map = (color_map * 255).astype(np.uint8)
    return color_map


def main():
    color_palette = plt.cm.get_cmap('jet_r')(range(255))
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Color)
    pTime = 0

    from SpecialTopic.YoloxObjectDetection.api import init_model, detect_image
    import torch
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = init_model(pretrained=r'C:\Checkpoint\YoloxFoodDetection\900_yolox_850.25.pth',
    #                    num_classes=9)

    while True:
        if kinect.has_new_depth_frame():
            color_frame = kinect.get_last_color_frame()
            colorImage = color_frame.reshape((kinect.color_frame_desc.Height,
                                              kinect.color_frame_desc.Width, 4)).astype(np.uint8)
            colorImage = cv2.flip(colorImage, 1)
            img = depth_2_color_space(kinect, _DepthSpacePoint, kinect._depth_frame_data, show=False,
                                      return_aligned_image=True)
            rgb_image = cv2.resize(colorImage, (1920 // 2, 1080 // 2))
            deep_image = cv2.resize(cv2.flip(img, 1), (1920 // 2, 1080 // 2))
            deep_color = deep_color_image(deep_image, color_palette)
            #
            # results = detect_image(model, device, rgb_image, [640, 640], num_classes=9)
            # image_height, image_width = rgb_image.shape[:2]
            # labels, scores, boxes = results
            # for label, score, box in zip(labels, scores, boxes):
            #     ymin, xmin, ymax, xmax = box
            #     if ymin <= 0 or xmin <= 0 or ymax >= image_height or xmax >= image_width:
            #         continue
            #     ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
            #     xmin, ymin = max(0, xmin), max(0, ymin)
            #     xmax, ymax = min(image_width, xmax), min(image_height, ymax)
            #     cv2.rectangle(rgb_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            #     cv2.rectangle(deep_color, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(rgb_image, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.imshow('RGB Image', rgb_image)
            cv2.imshow('Deep Image', deep_image)
            cv2.imshow('Color Deep Image', deep_color)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
