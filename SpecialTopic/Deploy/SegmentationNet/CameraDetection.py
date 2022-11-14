import argparse
import cv2
import time
from SpecialTopic.Deploy.SegmentationNet.api import create_tensorrt_engine, tensorrt_engine_detect_image


def parse_args():
    parser = argparse.ArgumentParser()
    # 使用的相機ID
    parser.add_argument('--camera-id', type=int, default=0)
    # 相機FPS值
    parser.add_argument('--fps', type=int, default=30)
    #
    parser.add_argument('--onnx-file', type=str, default=None)
    parser.add_argument('--trt-engine-path', type=str, default='SegmentationNetM.trt')
    parser.add_argument('--save-engine-path', type=str, default='SegmentationNetM.trt')
    parser.add_argument('--fp16', action='store_false')
    args = parser.parse_args()
    return args
