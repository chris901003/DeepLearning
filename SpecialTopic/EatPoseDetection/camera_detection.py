import argparse
import torch
import cv2
import time
import numpy as np
from collections import deque
from threading import Thread
from operator import itemgetter
from api import init_recognizer
from SpecialTopic.ST.dataset.utils import Compose


FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1

EXCLUDE_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]


def parse_args():
    parser = argparse.ArgumentParser('Camera Detection')
    parser.add_argument('--checkpoint', type=str, default=r'C:\Checkpoint\Kinetics400\10_1.94.pth')
    parser.add_argument('--label', type=str, default=r'C:\Dataset\kinetics400\label_map_k400.txt')
    parser.add_argument('--camera-id', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--average-size', type=int, default=1)
    parser.add_argument('--drawing-fps', type=int, default=20)
    parser.add_argument('--inference-fps', type=int, default=4)
    args = parser.parse_args()
    return args


def get_pipeline():
    pipeline = [
        {'type': 'PyAVInit'},
        {'type': 'SampleFrames', 'clip_len': 32, 'frame_interval': 2, 'num_clips': 2, 'test_mode': True},
        {'type': 'PyAVDecode'},
        {'type': 'Recognizer3dResize', 'scale': (-1, 256)},
        {'type': 'ThreeCrop', 'crop_size': 256},
        {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_bgr': False},
        {'type': 'FormatShape', 'input_format': 'NCTHW'},
        {'type': 'Collect', 'keys': ['imgs', 'label']},
        {'type': 'ToTensor', 'keys': ['imgs', 'label']}
    ]
    return pipeline


def show_results(result_queue, threshold, frame):
    text_info = dict()
    msg = 'Waiting for action ...'
    if len(result_queue) != 0:
        text_info = {}
        results = result_queue.popleft()
        for i, result in enumerate(results):
            selected_label, score = result
            if score < threshold:
                break
            location = (0, 40 + i * 20)
            text = selected_label + ': ' + str(round(score, 2))
            text_info[location] = text
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
    elif len(text_info) != 0:
        for location, text in text_info.items():
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
    else:
        cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR, THICKNESS, LINETYPE)
    return frame


def collate(batch):
    imgs, labels = list(), list()
    for info in batch:
        imgs.append(info['imgs'])
        labels.append(info['label'])
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    return imgs, labels


def inference(frame_queue, sample_length, data, test_pipeline, device, model, average_size, label, result_queue,
              inference_fps):
    score_cache = deque()
    scores_sum = 0
    cur_time = time.time()
    while True:
        cur_window = list()
        while len(cur_window) == 0:
            if len(frame_queue) == sample_length:
                cur_window = list(np.array(frame_queue))
                if data['img_shape'] is None:
                    data['img_shape'] = frame_queue.popleft().shape[:2]
        cur_data = data.copy()
        cur_data['imgs'] = cur_window
        cur_data = test_pipeline(cur_data)
        cur_data = collate([cur_data])
        imgs, labels = cur_data
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            scores = model(imgs, labels, mode='test').cpu().numpy()[0]
        score_cache.append(scores)
        scores_sum += scores
        if len(score_cache) == average_size:
            scores_avg = scores_sum / average_size
            num_selected_labels = min(len(label), 5)
            scores_tuples = tuple(zip(label, scores_avg))
            scores_sorted = sorted(scores_tuples, key=itemgetter(1), reverse=True)
            results = scores_sorted[:num_selected_labels]
            result_queue.append(results)
            scores_sum -= score_cache.popleft()
        if inference_fps > 0:
            sleep_time = 1 / inference_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()


def main():
    args = parse_args()
    average_size = args.average_size
    threshold = args.threshold
    drawing_fps = args.drawing_fps
    inference_fps = args.inference_fps
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = init_recognizer(num_classes=400, checkpoint=args.checkpoint, device=device)
    camera = cv2.VideoCapture(args.camera_id)
    data = dict(img_shape=None, modality='RGB', label=-1)
    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]
    sample_length = 0
    pipeline = get_pipeline()
    pipeline_ = pipeline.copy()
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            sample_length = step['clip_len'] * step['num_clips']
            data['num_clips'] = step['num_clips']
            data['clip_len'] = step['clip_len']
            pipeline_.remove(step)
        if step['type'] in EXCLUDE_STEPS:
            pipeline_.remove(step)
    test_pipeline = Compose(pipeline_)
    frame_queue = deque(maxlen=sample_length)
    result_queue = deque(maxlen=1)
    show_cur_time = time.time()
    pr = Thread(target=inference, args=(frame_queue, sample_length, data, test_pipeline, device, model, average_size,
                                        label, result_queue, inference_fps), daemon=True)
    pr.start()
    while True:
        _, frame = camera.read()
        frame_queue.append(np.array(frame[:, :, ::-1]))
        if drawing_fps > 0:
            sleep_time = 1 / drawing_fps - (time.time() - show_cur_time)
            if sleep_time <= 0:
                out_frame = show_results(result_queue, threshold, frame)
                show_cur_time = time.time()
                cv2.imshow('camera', out_frame)
        else:
            out_frame = show_results(result_queue, threshold, frame)
            show_cur_time = time.time()
            cv2.imshow('camera', out_frame)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
    camera.release()
    cv2.destroyAllWindows()
    pr.join()


if __name__ == '__main__':
    main()
    print('Finish')
