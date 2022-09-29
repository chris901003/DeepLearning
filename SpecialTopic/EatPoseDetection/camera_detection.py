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


EXCLUDE_STEPS = [
    'OpenCVInit', 'OpenCVDecode', 'DecordInit', 'DecordDecode', 'PyAVInit',
    'PyAVDecode', 'RawFrameDecode'
]


def parse_args():
    parser = argparse.ArgumentParser('Camera Detection')
    # 預訓練權重資料路徑
    parser.add_argument('--checkpoint', type=str, default='none')
    # classes檔案路徑
    parser.add_argument('--label', type=str, default='none')
    # 總共分類類別墅
    parser.add_argument('--num-classes', type=int, default=400)
    # 使用哪個相機，如果是對影片進行檢測請直接放影片路徑
    parser.add_argument('--camera-id', type=int, default=0)
    # 最小檢測閾值
    parser.add_argument('--threshold', type=float, default=0.01)
    # 多少次檢測的平均作為結果
    parser.add_argument('--average-size', type=int, default=1)
    # 顯示到螢幕上的最大FPS
    parser.add_argument('--drawing-fps', type=int, default=20)
    # 進行預測的最大FPS
    parser.add_argument('--inference-fps', type=int, default=4)
    # pipeline的超參數，如果有需要自行去看pipeline
    parser.add_argument('--clip-len', type=int, default=32)
    parser.add_argument('--frame-interval', type=int, default=2)
    parser.add_argument('--num-clips', type=int, default=2)
    args = parser.parse_args()
    return args


def get_pipeline(args):
    pipeline = [
        {'type': 'PyAVInit'},
        {'type': 'SampleFrames', 'clip_len': args.clip_len, 'frame_interval': args.frame_interval,
         'num_clips': args.num_clips, 'test_mode': True},
        {'type': 'PyAVDecode'},
        {'type': 'Recognizer3dResize', 'scale': (-1, 256)},
        {'type': 'ThreeCrop', 'crop_size': 256},
        {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_bgr': False},
        {'type': 'FormatShape', 'input_format': 'NCTHW'},
        {'type': 'Collect', 'keys': ['imgs', 'label']},
        {'type': 'ToTensor', 'keys': ['imgs', 'label']}
    ]
    return pipeline


def show_results(result_queue, threshold, frame, text_info):
    msg = 'Waiting for action ...'
    image_height, image_width = frame.shape[:2]
    min_length = min(image_width, image_height)
    FONTFACE = cv2.FONT_HERSHEY_COMPLEX_SMALL
    FONTSCALE = 1 if min_length < 500 else 2
    FONTCOLOR = (255, 255, 255)  # BGR, white
    MSGCOLOR = (128, 128, 128)  # BGR, gray
    THICKNESS = 1 if min_length < 500 else 2
    LINETYPE = 1
    if len(result_queue) != 0:
        text_info = {}
        results = result_queue.popleft()
        for i, result in enumerate(results):
            selected_label, score = result
            if score < threshold:
                break
            location = (0, int(image_height * 0.05) + i * int(image_height * 0.05 * 0.5))
            text = selected_label + ': ' + str(round(score, 2))
            text_info[location] = text
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
    elif len(text_info) != 0:
        for location, text in text_info.items():
            cv2.putText(frame, text, location, FONTFACE, FONTSCALE, FONTCOLOR, THICKNESS, LINETYPE)
    else:
        cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR, THICKNESS, LINETYPE)
    return frame, text_info


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
    model = init_recognizer(num_classes=args.num_classes, checkpoint=args.checkpoint, device=device)
    camera = cv2.VideoCapture(args.camera_id)
    data = dict(img_shape=None, modality='RGB', label=-1)
    with open(args.label, 'r') as f:
        label = [line.strip() for line in f]
    sample_length = 0
    pipeline = get_pipeline(args)
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
    text_info = dict()
    pr = Thread(target=inference, args=(frame_queue, sample_length, data, test_pipeline, device, model, average_size,
                                        label, result_queue, inference_fps), daemon=True)
    pr.start()
    while True:
        _, frame = camera.read()
        frame_queue.append(np.array(frame[:, :, ::-1]))
        if drawing_fps > 0:
            sleep_time = 1 / drawing_fps - (time.time() - show_cur_time)
            if sleep_time <= 0:
                out_frame, text_info = show_results(result_queue, threshold, frame, text_info)
                show_cur_time = time.time()
                cv2.imshow('camera', out_frame)
        else:
            out_frame, text_info = show_results(result_queue, threshold, frame, text_info)
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
