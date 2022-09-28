# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import time
from collections import deque
from operator import itemgetter
from threading import Thread

import cv2
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import collate, scatter

from mmaction.apis import init_recognizer
from mmaction.datasets.pipelines import Compose

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
    # 使用攝影機進行時時判斷
    parser = argparse.ArgumentParser(description='MMAction2 webcam demo')
    # 選擇模型配置文件
    parser.add_argument('config', help='test config file path')
    # 預訓練權重地址
    parser.add_argument('checkpoint', help='checkpoint file')
    # 標註資料對照表
    parser.add_argument('label', help='label file')
    # 運行設備，如果是使用cpu進行運算需要調整
    parser.add_argument('--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    # 使用的攝影機id
    parser.add_argument('--camera-id', type=int, default=0, help='camera device id')
    # 閾值
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='recognition score threshold')
    # 會總結多少個預測置信度分數最後給定預測結果
    parser.add_argument(
        '--average-size',
        type=int,
        default=1,
        help='number of latest clips to be averaged for prediction')
    # 畫出結果的最大fps數，調低一點可以讓預測模型有更大的資源計算
    parser.add_argument(
        '--drawing-fps',
        type=int,
        default=20,
        help='Set upper bound FPS value of the output drawing')
    # 模型預測的最大fps數，可以避免整個電腦的資源被佔用
    parser.add_argument(
        '--inference-fps',
        type=int,
        default=4,
        help='Set upper bound FPS value of model inference')
    # 額外添加config設定
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    # 打包參數
    args = parser.parse_args()
    # 檢查fps設定需大於0
    assert args.drawing_fps >= 0 and args.inference_fps >= 0, \
        'upper bound FPS value of drawing and inference should be set as ' \
        'positive number, or zero for no limit'
    return args


def show_results():
    # 展示結果的函數
    # 這裡表示可以透過按下哪些案件結束程式
    print('Press "Esc", "q" or "Q" to exit')

    # 文字相關資訊
    text_info = {}
    # 獲取當前時間
    cur_time = time.time()
    while True:
        # 會先表示等待動作
        msg = 'Waiting for action ...'
        # 透過攝影機獲取圖像
        _, frame = camera.read()
        # 將圖像資料放到frame_queue當中，當收集足夠時模型就會進行驗證
        frame_queue.append(np.array(frame[:, :, ::-1]))

        if len(result_queue) != 0:
            # 如果有收到預測結果就會到這裡
            # 構建文字訊息字典
            text_info = {}
            # 將結果資訊提取出來
            results = result_queue.popleft()
            # 遍歷結果當中資料
            for i, result in enumerate(results):
                # 獲取遍歷到的類別以及置信度分數
                selected_label, score = result
                if score < threshold:
                    # 如果置信度分數小於閾值就直接跳出，因為置信度是由大到小排序
                    break
                # 獲取需要填上資料的位置
                # 將資料寫到圖像上
                location = (0, 40 + i * 20)
                text = selected_label + ': ' + str(round(score, 2))
                text_info[location] = text
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        elif len(text_info) != 0:
            # 如果下次的預測結果尚未出來就先顯示之前的
            for location, text in text_info.items():
                cv2.putText(frame, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)

        else:
            cv2.putText(frame, msg, (0, 40), FONTFACE, FONTSCALE, MSGCOLOR,
                        THICKNESS, LINETYPE)

        # 將畫面顯示到視窗上
        cv2.imshow('camera', frame)
        ch = cv2.waitKey(1)

        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        if drawing_fps > 0:
            # add a limiter for actual drawing fps <= drawing_fps
            sleep_time = 1 / drawing_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()


def inference():
    # 構建保存預測置信度分數的deque
    score_cache = deque()
    # 分數總和
    scores_sum = 0
    # 獲取當前時間
    cur_time = time.time()
    # 開始進入無窮迴圈
    while True:
        # 構建一個空的list
        cur_windows = []

        while len(cur_windows) == 0:
            # 當cur_windows長度是0時就會進來
            if len(frame_queue) == sample_length:
                # 如果在frame_queue當中的圖像數量與sample_length相同就會進來
                # 將frame_queue資料轉成ndarray同時放到cur_windows當中
                cur_windows = list(np.array(frame_queue))
                if data['img_shape'] is None:
                    # 如果在data當中沒有說明圖像大小就會到這裡
                    # 將最左端的圖像pop出來，並且獲取圖像的高寬
                    data['img_shape'] = frame_queue.popleft().shape[:2]

        # 拷貝一份data資料到cur_data當中
        cur_data = data.copy()
        # 將cur_windows圖像作為cur_data的圖像資料
        cur_data['imgs'] = cur_windows
        # 將圖像資料通過圖像處理流
        cur_data = test_pipeline(cur_data)
        # 使用collate將資料整理成一個batch的樣子
        cur_data = collate([cur_data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            cur_data = scatter(cur_data, [device])[0]

        with torch.no_grad():
            # 將圖像資料傳入進行正向推理
            scores = model(return_loss=False, **cur_data)[0]

        # 將結果分數放到置信度分數暫存
        score_cache.append(scores)
        # 進行類別分數加總
        scores_sum += scores

        if len(score_cache) == average_size:
            # 如果置信度分數暫存數量到達average_size就會到這裡
            # 計算每個分類類別的平均置信度分數
            scores_avg = scores_sum / average_size
            # 獲取需要選出前k大的可能行為
            num_selected_labels = min(len(label), 5)

            # 將類別名稱與平均置信度分數進行打包，也就是一個置信度分數會配上一個類別
            scores_tuples = tuple(zip(label, scores_avg))
            # 進行排序，這裡會依照平均置信度分數較高的會放到前面
            scores_sorted = sorted(scores_tuples, key=itemgetter(1), reverse=True)
            # 將前k大的置信度分數提取出來作為結果
            results = scores_sorted[:num_selected_labels]

            # 將結果放到結果的deque當中
            result_queue.append(results)
            # 將最左端的置信度分數資料去除，記得置信度總合也需要進行刪除
            scores_sum -= score_cache.popleft()

        if inference_fps > 0:
            # 如果有設定最大fps就會到這裡
            # add a limiter for actual inference fps <= inference_fps
            # 計算需要睡多長的時間讓fps符合設定標準
            sleep_time = 1 / inference_fps - (time.time() - cur_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            cur_time = time.time()

    # 當結束時會將攝影機資源釋放
    camera.release()
    # 同時將cv2的視窗關閉
    cv2.destroyAllWindows()


def main():
    # 設定一堆全域變數
    global frame_queue, camera, frame, results, threshold, sample_length, \
        data, test_pipeline, model, device, average_size, label, \
        result_queue, drawing_fps, inference_fps

    # 獲取啟動時傳入的參數
    args = parse_args()
    # 獲取args當中的參數，這樣之後比較好寫
    average_size = args.average_size
    threshold = args.threshold
    drawing_fps = args.drawing_fps
    inference_fps = args.inference_fps

    # 獲取推理設備
    device = torch.device(args.device)

    # 將config資料讀取出來
    cfg = Config.fromfile(args.config)
    # 將額外添加上的config設定添加上去
    cfg.merge_from_dict(args.cfg_options)

    # 初始化模型，同時加載預訓練權重
    model = init_recognizer(cfg, args.checkpoint, device=device)
    # 設定攝影機
    camera = cv2.VideoCapture(args.camera_id)
    # 構建data字典
    data = dict(img_shape=None, modality='RGB', label=-1)

    with open(args.label, 'r') as f:
        # 讀取標註類別資料，每一行表示一種類別
        label = [line.strip() for line in f]

    # prepare test pipeline from non-camera pipeline
    # 獲取模型的config資料
    cfg = model.cfg
    # 將取樣長度變數設定成0
    sample_length = 0
    # 獲取測試時的影像處理流
    pipeline = cfg.data.test.pipeline
    # 拷貝一份出來
    pipeline_ = pipeline.copy()
    # 開始遍歷影像處理層
    for step in pipeline:
        if 'SampleFrames' in step['type']:
            # 如果是SampleFrames就會到這裡
            # 獲取採樣需要的長度，這裡會是每段的長度以及需要多少段
            sample_length = step['clip_len'] * step['num_clips']
            # 將需要多少段資訊放到data當中
            data['num_clips'] = step['num_clips']
            # 將每段長度資訊放到data當中
            data['clip_len'] = step['clip_len']
            # 將SampleFrames層從pipeline中移除
            pipeline_.remove(step)
        if step['type'] in EXCLUDE_STEPS:
            # 如果當前處理層有在EXCLUDE_STEPS當中就會移除
            # 這裡主要是Init以及Decode層，因為會直接從攝影機讀取所以不需要解碼過程
            # remove step to decode frames
            pipeline_.remove(step)
    # 將pipeline進行Compose
    test_pipeline = Compose(pipeline_)

    # 這裡檢查sample_length需要至少一張圖像
    assert sample_length > 0

    try:
        # 構建保存圖像的deque，這裡最長保存長度會與sample_length相同
        frame_queue = deque(maxlen=sample_length)
        # 構建保存結果的deque，這裡最多只會保存一次的資料
        result_queue = deque(maxlen=1)
        # 這裡使用多線程進行同時處理
        pw = Thread(target=show_results, args=(), daemon=True)
        pr = Thread(target=inference, args=(), daemon=True)
        # 開始運作
        pw.start()
        pr.start()
        pw.join()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
