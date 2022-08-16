# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import cv2
# import decord
import numpy as np
import torch
import webcolors
from mmcv import Config, DictAction

from mmaction.apis import inference_recognizer, init_recognizer


def parse_args():
    # 已看過，demo部分的傳入參數
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    # 指定的模型config檔案位置
    parser.add_argument('config', help='test config file path')
    # 訓練權重檔案位置，這裡可以傳入的是url
    parser.add_argument('checkpoint', help='checkpoint file/url')
    # 要進行預測的影片資料，可以是檔案或是網址，可以是影片或是每一幀的圖像
    parser.add_argument('video', help='video file/url or rawframes directory')
    # 標註列表的檔案，最後需要透過index映射回到對應的字串
    parser.add_argument('label', help='label file')
    # 額外添加到config當中的參數
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    # 傳入的影片資料如果是分解成一幀一幀就會需要使用use_frames
    parser.add_argument(
        '--use-frames',
        default=False,
        action='store_true',
        help='whether to use rawframes as input')
    # 推理的設備，這裡預設會是去掉用gpu，所以如果是要用cpu進行推理要特別設定
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    # 指定輸出的fps，當輸入是透過一幀一幀進行輸入時需要指定
    parser.add_argument(
        '--fps',
        default=30,
        type=int,
        help='specify fps value of the output video when using rawframes to '
        'generate file')
    # 標註文字大小
    parser.add_argument(
        '--font-scale',
        default=0.5,
        type=float,
        help='font scale of the label in output video')
    # 標註文字顏色
    parser.add_argument(
        '--font-color',
        default='white',
        help='font color of the label in output video')
    # 分辨率，如果調得較細緻會讓輸出時間變長
    parser.add_argument(
        '--target-resolution',
        nargs=2,
        default=None,
        type=int,
        help='Target resolution (w, h) for resizing the frames when using a '
        'video as input. If either dimension is set to -1, the frames are '
        'resized by keeping the existing aspect ratio')
    # 使用的resize算法
    parser.add_argument(
        '--resize-algorithm',
        default='bicubic',
        help='resize algorithm applied to generate video')
    # 如果需要將結果輸出就需要設定輸出的位置，如果沒有設定就不會輸出影片，只會打印出可能的行為
    parser.add_argument('--out-filename', default=None, help='output filename')
    # 將參數壓縮到args當中
    args = parser.parse_args()
    # 回傳args
    return args


def get_output(video_path,
               out_filename,
               label,
               fps=30,
               font_scale=0.5,
               font_color='white',
               target_resolution=None,
               resize_algorithm='bicubic',
               use_frames=False):
    """Get demo output using ``moviepy``.

    This function will generate video file or gif file from raw video or
    frames, by using ``moviepy``. For more information of some parameters,
    you can refer to: https://github.com/Zulko/moviepy.

    Args:
        video_path (str): The video file path or the rawframes directory path.
            If ``use_frames`` is set to True, it should be rawframes directory
            path. Otherwise, it should be video file path.
        out_filename (str): Output filename for the generated file.
        label (str): Predicted label of the generated file.
        fps (int): Number of picture frames to read per second. Default: 30.
        font_scale (float): Font scale of the label. Default: 0.5.
        font_color (str): Font color of the label. Default: 'white'.
        target_resolution (None | tuple[int | None]): Set to
            (desired_width desired_height) to have resized frames. If either
            dimension is None, the frames are resized by keeping the existing
            aspect ratio. Default: None.
        resize_algorithm (str): Support "bicubic", "bilinear", "neighbor",
            "lanczos", etc. Default: 'bicubic'. For more information,
            see https://ffmpeg.org/ffmpeg-scaler.html
        use_frames: Determine Whether to use rawframes as input. Default:False.
    """

    if video_path.startswith(('http://', 'https://')):
        raise NotImplementedError

    try:
        from moviepy.editor import ImageSequenceClip
    except ImportError:
        raise ImportError('Please install moviepy to enable output file.')

    # Channel Order is BGR
    if use_frames:
        frame_list = sorted(
            [osp.join(video_path, x) for x in os.listdir(video_path)])
        frames = [cv2.imread(x) for x in frame_list]
    else:
        video = decord.VideoReader(video_path)
        frames = [x.asnumpy()[..., ::-1] for x in video]

    if target_resolution:
        w, h = target_resolution
        frame_h, frame_w, _ = frames[0].shape
        if w == -1:
            w = int(h / frame_h * frame_w)
        if h == -1:
            h = int(w / frame_w * frame_h)
        frames = [cv2.resize(f, (w, h)) for f in frames]

    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale,
                               1)[0]
    textheight = textsize[1]
    padding = 10
    location = (padding, padding + textheight)

    if isinstance(font_color, str):
        font_color = webcolors.name_to_rgb(font_color)[::-1]

    frames = [np.array(frame) for frame in frames]
    for frame in frames:
        cv2.putText(frame, label, location, cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, font_color, 1)

    # RGB order
    frames = [x[..., ::-1] for x in frames]
    video_clips = ImageSequenceClip(frames, fps=fps)

    out_type = osp.splitext(out_filename)[1][1:]
    if out_type == 'gif':
        video_clips.write_gif(out_filename)
    else:
        video_clips.write_videofile(out_filename, remove_temp=True)


def main():
    # 已看過，進行展示
    # 獲取啟動時傳入的參數
    args = parse_args()
    # assign the desired device.
    # 指定接下來會在哪個設備上執行
    device = torch.device(args.device)

    # 讀取config資料，將config資料讀出
    cfg = Config.fromfile(args.config)
    # 如果有設定額外的config資料就會到這裡，將額外的資料進行融合
    cfg.merge_from_dict(args.cfg_options)

    # build the recognizer from a config file and checkpoint file/url
    # 初始化模型，並且將模型轉到指定設備上
    model = init_recognizer(cfg, args.checkpoint, device=device)

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    # test a single video or rawframes of a single video
    if output_layer_names:
        results, returned_feature = inference_recognizer(
            model, args.video, outputs=output_layer_names)
    else:
        # 進行正向傳遞獲取預測結果
        results = inference_recognizer(model, args.video)

    # 將類別的label從檔案當中讀取出來
    labels = open(args.label).readlines()
    # 將每個str的開頭以及結尾的換行符去除
    labels = [x.strip() for x in labels]
    # 將results當中的index部分用對應的index替代掉
    results = [(labels[k[0]], k[1]) for k in results]

    # 打印出當前狀態
    print('The top-5 labels with corresponding scores are:')
    for result in results:
        # 將結果打印出來
        print(f'{result[0]}: ', result[1])

    if args.out_filename is not None:
        # 如果有需要將結果透過影片進行輸出就會到這裡

        if args.target_resolution is not None:
            if args.target_resolution[0] == -1:
                assert isinstance(args.target_resolution[1], int)
                assert args.target_resolution[1] > 0
            if args.target_resolution[1] == -1:
                assert isinstance(args.target_resolution[0], int)
                assert args.target_resolution[0] > 0
            args.target_resolution = tuple(args.target_resolution)

        get_output(
            args.video,
            args.out_filename,
            results[0][0],
            fps=args.fps,
            font_scale=args.font_scale,
            font_color=args.font_color,
            target_resolution=args.target_resolution,
            resize_algorithm=args.resize_algorithm,
            use_frames=args.use_frames)


if __name__ == '__main__':
    main()
