import tensorrt
import argparse
import os
import time
import cv2
from SpecialTopic.WorkingFlow.utils import parser_cfg
from SpecialTopic.WorkingFlow.build import WorkingSequence


def args_parse():
    parser = argparse.ArgumentParser('All starting from here')
    # 設定每個模塊需要接收的資料以及輸出的結果
    parser.add_argument('--working-flow-cfg', type=str,
                        default='./config/WorkingFlow/working_flow_cfg.json')
    # 如果有需要手動設定高寬就填上(height, width)，使用-1就是直接使用原始大小
    parser.add_argument('--show-window-size', type=int, default=-1, nargs='+')
    # 如果不需要將過程轉成影片保存就改成none
    parser.add_argument('--save-video', type=str, default='none')
    # 接下來還會有一堆下游工作，目前先處理這兩個的連動
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    working_flow_cfg = parser_cfg(args.working_flow_cfg)
    save_video_path = args.save_video
    video_write = None
    if save_video_path != 'none':
        if not os.path.exists(save_video_path):
            os.mkdir(save_video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_name = os.path.join(save_video_path, 'test.mp4')
        video_write = cv2.VideoWriter(video_name, fourcc, 30, (1920, 1080))
    show_window_size = args.show_window_size
    working_flow = WorkingSequence(working_flow_cfg)
    step_add_input = {'ObjectClassifyToRemainClassify': {'0': {'using_dict_name': 'FoodDetection9'}}}
    pTime = 0
    while True:
        result = working_flow(step_add_input=step_add_input)
        result_image = result['image']
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(result_image, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if isinstance(show_window_size, (tuple, list)):
            assert len(show_window_size) == 2, '需提供高寬大小'
            cv2.namedWindow("result", 0)
            cv2.resizeWindow("result", show_window_size[1], show_window_size[0])
        cv2.imshow('result', result_image)
        if video_write is not None:
            video_write.write(result_image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
    print('Finish')
