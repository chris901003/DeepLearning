import time
import torch
from Gesture_API import GestureRecognitionAPI
import cv2
import argparse
import pyautogui
import platform


def parse_args():
    parse = argparse.ArgumentParser(description='手部控制參數')
    parse.add_argument('--model_type', default='GestureRecognition', help='指定的檢測模型')
    parse.add_argument('--num_joints', default=21, help='關節點數量，需要與model_type配合使用')
    parse.add_argument('--num_classes', default=5, help='手部姿勢數量，需要與model_type配合使用')
    parse.add_argument('--keypoint_line', default=None, help='關節點連線關係')
    parse.add_argument('--pretrain', default='best_model.pkt', help='預訓練權重位置')
    parse.add_argument('--pose_threshold', default=0.5, help='設定閾值大於多少認定成功')
    parse.add_argument('--continuous_second', default=2, help='觀察幾秒內的手勢')
    parse.add_argument('--pose_percentage', default=0.7, help='指定秒數內達到多少概率為檢測手勢')
    parse.add_argument('--cool_down', default=2, help='兩指令之間的間隔秒數')
    parse.add_argument('--camera', default=0, help='使用電腦的哪個攝像頭，如果要使用默認的就是0')
    args = parse.parse_args()
    return args


def CreateDetector(args):
    keypoint_line = args.keypoint_line
    if keypoint_line is None:
        keypoint_line = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
                          [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20],
                          [0, 17]]
    model_cfg = {
        'type': args.model_type,
        'num_joints': args.num_joints,
        'num_classes': args.num_classes,
        'keypoint_line': keypoint_line,
        'bilateral': True,
        'pretrain': args.pretrain
    }
    device = torch.device('cpu')
    model = GestureRecognitionAPI(model_cfg, device)
    return model


def PutText(img, strings, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=1, color=(255, 0, 0), width=3):
    if isinstance(strings, str):
        strings = [strings]
    if isinstance(position, tuple):
        position = [position]
    num_msg = len(strings)
    assert len(strings) == len(position), '每個文字需要有指定的位置'
    if not isinstance(font, list):
        font = [font for _ in range(num_msg)]
    if not isinstance(font_size, list):
        font_size = [font_size for _ in range(num_msg)]
    if not isinstance(color, list):
        color = [color for _ in range(num_msg)]
    if not isinstance(width, list):
        width = [width for _ in range(num_msg)]
    assert num_msg == len(font) == len(font_size) == len(color) == len(width), '字型以及字體以及顏色以及寬度需要對應到文字數量'
    for S, P, F, FS, C, W in zip(strings, position, font, font_size, color, width):
        cv2.putText(img, S, P, F, FS, C, W)
    return img


def Lock(gesture):
    if gesture == 5:
        return 1, 'Unlock'
    else:
        return 0, 'Pose 5 to unlock'


def ChoosingMode(gesture):
    if gesture == 1:
        return 0, 'Lock'
    elif gesture == 2:
        return 2, 'Change mode to adjust volume'
    elif gesture == 3:
        return 3, 'Change mode to adjust video'
    else:
        return 1, '1 => Lock, 2 => Adjust volume, 3 => Adjust video'


def AdjustVolume(gesture):
    plt = platform.system().lower()
    if gesture == 5:
        return 1, 'Back to choosing mode'
    elif gesture == 1:
        if plt == 'windows':
            pyautogui.press('volumeup')
        else:
            pyautogui.press('KEYTYPE_SOUND_UP')
        return 2, 'Volume up'
    elif gesture == 2:
        if plt == 'windows':
            pyautogui.press('volumedown')
        else:
            pyautogui.press('KEYTYPE_SOUND_DOWN')
        return 2, 'Volume down'
    else:
        return 2, '1 => Volume up, 2 => Volume down, 5 => Back to choosing mode'


def AdjustVideo(gesture):
    plt = platform.system().lower()
    if gesture == 5:
        return 1, 'Back to choosing mode'
    if gesture == 1:
        if plt == 'windows':
            pyautogui.press('pause')
        else:
            pyautogui.press('KEYTYPE_PLAY')
        return 3, 'Play / Stop video'
    elif gesture == 2:
        pyautogui.press('KEYTYPE_NEXT')
        return 3, 'Next video'
    elif gesture == 3:
        pyautogui.press('KEYTYPE_PREVIOUS')
        return 3, 'Previous video'
    else:
        return 3, '1 => Play or Stop, 2 => Next video, 3 => Previous video'


def main():
    args = parse_args()
    detector = CreateDetector(args)
    cap = cv2.VideoCapture(0)
    pTime = 0
    support_mode = [Lock, ChoosingMode, AdjustVolume, AdjustVideo]
    current_mode_name = ['Lock', 'Choosing', 'Adjust Volume', 'Adjust Video']
    current_mode = 0
    last_operation = 'None'
    last_operation_time = 0
    pose_count = [0 for _ in range(args.num_classes + 1)]

    while True:
        ret, img = cap.read()
        if ret:
            img, res, score = detector(img)
            if score >= args.pose_threshold:
                pose_count[res] += 1

            if time.time() - last_operation_time >= args.continuous_second:
                total_detection = sum(pose_count)
                most_time = max(pose_count)
                most_time_index = pose_count.index(most_time)
                if most_time_index != 0:
                    if most_time / total_detection >= args.pose_threshold:
                        mode = support_mode[current_mode]
                        current_mode, last_operation = mode(res)
                    last_operation_time = time.time()
                    pose_count = [0 for _ in range(args.num_classes + 1)]

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            msg = [f'Predict pose : {res}', f'Predict score : {score}', f'FPS : {int(fps)}',
                   f'Mode : {current_mode_name[current_mode]}', f'Last operation : {last_operation}']
            pos = [(30, 100), (30, 150), (30, 50), (30, 200), (30, 250)]
            img = PutText(img, msg, pos)
            cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
