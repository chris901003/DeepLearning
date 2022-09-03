import cv2
import torch
from TemporalGestureRecognition_API import TemporalGestureRecognitionAPI
import time


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


def main():
    keypoint_extract_cfg = {
        'type': 'mediapipe_hands',
        'max_num_hands': 1,
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5
    }
    model_cfg = {
        'type': 'SkeletonGCN',
        'backbone': {
            'type': 'STGCN',
            'in_channels': 3,
            'edge_importance_weighting': True,
            'graph_cfg': {
                'layout': 'mediapipe_hands',
                'strategy': 'spatial'
            }
        },
        'cls_head': {
            'type': 'STGCNHead',
            'num_classes': 4,
            'in_channels': 256,
            'loss_cls': {
                'type': 'CrossEntropyLoss'
            }
        }
    }
    norm_cfg = {
        'min_value': (0., 0., 0.), 'max_value': (1920., 1080., 1.), 'mean': (940., 540., 0.5)
    }
    pretrained = '/Users/huanghongyan/Documents/DeepLearning/pytorch_geometric/' \
                 'TemporalGestureRecognition/best_model.pkt'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    predict_api = TemporalGestureRecognitionAPI(keypoint_extract_cfg, model_cfg, norm_cfg, device,
                                                keep_time=60, pretrained=pretrained)
    pTime = 0
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if ret:
            result = predict_api(img)
            pose = result['classes']
            score = result['score']
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            msg = [f'Predict pose : {pose}', f'Predict score : {score}', f'FPS : {int(fps)}']
            pos = [(30, 100), (30, 150), (30, 50)]
            img = PutText(img, msg, pos)
            cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
    print('Finish')
