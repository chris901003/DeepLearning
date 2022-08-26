import torch
import mediapipe as mp
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
import cv2
from torch_geometric.nn import GCNConv
import time


def CreateModel(cfg):
    support_model = {
        'GestureRecognition': GestureRecognition
    }
    model_type = cfg.pop('type', None)
    assert model_type is not None, "模型cfg當中需要type指定使用模型"
    if model_type not in support_model:
        assert False, "該模型沒有實作對象"
    model_cls = support_model[model_type]
    model = model_cls(**cfg)
    return model


class GestureRecognition(nn.Module):
    def __init__(self, num_joints, num_classes, keypoint_line=None, bilateral=None):
        super(GestureRecognition, self).__init__()
        self.conv1 = GCNConv(num_joints * 2, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        if keypoint_line is not None:
            if bilateral:
                keypoint_line_bilateral = [[p[1], p[0]]for p in keypoint_line]
                keypoint_line.extend(keypoint_line_bilateral)
            self.keypoint_line = keypoint_line
            if not torch.is_tensor(self.keypoint_line):
                self.keypoint_line = torch.LongTensor(self.keypoint_line)
            if self.keypoint_line.shape[0] != 2:
                self.keypoint_line = self.keypoint_line.t()

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = self.keypoint_line
        assert edge_index is not None, "使用GCNConv需要提供連線資訊"
        out = self.conv1(x, edge_index)
        out = self.relu(out)
        out = self.conv2(out, edge_index)
        out = self.relu(out)
        out = self.fc(out)
        return out


def get_keypoint_feature(keypoints_position, idx1, idx2):
    dif_x = keypoints_position[idx2].x - keypoints_position[idx1].x
    dif_y = keypoints_position[idx2].y - keypoints_position[idx1].y
    return torch.tensor([dif_x, dif_y])


def main():
    model_cfg = {
        'type': 'GestureRecognition',
        'num_joints': 21,
        'num_classes': 5,
        'keypoint_line': [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
                          [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20],
                          [0, 17]],
        'bilateral': True
    }
    gesture_model = CreateModel(model_cfg)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gesture_model.load_state_dict(torch.load('best_model.pkt', map_location='cpu'))
    gesture_model.to(device)
    gesture_model.eval()
    cap = cv2.VideoCapture(0)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)
    pTime = 0
    cTime = 0
    while True:
        ret, img = cap.read()
        if ret:
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(imgRGB)

            imgHeight = img.shape[0]
            imgWidth = img.shape[1]
            res = -1
            score = 0

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                    num_points = len(handLms.landmark)
                    keypoint_feature = [torch.stack([get_keypoint_feature(handLms.landmark, i, j)
                                                     for i in range(num_points)]) for j in range(num_points)]
                    keypoint_feature = torch.stack(keypoint_feature)
                    keypoint_feature = keypoint_feature.reshape(num_points, -1)
                    keypoint_feature = keypoint_feature.unsqueeze(dim=0)
                    predict = gesture_model(keypoint_feature)
                    results = predict.sum(dim=1)
                    predict = torch.softmax(results, dim=1)
                    res = torch.argmax(predict, dim=1).item()
                    score = predict[0][res]
                    if score < 0.5:
                        res = 'Not sure'
                    else:
                        res += 1

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img, f'Predict pose : {res}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f'Predict score : {score}', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            cv2.imshow('img', img)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
