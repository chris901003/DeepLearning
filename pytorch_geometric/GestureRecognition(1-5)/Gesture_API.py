import torch
import mediapipe as mp
from torch import nn
import cv2
from torch_geometric.nn import GCNConv
import time


def CreateModel(model_cfg):
    support_model = {
        'GestureRecognition': GestureRecognition
    }
    model_type = model_cfg.pop('type', None)
    assert model_type is not None, '需指定使用辨識模型'
    if model_type not in support_model:
        assert False, f'尚未實做 {model_type} 模型'
    pretrain = model_cfg.pop('pretrain', None)
    model_cls = support_model[model_type]
    model = model_cls(**model_cfg)
    if pretrain is not None:
        model.load_state_dict(torch.load(pretrain, map_location='cpu'))
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


class GestureRecognitionAPI:
    def __init__(self, model_cfg, device):
        self.model = CreateModel(model_cfg)
        self.model = self.model.to(device)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        self.handLmsStyle = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)
        self.handConStyle = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)

    def __call__(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        keypoint_result = self.hands.process(imgRGB)
        if keypoint_result.multi_hand_landmarks:
            for handLms in keypoint_result.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS,
                                           self.handLmsStyle, self.handConStyle)
                num_points = len(handLms.landmark)
                keypoint_feature = [torch.stack([get_keypoint_feature(handLms.landmark, i, j)
                                                 for i in range(num_points)]) for j in range(num_points)]
                keypoint_feature = torch.stack(keypoint_feature)
                keypoint_feature = keypoint_feature.reshape(num_points, -1)
                keypoint_feature = keypoint_feature.unsqueeze(dim=0)
                predict = self.model(keypoint_feature)
                results = predict.sum(dim=1)
                predict = torch.softmax(results, dim=1)
                res = torch.argmax(predict, dim=1).item()
                score = predict[0][res]
                return img, res + 1, score
        else:
            return img, 0, 0
