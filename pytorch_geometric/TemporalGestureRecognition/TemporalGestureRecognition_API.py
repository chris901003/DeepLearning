import numpy as np
import mediapipe as mp
from TemporalGestureRecognition_Train import CreateModel
from HandKeypointExtract import normalize_z_axis
import cv2
import torch


def get_cls_from_cfg(support, cfg):
    cls_type = cfg.pop('type', None)
    assert cls_type is not None, '在設定檔當中沒有指定type'
    assert cls_type in support, f'指定的{cls_type}尚未支援'
    cls = support[cls_type]
    return cls


class NormKeypoint:
    def __init__(self, min_value, max_value, mean):
        self.min_value = np.array(min_value, dtype=np.float32).reshape(-1, 1)
        self.max_value = np.array(max_value, dtype=np.float32).reshape(-1, 1)
        self.mean = np.array(mean, dtype=np.float32).reshape(-1, 1)

    def __call__(self, keypoint):
        keypoint = (keypoint - self.mean) / (self.max_value - self.min_value)
        return keypoint


class TemporalGestureRecognitionAPI:
    def __init__(self, keypoint_extract_cfg, model_cfg, norm_cfg, device, keep_time=150, pretrained=None):
        self.keypoint_extract_cfg = keypoint_extract_cfg
        self.build_keypoint_extract()
        self.gesture_model = CreateModel(model_cfg)
        self.norm_keypoint = NormKeypoint(**norm_cfg)
        self.gesture_model = self.gesture_model.to(device)
        self.keep_time = keep_time
        self.keypoints = None
        self.score = 0
        self.cls = 0
        self.pretrained = pretrained
        self.init_weight()

    def init_weight(self):
        if self.pretrained is not None:
            self.gesture_model.load_state_dict(torch.load(self.pretrained, map_location='cpu'))
        self.gesture_model = self.gesture_model.eval()

    def __call__(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgHeight, imgWidth = imgRGB.shape[:2]
        keypoint_result = self.hands.process(imgRGB)
        if keypoint_result.multi_hand_landmarks:
            keypoint = list()
            z = list()
            for lm in keypoint_result.multi_hand_landmarks[0].landmark:
                xPos = int(lm.x * imgWidth)
                yPos = int(lm.y * imgHeight)
                # 添加上z座標，先將所有點進行標準化
                keypoint.append([xPos, yPos])
                z.append(lm.z)
            z = normalize_z_axis(z)
            keypoint = torch.tensor(keypoint)
            z = torch.tensor(z)
            z = z.unsqueeze(dim=-1)
            # keypoint shape = [num_node, channel]
            keypoint = torch.concat((keypoint, z), dim=-1)
            # keypoint shape = [channel, num_node]
            keypoint = keypoint.permute(1, 0).contiguous()
            keypoint = self.norm_keypoint(keypoint)
            channel, num_node = keypoint.shape
            # keypoint shape = [batch_size, channel(x, y, z), frames, num_node, people]
            keypoint = keypoint.reshape(1, channel, 1, num_node, 1)
            if self.keypoints is None:
                self.keypoints = keypoint
            else:
                self.keypoints = torch.cat((keypoint, self.keypoints), dim=2)
        if self.keypoints is not None and self.keypoints.shape[2] > 30:
            self.keypoints = self.keypoints[:, :, :self.keep_time]
            labels = torch.LongTensor([[0]])
            with torch.no_grad():
                predict = self.gesture_model(self.keypoints, labels)
            pred_score = predict['pred'].cpu().detach().numpy()
            self.cls = predict['cls'].item()
            self.score = pred_score[0][self.cls]
        result = {
            'classes': self.cls,
            'score': self.score
        }
        return result

    def build_keypoint_extract(self):
        support_keypoint_extract = {
            'mediapipe_hands': self.build_mediapipe_hands
        }
        keypoint_extract_cls = get_cls_from_cfg(support_keypoint_extract, self.keypoint_extract_cfg)
        keypoint_extract_cls()

    def build_mediapipe_hands(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(**self.keypoint_extract_cfg)
        self.mpDraw = mp.solutions.drawing_utils
        self.handLmsStyle = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)
        self.handConStyle = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)
