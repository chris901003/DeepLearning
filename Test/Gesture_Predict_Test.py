import cv2
import mediapipe as mp
import time
import math
import torch
from torch import nn


def GetDistance(landmark, scale, idx1, idx2):
    dis_x = max(landmark[idx1].x, landmark[idx2].x) - min(landmark[idx1].x, landmark[idx2].x)
    dis_y = max(landmark[idx1].y, landmark[idx2].y) - min(landmark[idx1].y, landmark[idx2].y)
    dis = math.sqrt(dis_x * dis_x + dis_y * dis_y)
    if scale != -1:
        dis = dis * scale
    return dis


class PoseDetection(nn.Module):
    def __init__(self, num_joints, num_classes):
        super(PoseDetection, self).__init__()
        self.fc1 = nn.Linear(num_joints * num_joints, 200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, num_classes)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def CreateModel(num_joints, num_classes):
    return PoseDetection(num_joints, num_classes)


def main():
    PoseModel = CreateModel(21, 5)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # PoseModel.load_state_dict(torch.load('', map_location='cpu'))
    PoseModel.to(device)
    PoseModel.eval()
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

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                    num_points = len(handLms.landmark)
                    dis_5_17 = GetDistance(handLms.landmark, -1, 5, 17)
                    dis_scale = 1 / dis_5_17
                    all_distance = [[GetDistance(handLms.landmark, dis_scale, i, j)
                                     for j in range(num_points)] for i in range(num_points)]
                    all_distance = torch.Tensor(all_distance)
                    all_distance = all_distance.unsqueeze(dim=0)
                    all_distance = all_distance.to(device)
                    with torch.no_grad():
                        out = PoseModel(all_distance)
                    predict = torch.softmax(out, dim=1)
                    res = torch.argmax(predict, dim=1)
                    # for i, lm in enumerate(handLms.landmark):
                    #     xPos = int(lm.x * imgWidth)
                    #     yPos = int(lm.y * imgHeight)

            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime
            cv2.putText(img, f'Predict pose : {res + 1}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            cv2.imshow('img', img)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
