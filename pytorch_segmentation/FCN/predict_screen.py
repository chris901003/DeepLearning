import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import fcn_resnet50
import pyautogui
import cv2


def time_synchronized():
    # 已看過
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # 已看過
    # 不使用輔助計算因為我們是要預測
    aux = False  # inference time not need aux_classifier
    # 分類數量
    classes = 20
    # 預訓練權重地址
    weights_path = "./save_weights/model_0.pth"
    # 調色盤檔案
    palette_path = "./palette.json"
    # 檢查上面檔案是否都存在
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    # 記錄下調色板
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        # 把調色板中的值全部放入list當中 [0, 0, 0, 128, 0, 0, ...]
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = fcn_resnet50(aux=aux, num_classes=classes+1)

    # delete weights about aux_classifier
    # 加載權重，這裡會把輔助分類的權重捨棄
    weights_dict = torch.load(weights_path, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])

    model.eval()  # 进入验证模式
    with torch.no_grad():

        # 獲取開始時間
        fps = 0.0
        while True:
            # 顯示視窗需要用到的，不加的話會報錯
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            t_start = time_synchronized()
            # 開始預測
            img = pyautogui.screenshot(region=[0, 0, 850, 500])
            orig_img = img.copy()
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)
            output = model(img.to(device))
            # 結束時間
            t_end = time_synchronized()

            # 取出預測並經過argmax還有把前面的batch_size維度拿掉
            prediction = output['out'].argmax(1).squeeze(0)
            # 轉成numpy格式
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            # 根據調色板與預測獲得圖片
            mask = Image.fromarray(prediction)
            mask.putpalette(pallette)
            # 從P模式轉成RGB模式
            mask = mask.convert('RGB')
            # 原圖與分割圖調整到一樣大小才可以融合
            imgSize = (850, 500)
            mask = mask.resize(imgSize)
            orig_img = orig_img.resize(imgSize)
            # 將兩原圖與分割圖融合
            combine = Image.blend(mask, orig_img, alpha=0.5)
            frame = cv2.cvtColor(np.asarray(combine), cv2.COLOR_RGB2BGR)
            fps = (fps + (1. / (time.time() - t_start))) / 2
            print("fps= %.2f" % fps)
            frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
