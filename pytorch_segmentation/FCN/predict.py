import os
import time
import json

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import fcn_resnet50


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
    weights_path = "./save_weights/model_29.pth"
    # 測試照片
    img_path = "./test.jpg"
    # 調色盤檔案
    palette_path = "./palette.json"
    # 檢查上面檔案是否都存在
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
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

    # load image
    original_img = Image.open(img_path)

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        # 構建一個全為零且shape與圖像一樣的tensor
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        # 這裡是先讓模型跑一次，第一次時模型會比較久為了精確知道執行時間所以先熱身一次
        model(init_img)

        # 獲取開始時間
        t_start = time_synchronized()
        # 開始預測
        output = model(img.to(device))
        # 結束時間
        t_end = time_synchronized()
        # 輸出預測一張照片花的時間
        print("inference+NMS time: {}".format(t_end - t_start))

        # 取出預測並經過argmax還有把前面的batch_size維度拿掉
        prediction = output['out'].argmax(1).squeeze(0)
        # 轉成numpy格式
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 根據調色板與預測獲得圖片
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)
        mask.save("test_result.png")


if __name__ == '__main__':
    main()
