import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models import HighResolutionNet
from draw_utils import draw_keypoints
import transforms


def predict_all_person():
    # TODO
    pass


def predict_single_person():
    # 已看過
    # 指認預測設備
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # 設定是否要透過將照片翻轉進行兩次標註
    flip_test = True
    # 輸入到模型的圖像高寬
    resize_hw = (256, 192)
    # 指定圖像路徑
    img_path = "./person.png"
    # 預訓練權重地址
    weights_path = "./pose_hrnet_w32_256x192.pth"
    # 關節點json檔案
    keypoint_json_path = "person_keypoints.json"
    # 檢查檔案是否都存在
    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    # 將輸入的照片進行轉換
    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read json file
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)

    # read single-person image
    # 讀入單張照片，記得照片是要以人物為主體
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 在進行圖像轉換時需要隨便給一個target因為在訓練以及驗證的時候有，這裡隨便給就可以了
    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    # 在batch_size上面做擴維
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # create models
    # HRNet-W32: base_channel=32
    # HRNet-W48: base_channel=48
    # 構建模型，這裡我們是用base_channel為32的
    model = HighResolutionNet(base_channel=32)
    # 載入預訓練權重
    weights = torch.load(weights_path, map_location=device)
    # 因為在訓練時有保存其他資訊，這裡我們只需要models的內容就可以了
    weights = weights if "models" not in weights else weights["models"]
    # 載入到模型當中
    model.load_state_dict(weights)
    # 將模型放到設備上
    model.to(device)
    # 調整成驗證模式
    model.eval()

    with torch.inference_mode():
        # 將圖像放入模型中進行預測
        outputs = model(img_tensor.to(device))

        # 再將原始圖像做左右翻轉後再次輸入到模型當中，最後將結果取平均值
        if flip_test:
            flip_tensor = transforms.flip_images(img_tensor)
            flip_outputs = torch.squeeze(
                transforms.flip_back(model(flip_tensor.to(device)), person_info["flip_pairs"]),
            )
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
            outputs = (outputs + flip_outputs) * 0.5

        # 將結果映射回原圖上面
        keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
        # 去除掉原本的batch_size維度
        # keypoints shape [num_kps, 2]，scores shape [num_kps, 1]
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)

        # 將結果畫在原圖上方
        plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=3)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save("test_result.jpg")


if __name__ == '__main__':
    predict_single_person()
