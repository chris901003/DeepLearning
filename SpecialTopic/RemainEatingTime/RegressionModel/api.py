import os
import torch
import json
from typing import Union
import numpy as np
from SpecialTopic.ST.build import build_detector


def parse_setting(setting_path):
    with open(setting_path, 'r') as f:
        return json.load(f)


def init_model(cfg: Union[str, dict] = 'Default', setting_path=None, pretrained=None):
    """ 初始化回歸模型
    Args:
        cfg: 模型設定資料，如果要預設就不用特地指定
        setting_path: 設定相關參數值的檔案
        pretrained: 訓練權重資料
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    assert setting_path is not None and os.path.exists(setting_path), '給定的setting檔案不存在'
    assert pretrained is not None and os.path.exists(pretrained), '需給定預訓練權重位置'
    if isinstance(cfg, str) and cfg == 'Default':
        cfg = {
            'type': 'RemainTimeRegression',
            'input_size': 32,
            'hidden_size': 64,
            'num_layers': 2
        }
    elif isinstance(cfg, dict):
        pass
    else:
        raise ValueError('cfg只能是"Default"或是dict格式')
    settings = parse_setting(setting_path)
    cfg['remain_time_classes'] = settings['remain_time_padding_value'] + 1
    model = build_detector(cfg)
    model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    model = model.to(device)
    model.eval()
    model.settings = settings
    return model


def predict_remain_time(model, remain, device: Union[str, torch.device] = 'Default'):
    """ 預測此時刻所需的剩餘時間
    Args:
        model: 回歸模型
        remain: 剩餘量資料，只需要給正常的就好不需要加工過
        device: 運行設備
    """
    if isinstance(device, str) and device == 'Default':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        assert isinstance(device, torch.device), 'device型別錯誤'
    settings = model.settings
    max_length = settings['max_length']
    remain_start_value = settings['remain_start_value']
    remain_end_value = settings['remain_end_value']
    remain_padding_value = settings['remain_padding_value']
    remain = np.array(remain)
    remain = np.append(np.array([remain_start_value]), remain)
    remain = np.append(remain, np.array([remain_end_value]))
    remain = np.append(remain, np.array([remain_padding_value] * max_length))[:max_length]
    remain = torch.Tensor(remain).to(torch.long).to(device).unsqueeze(dim=0)
    with torch.no_grad():
        prediction, _, _ = model(remain)
    prediction = prediction.permute((0, 2, 1))
    prediction = prediction.argmax(dim=1).cpu().detach().numpy().flatten()
    return prediction


def test():
    model = init_model(setting_path='setting.json', pretrained='regression_model.pth')
    remain = [100, 99, 98, 97, 96, 94, 93, 92, 92, 90, 90, 90, 89, 89, 89, 88, 88, 87, 87, 83, 82, 82, 82, 80, 79, 78,
              76, 76, 76, 75, 75, 74, 74, 72, 72, 69, 68, 67, 66, 66, 65, 64, 64, 62, 62, 60, 60, 59, 56, 55, 55, 55,
              54, 54, 52, 51, 50, 47, 46, 43, 43, 42, 41, 40, 40, 39, 38, 38, 36, 35, 34, 34, 34, 33, 32, 32, 31, 30,
              29, 26, 24, 23, 23, 23, 23, 21, 16, 16, 15, 14, 13, 11, 11, 9, 9, 7, 7, 6, 5]
    prediction = predict_remain_time(model, remain)
    print(prediction)


if __name__ == '__main__':
    test()
