import torch
import os
import json
import pickle
import numpy as np
from SpecialTopic.ST.dataset.utils import Compose
from SpecialTopic.ST.build import build_detector


def load_json(setting_file_path):
    with open(setting_file_path, 'r') as f:
        results = json.load(f)
    return results


def load_pickle(setting_file_path):
    with open(setting_file_path, 'rb') as f:
        results = pickle.load(f)
    return results


def init_model(phi, setting_file_path, pretrained):
    if not os.path.exists(pretrained):
        pretrained = 'none'
        print('未載入訓練權重出來的結果會是無效的')
    assert os.path.exists(setting_file_path), '給定的參數檔案資料不存在'
    if os.path.splitext(setting_file_path)[1] == '.py':
        # import python文件時需要將文件方在當前目錄下，並且直接給檔名就可以了
        # 並且資料的字典變數名稱需要是setting
        setting_info = __import__(os.path.splitext(setting_file_path)[0])
        setting = setting_info.setting
    elif os.path.splitext(setting_file_path)[1] == '.json':
        setting = load_json(setting_file_path)
    elif os.path.splitext(setting_file_path)[1] == '.pickle':
        setting = load_pickle(setting_file_path)
    else:
        raise NotImplementedError('尚未實作')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    variables = {
        'max_len': setting['max_len'],
        'remain_to_index': setting['remain_to_index'], 'time_to_index': setting['time_to_index'],
        'remain_pad_val': setting['remain_pad_val'], 'time_pad_val': setting['time_pad_val'],
        'remain_SOS_val': setting['remain_SOS_val'], 'time_SOS_val': setting['time_SOS_val'],
        'remain_EOS_val': setting['remain_EOS_val'], 'time_EOS_val': setting['time_EOS_val']
    }
    pipeline_cfg = [
        {'type': 'FormatRemainEatingData', 'variables': variables},
        {'type': 'Collect', 'keys': ['food_remain_data']}
    ]
    pipeline = Compose(pipeline_cfg)
    model_cfg = {
        'type': 'RemainEatingTime',
        'phi': phi,
        'num_remain_classes': setting['num_remain_classes'],
        'num_time_classes': setting['num_time_classes'],
        'max_len': setting['max_len'],
        'remain_pad_val': setting['remain_pad_val'],
        'time_pad_val': setting['time_pad_val'],
        'pretrained': pretrained
    }
    model = build_detector(model_cfg)
    model = model.to(device)
    model.variables = variables
    model.pipeline = pipeline
    return model


def detect_single_remain_time(model, food_remain, pipeline=None, device='auto'):
    """
    Args:
        model: 模型本身
        food_remain: 食物剩餘量列表
        pipeline: 資料處理列表，如果在model已經有了就可以不用傳入，這裡也可以直接傳入Compose後的pipeline
        device: 使用的設備
    """
    model = model.eval()
    if device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if pipeline is not None:
        if isinstance(pipeline, list):
            pipeline = Compose(pipeline)
        elif isinstance(pipeline, Compose):
            pass
        else:
            raise ValueError('傳入的pipeline有錯誤')
    assert hasattr(model, 'pipeline') or pipeline is not None, '至少需要提供一種資料處理方式'
    data = dict(food_remain=food_remain, time_remain=np.array([]))
    if pipeline is not None:
        data = pipeline(data)
    else:
        data = model.pipeline(data)
    food_remain = torch.LongTensor(data['food_remain_data'])
    assert hasattr(model, 'variables'), '在model當中需要有variables來儲存參數，理論上在init_model時就會放入'
    model = model.to(device)
    with torch.no_grad():
        time_remain = [model.variables['time_SOS_val']] + [model.variables['time_pad_val']] * \
                      (model.variables['max_len'] - 1)
        time_remain = time_remain[:model.variables['max_len']]
        time_remain = torch.LongTensor(time_remain)
        time_remain = time_remain.to(device)
        food_remain = food_remain.to(device)
        food_remain = food_remain.unsqueeze(dim=0)
        time_remain = time_remain.unsqueeze(dim=0)
        food_remain_mask = model.backbone.mask_pad(food_remain, model.variables['remain_pad_val'],
                                                   model.variables['max_len'])
        food_remain = model.backbone.embed_remain(food_remain)
        food_remain = model.backbone.encoder(food_remain, food_remain_mask)
        for i in range(model.variables['max_len'] - 1):
            y = time_remain
            mask_tril_y = model.backbone.mask_tril(y, model.variables['time_pad_val'], model.variables['max_len'])
            y = model.backbone.embed_time(y)
            y = model.backbone.decoder(food_remain, y, food_remain_mask, mask_tril_y)
            out = model.cls_head.cls_fc(y)
            out = out[:, i, :]
            out = out.argmax(dim=1).detach()
            time_remain[:, i + 1] = out
        time_remain = time_remain.tolist()[0]
    return time_remain


def test():
    model = init_model(phi='m', setting_file_path='./train_annotation.pickle', pretrained='./save/auto_eval.pth')
    food_remain = [100, 97, 93, 90, 90, 89, 86, 85, 85, 84, 82, 81, 79, 76, 76, 75, 75, 73, 73, 72, 71, 67, 61, 59, 58,
                   58, 55, 49, 49, 45, 44, 42, 34, 28, 24, 24, 20, 16, 14, 11, 7, 4, 3, 1, 0]
    for idx in range(len(food_remain)):
        cur_food_remain = food_remain[:idx + 1]
        results = detect_single_remain_time(model, cur_food_remain)
        EOS_index = results.index(model.variables['time_EOS_val'])
        results = results[:EOS_index]
        result = results[idx] if idx < len(results) else results[-1]
        print(f'Current Time: {idx}, Remain Time: {result}')
        print(results)
    ans = [idx for idx in range(len(food_remain))]
    ans = ans[::-1]
    print(ans)
    detect_single_remain_time(model, food_remain)


if __name__ == '__main__':
    print('Testing Remain eating time API')
    test()
