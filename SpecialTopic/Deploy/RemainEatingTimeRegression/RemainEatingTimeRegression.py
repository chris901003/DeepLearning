import argparse
import json
import os
import torch
import numpy as np
from torch import nn


class RemainEatingTimeRegressionNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, remain_time_classes):
        super(RemainEatingTimeRegressionNet, self).__init__()
        self.remain_embed = nn.Embedding(104, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.cls = nn.Linear(hidden_size, remain_time_classes)

    def forward(self, x):
        x = self.remain_embed(x)
        r_out, (h_state, c_state) = self.lstm(x)
        outs = list()
        for time_step in range(r_out.size(1)):
            outs.append(self.cls(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state, c_state


def parse_args():
    parser = argparse.ArgumentParser()
    # 設定檔的路徑
    parser.add_argument('--setting-path', type=str, default='./prepare/setting_0.json')
    # 預訓練權重路徑
    parser.add_argument('--pretrained-path', type=str, default='./prepare/regression_0.pth')
    # 是否需要將onnx進行簡化，通常都需要不然會有無法轉成onnx的問題，經過簡化後可以有效解決函數不支援問題
    parser.add_argument('--with-simplify', action='store_false')

    # 以下如果在訓練時有調整才需要更動
    parser.add_argument('--input_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)

    args = parser.parse_args()
    return args


def parser_setting(setting_path):
    with open(setting_path, 'r') as f:
        return json.load(f)


def load_pretrained(model, pretrained_path):
    assert os.path.exists(pretrained_path), '給定的權重路徑錯誤'
    print(f'Load weights from: {pretrained_path}')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    if 'model_weight' in pretrained_dict:
        pretrained_dict = pretrained_dict['model_weight']
    load_key, no_load_key, temp_dict = list(), list(), dict()
    for k, v in pretrained_dict.items():
        if k in model_dict.keys() and np.shape(v) == np.shape(model_dict[k]):
            temp_dict[k] = v
            load_key.append(k)
        else:
            no_load_key.append(k)
    model_dict.update(temp_dict)
    model.load_state_dict(model_dict)
    print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
    print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
    assert len(no_load_key) == 0, '給定的預訓練權重與模型不匹配'
    return model


def get_mock_data(settings):
    max_length = settings['max_length']
    remain_start_value = settings['remain_start_value']
    remain_end_value = settings['remain_end_value']
    remain_padding_value = settings['remain_padding_value']
    remain = np.array([remain_start_value, 100, remain_end_value])
    remain = np.append(remain, np.array([remain_padding_value] * max_length))[:max_length]
    remain = torch.Tensor(remain).to(torch.long).unsqueeze(dim=0)
    return remain


def simplify_onnx():
    from onnxsim import simplify
    import onnx
    onnx_path = 'RemainEatingTimeRegression.onnx'
    output_path = 'RemainEatingTimeRegression_Simplify.onnx'
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    assert check, 'Simplified ONNX model could not be validated'
    onnx.save(model_simp, output_path)
    print('Finished exporting simplify onnx')


def main():
    args = parse_args()
    setting_path = args.setting_path
    pretrained_path = args.pretrained_path
    with_simplify = args.with_simplify
    settings = parser_setting(setting_path)
    model_cfg = {
        'input_size': args.input_size,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'remain_time_classes': settings['remain_time_padding_value'] + 1
    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = RemainEatingTimeRegressionNet(**model_cfg)
    model = load_pretrained(model, pretrained_path)
    mock_input = get_mock_data(settings)
    model = model.to(device)
    mock_input = mock_input.to(device)
    input_names = ['food_remain']
    output_names = ['time_remain']
    with torch.no_grad():
        torch.onnx.export(model, mock_input, 'RemainEatingTimeRegression.onnx', input_names=input_names,
                          output_names=output_names, opset_version=11)
    if with_simplify:
        simplify_onnx()


def test_predict_remain_time(session, remain, settings):
    max_length = settings['max_length']
    remain_start_value = settings['remain_start_value']
    remain_end_value = settings['remain_end_value']
    remain_padding_value = settings['remain_padding_value']
    remain = np.array(remain)
    remain = np.append(np.array([remain_start_value]), remain)
    remain = np.append(remain, np.array([remain_end_value]))
    remain = np.append(remain, np.array([remain_padding_value] * max_length))[:max_length]
    remain = torch.Tensor(remain).to(torch.long).unsqueeze(dim=0)
    remain = remain.numpy()
    onnx_inputs = {'food_remain': remain}
    onnx_outputs = ['time_remain']
    onnx_preds = session.run(onnx_outputs, onnx_inputs)[0]
    onnx_preds = torch.from_numpy(onnx_preds)
    prediction = onnx_preds.permute((0, 2, 1))
    prediction = prediction.argmax(dim=1).cpu().detach().numpy().flatten()
    return prediction


def test():
    # 已通過測試，沒有問題結果與使用pytorch預測相同
    import onnxruntime
    settings = parser_setting('./prepare/setting_0.json')
    session = onnxruntime.InferenceSession('RemainEatingTimeRegression_Simplify.onnx',
                                           providers=['CUDAExecutionProvider'])
    remain = [100, 99, 98, 97, 96, 94, 93, 92, 92, 90, 90, 90, 89, 89, 89, 88, 88, 87, 87, 83, 82, 82, 82, 80, 79, 78,
              76, 76, 76, 75, 75, 74, 74, 72, 72, 69, 68, 67, 66, 66, 65, 64, 64, 62, 62, 60, 60, 59, 56, 55, 55, 55,
              54, 54, 52, 51, 50, 47, 46, 43, 43, 42, 41, 40, 40, 39, 38, 38, 36, 35, 34, 34, 34, 33, 32, 32, 31, 30,
              29, 26, 24, 23, 23, 23, 23, 21, 16, 16, 15, 14, 13, 11, 11, 9, 9, 7, 7, 6, 5]
    prediction = test_predict_remain_time(session, remain, settings)
    print(prediction)


if __name__ == '__main__':
    print('Testing Remain Eating Time Regression transfer to onnx model')
    main()
