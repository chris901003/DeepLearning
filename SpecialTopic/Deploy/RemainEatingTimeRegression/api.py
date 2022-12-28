import tensorrt
import os
import torch
import json
from typing import Union
import numpy as np
from SpecialTopic.Deploy.RemainEatingTimeRegression.RemainEatingTimeRegression import RemainEatingTimeRegressionNet, \
    load_pretrained, get_mock_data, simplify_onnx, parser_setting
from SpecialTopic.Deploy.OnnxToTensorRT.TensorrtBase import TensorrtBase


def create_onnx_file(model_phi='stander', setting='./prepare/setting_0.json', pretrained='./prepare/regression_0.pth',
                     with_simplify=True, input_name='food_remain', output_name='time_remain',
                     onnx_file_name='RemainEatingTimeRegression.onnx',
                     simplify_file_name='RemainEatingTimeRegression_Simplify.onnx'):
    """ 創建onnx檔案，如果有需要可以同步創建onnx_simplify檔案
    Args:
         model_phi: 模型大小
         setting: 模型設定資料，會在生成訓練資料的時候同時生成
         pretrained: 訓練權重位置
         with_simplify: 是否要啟用簡化onnx步驟
         input_name: 輸入的名稱，主要會跟後面步驟有關係
         output_name: 輸出的名稱，主要會跟後面步驟有關係
         onnx_file_name: pytorch輸出的onnx檔案名稱
         simplify_file_name: 透過簡化後的onnx檔案名稱，如果沒有要進行簡化就不需要設定
    """
    support_model_phi = {
        'stander': {'input_size': 32, 'hidden_size': 64, 'num_layers': 2}
    }
    model_cfg = support_model_phi.get(model_phi, None)
    assert model_cfg is not None, f'指定的模型大小{model_phi}不支援，如有需要請自行添加'
    settings = parser_setting(setting)
    model_cfg['remain_time_classes'] = settings['remain_time_padding_value'] + 1
    model = RemainEatingTimeRegressionNet(**model_cfg)
    model = load_pretrained(model, pretrained_path=pretrained)
    mock_food_remain = get_mock_data(settings)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    mock_food_remain = mock_food_remain.to(device)
    input_names = [input_name]
    output_names = [output_name]
    with torch.no_grad():
        torch.onnx.export(model, mock_food_remain, onnx_file_name, input_names=input_names,
                          output_names=output_names, opset_version=11)
    if with_simplify:
        simplify_onnx(onnx_path=onnx_file_name, output_path=simplify_file_name)


def create_onnx_session(onnx_file='RemainEatingTimeRegression_Simplify.onnx', gpu='auto'):
    """ 創建可以用onnx執行的對象
    Args:
        onnx_file: 指定onnx檔案
        gpu: 設定成auto會自動偵測是否在gpu模式下
    """
    try:
        import onnxruntime
    except ImportError:
        raise ImportError('如果需要使用onnxruntime進行推理需要安裝onnxruntime')
    assert os.path.exists(onnx_file), '給定的onnx檔案路徑不存在'
    if gpu == 'auto':
        gpu = True if onnxruntime.get_device() == 'GPU' else False
    if not gpu:
        session = onnxruntime.InferenceSession(onnx_file)
    else:
        session = onnxruntime.InferenceSession(onnx_file, providers=['CUDAExecutionProvider'])
    return session


def transform_food_remain_to_model_input(food_remain, settings):
    """
    Args:
        food_remain: 食物剩餘量
        settings: 模型設定資料，這裡需要是dict格式，避免每次到這裡都需要對檔案進行讀取，浪費時間
    """
    assert isinstance(settings, dict), '傳入到transform food remain to model input的settings需要是dict格式，' \
                                       '這是為了避免效率問題'
    max_length = settings.get('max_length', None)
    remain_start_value = settings.get('remain_start_value', None)
    remain_end_value = settings.get('remain_end_value', None)
    remain_padding_value = settings.get('remain_padding_value', None)
    assert max_length is not None and remain_start_value is not None and remain_end_value is not None and \
           remain_padding_value is not None, '傳入的setting有誤'
    food_remain = np.array(food_remain)
    food_remain = np.append(np.array([remain_start_value]), food_remain)
    food_remain = np.append(food_remain, np.array([remain_end_value]))
    food_remain = np.append(food_remain, np.array([remain_padding_value] * max_length))[:max_length]
    food_remain = np.expand_dims(food_remain, axis=0)
    return food_remain


def onnxruntime_detection_image(onnx_model, food_remain, settings: Union = (str, dict), input_name='food_remain',
                                output_name='time_remain'):
    """ 使用onnxruntime進行推理
    Args:
        onnx_model: onnxruntime模型
        food_remain: 食物剩餘量
        settings: 模型訓練參數，可以傳入dict或是檔案路徑
        input_name: 輸入到onnx的名稱，需要與生成onnx相同名稱
        output_name: 從onnx輸出的名稱，需要與生成onnx相同名稱
    """
    if isinstance(settings, str):
        settings = parser_setting(settings)
    food_remain = transform_food_remain_to_model_input(food_remain, settings)
    food_remain = food_remain.astype(np.longlong)
    onnx_inputs = {input_name: food_remain}
    onnx_outputs = [output_name]
    onnx_preds = onnx_model.run(onnx_outputs, onnx_inputs)[0]
    onnx_preds = np.transpose(onnx_preds, (0, 2, 1))
    prediction = onnx_preds.argmax(axis=1).flatten()
    return prediction


def create_tensorrt_engine(onnx_file_path='RemainEatingTimeRegression_Simplify.onnx', fp16_mode=True, max_batch_size=1,
                           trt_engine_path=None, save_trt_engine_path=None, dynamic_shapes=None,
                           trt_logger_level='VERBOSE'):
    tensorrt_engine = TensorrtBase(onnx_file_path=onnx_file_path, fp16_mode=fp16_mode, max_batch_size=max_batch_size,
                                   dynamic_shapes=dynamic_shapes, save_trt_engine_path=save_trt_engine_path,
                                   trt_engine_path=trt_engine_path, trt_logger_level=trt_logger_level)
    return tensorrt_engine


def tensorrt_engine_detect_remain_time(tensorrt_engine, food_remain, settings: Union = (str, dict),
                                       input_name='food_remain', output_shapes='time_remain'):
    """ 使用tensorrt進行推理
    Args:
        tensorrt_engine: tensorrt推理引擎實力化對象
        food_remain: 食物剩餘量
        settings: 模型設定參數
        input_name: 輸入到onnx的名稱，需要與生成onnx相同名稱
        output_shapes: 從onnx輸出的名稱，需要與生成onnx相同名稱，或是指定的shape(這裡推薦可以用名稱就用名稱，比較方便也比較易懂)
    Returns:
        推理結果
    """
    if isinstance(settings, str):
        settings = parser_setting(settings)
    food_remain = transform_food_remain_to_model_input(food_remain, settings)
    food_remain = food_remain.astype(np.long)
    tensorrt_inputs = {input_name: np.ascontiguousarray(food_remain)}
    tensorrt_preds = tensorrt_engine.inference(input_datas=tensorrt_inputs, output_shapes=[output_shapes])[0]
    tensorrt_preds = np.transpose(tensorrt_preds, (0, 2, 1))
    prediction = tensorrt_preds.argmax(axis=1).flatten()
    return prediction


if __name__ == '__main__':
    # create_onnx_file()
    # session = create_onnx_session()
    setting_path = './prepare/setting_0.json'
    remain = [100, 99, 98, 97, 96, 94, 93, 92, 92, 90, 90, 90, 89, 89, 89, 88, 88, 87, 87, 83, 82, 82, 82, 80, 79, 78,
              76, 76, 76, 75, 75, 74, 74, 72, 72, 69, 68, 67, 66, 66, 65, 64, 64, 62, 62, 60, 60, 59, 56, 55, 55, 55,
              54, 54, 52, 51, 50, 47, 46, 43, 43, 42, 41, 40, 40, 39, 38, 38, 36, 35, 34, 34, 34, 33, 32, 32, 31, 30,
              29, 26, 24, 23, 23, 23, 23, 21, 16, 16, 15, 14, 13, 11, 11, 9, 9, 7, 7, 6, 5]
    # prediction = onnxruntime_detection_image(session, remain, settings=setting_path)
    # print(prediction)
    tensorrt_engine = create_tensorrt_engine(trt_engine_path='RemainEatingTimeRegression_Simplify.trt')
    prediction = tensorrt_engine_detect_remain_time(tensorrt_engine, remain, settings=setting_path)
    print(prediction)
