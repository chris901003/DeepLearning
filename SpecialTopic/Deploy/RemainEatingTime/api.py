try:
    import tensorrt
except ImportError:
    print('You will not able use TensorRT to speed up model')
import os
import pickle
import torch
import onnx
import numpy as np
from SpecialTopic.Deploy.RemainEatingTime.RemainEatingTime_M import RemainEatingTimeM
from SpecialTopic.Deploy.RemainEatingTime.utils import load_pretrained, parser_setting, load_encoder_pretrained, \
    load_decoder_pretrained, RemainEatingTimeEncoder, RemainEatingTimeDecoder


def simplify_onnx(onnx_path='RemainEatingTimeM.onnx', output_path='RemainEatingTimeM_Simplify.onnx'):
    try:
        from onnxsim import simplify
    except ImportError:
        raise ImportError('須先安裝onnx simplify才可以使用簡化onnx函數')
    onnx_model = onnx.load(onnx_path)
    model_simplify, check = simplify(onnx_model)
    assert check, 'Simplified ONNX model could not be validated'
    onnx.save(model_simplify, output_path)


def create_onnx_file(model_phi='m', setting_file_path=None, pretrained='pretrained.pth',
                     input_names=('remain_food', 'remain_time'), output_names='predict_time',
                     save_path='RemainTimeM.onnx', dynamic_axes=None, simplify=True,
                     simplify_save_path='RemainEatingTimeM_Simplify.onnx'):
    """ 構建onnx檔案
    Args:
        model_phi: 模型大小
        setting_file_path: 模型構建設定檔案路徑
        pretrained: 訓練權重路徑
        input_names: 輸入資料名稱
        output_names: 輸出資料的名稱
        save_path: 保存onnx格式的路徑
        dynamic_axes: 動態shape設定
        simplify: 是否需要使用簡化onnx
        simplify_save_path: 簡化後的onnx保存路徑
    """
    support_model_phi = {
        'm': {'model_cls': RemainEatingTimeM, 'input_shape': tuple()}
    }
    setting = parser_setting(setting_file_path)
    max_len = setting.get('max_len', None)
    assert max_len is not None, '需提供最大長度才可以構建'
    input_shape = (1, max_len)
    create_model_cfg = support_model_phi.get(model_phi, None)
    assert create_model_cfg is not None, '尚未支持該大小模型'
    create_model_cfg['input_shape'] = input_shape
    model = create_model_cfg['model_cls'](setting_file_path)
    model = load_pretrained(model, pretrained)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    remain = torch.randint(0, 1, (1, max_len)).to(device)
    remain_time = torch.randint(0, 1, (1, max_len)).to(device)
    assert isinstance(input_names, (list, tuple)) and len(input_names), '輸入資料的名稱有錯誤'
    if isinstance(output_names, str):
        output_names = [output_names]
    with torch.no_grad():
        torch.onnx.export(model, (remain, remain_time), save_path, input_names=input_names, output_names=output_names,
                          opset_version=11, dynamic_axes=dynamic_axes)
    if simplify:
        simplify_onnx(save_path, simplify_save_path)


def create_encoder_onnx(embed_dim=32, heads=4, encoder_layers=3, attention_norm=8, mlp_ratio=2, dropout_ratio=0.1,
                        pretrained_path=None, setting_file=None, save_path='RemainEatingTimeEncoder.onnx',
                        input_names='food_remain', output_names=('food_remain_encode', 'food_remain_mask'),
                        dynamic_axes=None, simplify=True,
                        simplify_save_path='RemainEatingTimeEncoder_Simplify.onnx'):
    """ 主要是在構建Encoder部分的Onnx模型格式，詳細參數懶得說了
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    assert pretrained_path is not None and os.path.exists(pretrained_path), '需提供訓練權重資料'
    setting = parser_setting(setting_file)
    max_len = setting.get('max_len', None)
    num_remain_classes = setting.get('num_remain_classes', None)
    remain_pad_val = setting.get('remain_pad_val', None)
    assert max_len is not None, '在setting檔案當中無法獲取max_len資料'
    assert num_remain_classes is not None, '在setting檔案當中無法獲取num_remain_classes資料'
    assert remain_pad_val is not None, '在setting檔案當中無法獲取remain_pad_val資料'
    encoder = RemainEatingTimeEncoder(embed_dim, heads, encoder_layers, attention_norm, mlp_ratio, dropout_ratio,
                                      max_len, remain_pad_val, num_remain_classes)
    load_encoder_pretrained(encoder, pretrained_path)
    encoder = encoder.to(device)
    encoder.eval()
    food_remain = torch.randint(0, 1, (1, max_len)).to(device)
    if isinstance(input_names, str):
        input_names = [input_names]
    if isinstance(output_names, str):
        output_names = [output_names]
    with torch.no_grad():
        torch.onnx.export(encoder, food_remain, save_path, input_names=input_names,
                          output_names=output_names, opset_version=11, dynamic_axes=dynamic_axes)
    if simplify:
        simplify_onnx(save_path, simplify_save_path)


def create_decoder_onnx(embed_dim=32, heads=4, decoder_layers=3, attention_norm=8, mlp_ratio=2, dropout_ratio=0.1,
                        pretrained_path=None, setting_file=None, save_path='RemainEatingTimeDecoder.onnx',
                        input_names=('remain_output', 'food_remain_mask', 'remain_time'),
                        output_names='time_output', dynamic_axes=None, simplify=True,
                        simplify_save_path='RemainEatingTimeDecoder_Simplify.onnx'):
    """ 主要是在構建Decoder部分的Onnx模型格式，詳細參數懶得說了
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    assert pretrained_path is not None and os.path.exists(pretrained_path), '給定的訓練權重不存在'
    setting = parser_setting(setting_file)
    max_len = setting.get('max_len', None)
    time_pad_val = setting.get('time_pad_val', None)
    num_time_classes = setting.get('num_time_classes', None)
    assert max_len is not None and time_pad_val is not None and num_time_classes is not None, 'Setting當中的資料有缺少'
    decoder = RemainEatingTimeDecoder(embed_dim, heads, decoder_layers, attention_norm, mlp_ratio, dropout_ratio,
                                      max_len, time_pad_val, num_time_classes)
    load_decoder_pretrained(decoder, pretrained_path)
    decoder = decoder.to(device)
    decoder.eval()
    food_remain = torch.randn((1, max_len, embed_dim)).to(device)
    food_remain_mask = torch.full((1, 1, max_len, max_len), 1, dtype=torch.bool).to(device)
    time_remain = torch.randint(0, 1, (1, max_len)).to(device)
    if isinstance(input_names, str):
        input_names = [input_names]
    if isinstance(output_names, str):
        output_names = [output_names]
    with torch.no_grad():
        torch.onnx.export(decoder, (food_remain, food_remain_mask, time_remain), save_path, input_names=input_names,
                          output_names=output_names, opset_version=11, dynamic_axes=dynamic_axes)
    if simplify:
        simplify_onnx(save_path, simplify_save_path)


def create_onnx_session(onnx_file='RemainEatingTimeM_Simplify.onnx', gpu='auto'):
    """
    Args:
        onnx_file: onnx檔案路徑位置
        gpu: 是否有使用gpu，默認會自動辨識

    Returns:
        session: onnxruntime執行對象
    """
    try:
        import onnxruntime
    except ImportError:
        raise ImportError('如需要使用onnxruntime進行推理需先安裝onnxruntime')
    assert os.path.exists(onnx_file), '給定的onnx檔案路徑不存在'
    if gpu == 'auto':
        gpu = True if onnxruntime.get_device() == 'GPU' else False
    if not gpu:
        session = onnxruntime.InferenceSession(onnx_file)
    else:
        session = onnxruntime.InferenceSession(onnx_file, providers=['CUDAExecutionProvider'])
    return session


def onnxruntime_detect(onnx_model, food_remain, input_names=('remain_food', 'remain_time'),
                       output_names='predict_time'):
    if isinstance(output_names, str):
        onnx_outputs = [output_names]
    elif isinstance(output_names, (list, tuple)):
        onnx_outputs = output_names
    else:
        raise ValueError('output_names格式錯誤')


if __name__ == '__main__':
    create_decoder_onnx(setting_file='/Users/huanghongyan/Documents/DeepLearning/SpecialTopic/RemainEating'
                                     'Time/train_annotation.pickle',
                        pretrained_path='/Users/huanghongyan/Documents/DeepLearning/SpecialTopic/Remain'
                                        'EatingTime/save/auto_eval.pth')
    # create_onnx_file(setting_file_path='/Users/huanghongyan/Documents/DeepLearning/SpecialTopic/RemainEating'
    #                                     'Time/train_annotation.pickle',
    #                  pretrained='/Users/huanghongyan/Documents/DeepLearning/SpecialTopic/Remain'
    #                             'EatingTime/save/auto_eval.pth')
    # food_remain = [100, 97, 93, 90, 90, 89, 86, 85, 85, 84, 82, 81, 79, 76, 76, 75, 75, 73, 73, 72, 71, 67, 61, 59, 58,
    #                58, 55, 49, 49, 45, 44, 42, 34, 28, 24, 24, 20, 16, 14, 11, 7, 4, 3, 1, 0]
    #
    # session = create_onnx_session()

