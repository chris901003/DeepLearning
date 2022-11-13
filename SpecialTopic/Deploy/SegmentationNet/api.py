import tensorrt
import os
import torch
from typing import Union
from SpecialTopic.Deploy.SegmentationNet.utils import load_pretrained
from SpecialTopic.Deploy.SegmentationNet.Models.SegmentationNet_M import SegmentationNetM


def create_onnx_file(model_phi='m', num_classes=3, pretrained='pretrained.pth', input_name='images_seg',
                     output_name='outputs_seg', save_path='SegmentationNetM.onnx',
                     dynamic_axes: Union[str, dict] = 'Default'):
    """ 構建Segformer的onnx模型格式
    Args:
        model_phi: 模型大小
        num_classes: 類別總數
        pretrained: 訓練銓重資料位置
        input_name: 輸入資料的名稱
        output_name: 輸出資料的名稱
        save_path: 保存onnx檔案的位置
        dynamic_axes: 動態shape設定，這裡默認會將圖像的高寬設定成動態shape，如果相要提升效率就將其設定成None
    Returns:
        None，會將onnx檔案直接保存到指定位置
    """
    support_model_phi = {
        'm': {'model_cls': SegmentationNetM, 'input_shape': (1, 3, 512, 512)}
    }
    if dynamic_axes == 'Default':
        dynamic_axes = {input_name: {2: 'image_height', 3: 'image_width'}}
    create_model_cfg = support_model_phi.get(model_phi, None)
    assert create_model_cfg is not None, '尚未支持該大小模型'
    model = create_model_cfg['model_cls'](num_classes=num_classes)
    model = load_pretrained(model, pretrained)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)
    images = torch.randn(*create_model_cfg['input_shape']).to(device)
    input_names = [input_name]
    output_names = [output_name]
    with torch.no_grad():
        torch.onnx.export(model, images, save_path, input_names=input_names, output_names=output_names,
                          opset_version=11, dynamic_axes=dynamic_axes)


def create_onnx_session(onnx_file='SegmentationNetM.onnx', gpu='auto'):
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
