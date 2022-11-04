import torch
from utils import load_pretrained
from YoloxObjectDetection_L import YoloxObjectDetection as YoloxObjectDetectionL


def create_onnx(model_phi='l', num_classes=9, pretrained='/Users/huanghongyan/Downloads/900_yolox_850.25.pth',
                input_name='images', output_name='yolox_outputs', save_path='YoloxObjectDetectionL.onnx',
                dynamic_axes=None):
    """ 構建Yolox object detection的onnx模型格式
    Args:
        model_phi: 模型大小
        num_classes: 分類類別數量
        pretrained: 預訓練權重位置
        input_name: 輸入資料名稱
        output_name: 輸出資料名稱
        save_path: onnx檔案保存位置
        dynamic_axes: 動態維度資料
    """
    support_model_phi = {
        'l': {'model_cls': YoloxObjectDetectionL, 'input_shape': (1, 3, 640, 640)}
    }
    create_model_cfg = support_model_phi.get(model_phi, None)
    assert create_model_cfg is not None, '尚未支援該模型大小'
    model = create_model_cfg['model_cls'](num_classes=num_classes)
    model = load_pretrained(model, pretrained)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model = model.to(device)
    images = torch.randn(*create_model_cfg['input_shape'])
    input_names = [input_name]
    output_names = [output_name]
    with torch.no_grad():
        model_script = torch.jit.script(model)
        torch.onnx.export(model_script, images, save_path, input_names=input_names,
                          output_names=output_names, opset_version=11, dynamic_axes=dynamic_axes)


if __name__ == '__main__':
    print('Testing Deploy Yolox object detection')
    create_onnx()
