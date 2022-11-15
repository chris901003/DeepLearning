try:
    import tensorrt
except ImportError:
    print('You will not able use TensorRT to speed up model')
import os
import torch
import cv2
import numpy as np
from typing import Union
import time
from onnxsim import simplify
import onnx
from SpecialTopic.Deploy.OnnxToTensorRT.TensorrtBase import TensorrtBase
from SpecialTopic.ST.dataset.utils import Compose
from SpecialTopic.ST.dataset.config.segmentation_classes_platte import FoodAndNotFood
from SpecialTopic.SegmentationNet.api import image_draw
from SpecialTopic.Deploy.SegmentationNet.utils import load_pretrained
from SpecialTopic.Deploy.SegmentationNet.Models.SegmentationNet_M import SegmentationNetM
from SpecialTopic.Deploy.SegmentationNet.Models.SegmentationNet_Nano import SegmentationNetNano


def simplify_onnx(onnx_path='SegmentationNetM.onnx', output_path='SegmentationNetM_Simplify.onnx'):
    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, output_path)


def create_onnx_file(model_phi='m', num_classes=3, pretrained='pretrained.pth', input_name='images_seg',
                     output_name='outputs_seg', save_path='SegmentationNetM.onnx',
                     dynamic_axes: Union[dict] = None):
    """ 構建Segformer的onnx模型格式
    Args:
        model_phi: 模型大小
        num_classes: 類別總數
        pretrained: 訓練銓重資料位置
        input_name: 輸入資料的名稱
        output_name: 輸出資料的名稱
        save_path: 保存onnx檔案的位置
        dynamic_axes: 動態shape設定，這裡只提供對於batch方面的動態shape，高寬方面的因為有插值運算目前無法處理
    Returns:
        None，會將onnx檔案直接保存到指定位置
    """
    support_model_phi = {
        'nano': {'model_cls': SegmentationNetNano, 'input_shape': (1, 3, 512, 512)},
        'm': {'model_cls': SegmentationNetM, 'input_shape': (1, 3, 512, 512)}
    }
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


def onnxruntime_detect_image(onnx_model, image_info, threshold=0.7, opacity=0.5, pipeline=None, input_name='images_seg',
                             output_name='outputs_seg', input_size=(512, 512),
                             category_info: Union[str, dict] = 'FoodAndNotFood', with_class=True, with_draw=True):
    """ 使用onnxruntime對onnx格式模型進行推理，只推薦在測試onnx格式模型時使用
    Args:
        onnx_model: onnx實例化模型對象
        image_info: 圖像資料，這裡可以傳入圖像路徑
        threshold: 置信度閾值
        opacity: 混和圖像時標註圖像的透明度
        pipeline: 圖像處理流程，這裡需要輸入的會是Compose類型，如果沒有傳入會自動使用默認的圖像處理流
        input_name: 輸入到網路時的名稱，需要與創建onnx格式模型時相同
        output_name: 輸出網路的名稱，需要與創建onnx格式模型時相同
        input_size: 最終輸入到模型當中的圖像大小，如果有自定義pipeline就部會啟用，需自行在pipeline中調整圖像大小
        category_info: 標註圖的著色依據，可以支持指定字典的名稱或是傳入自定義的字典
            dict = (CLASSES=[], PALETTE=[])
        with_class: 在畫圖時是否需要將有出現的類別顏色對應的類別名稱顯示
        with_draw: 是否需要輸出混和標註圖與原始圖像
    """
    if isinstance(image_info, str):
        image = cv2.imread(image_info)
    elif isinstance(image_info, torch.Tensor):
        image = image_info.cpu().numpy()
    elif isinstance(image_info, np.ndarray):
        image = image_info
    else:
        raise NotImplementedError('目前對該圖像格式並不支援')
    origin_image = image.copy()
    assert isinstance(pipeline, Compose) or (pipeline is None), '目前的圖像預處理流只支援Compose對象，或是直接放空就會使用默認'
    # 目前有支援的類別色盤資料
    support_category_cls = {
        'FoodAndNotFood': FoodAndNotFood
    }
    if isinstance(category_info, dict):
        CLASSES = category_info.get('CLASSES', None)
        PALETTE = category_info.get('PALETTE', None)
    elif isinstance(category_info, str):
        category_cls = support_category_cls.get(category_info, None)
        assert category_cls is not None, f'指定的{category_cls}沒有對應的色盤資料'
        CLASSES = category_cls.get('CLASSES', None)
        PALETTE = category_cls.get('PALETTE', None)
    else:
        raise ValueError('指定標註類別時，只支援使用字典傳入或是指定已有的字典名稱')
    assert CLASSES is not None and PALETTE is not None, '無法獲取對應的調色盤'
    if pipeline is not None:
        # 如果有傳入pipeline就會使用pipeline的圖像處理流，但是需要注意輸出的圖像大小要符合模型的輸入大小
        data = dict(img=image)
        data = pipeline(data)
        image = data['img'][0].unsqueeze(dim=0)
        image = image.cpu().numpy()
    else:
        # 如果沒有傳入pipeline就會使用默認簡單的圖像處理
        image = cv2.resize(image, input_size).astype(np.float32)
        image -= np.array((123.675, 116.28, 103.53))
        image /= np.array((58.395, 57.12, 57.375))
        image = np.expand_dims(image[..., ::-1].transpose(2, 0, 1), axis=0)
    onnx_inputs = {input_name: image}
    onnx_outputs = [output_name]
    onnx_preds = onnx_model.run(onnx_outputs, onnx_inputs)[0]
    onnx_preds = onnx_preds[0].transpose(1, 2, 0)
    seg_pred = cv2.resize(onnx_preds, image.shape[2:][::-1], interpolation=cv2.INTER_LINEAR)
    seg_pred = cv2.resize(seg_pred, origin_image.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    seg_pred = seg_pred.transpose(2, 0, 1)
    seg_pred -= np.max(seg_pred, axis=0, keepdims=True)
    seg_pred = np.exp(seg_pred) / np.sum(np.exp(seg_pred), axis=0, keepdims=True)
    mask = (seg_pred > threshold)
    seg_pred = np.argmax(seg_pred, axis=0)
    if not with_draw:
        return seg_pred
    draw_image_mix, draw_image = image_draw(origin_image, seg_pred, palette=PALETTE, classes=CLASSES, opacity=opacity,
                                            with_class=with_class, mask=mask)
    return draw_image_mix, draw_image


def create_tensorrt_engine(onnx_file_path=None, fp16_mode=True, max_batch_size=1, trt_engine_path=None,
                           save_trt_engine_path=None, dynamic_shapes=None, trt_logger_level='VERBOSE'):
    """ 生成TensorRT推理引擎，包裝後的實例化對象
    Args:
        onnx_file_path: onnx檔案路徑
        fp16_mode: 是否使用fp16模式，開啟後可以提升推理速度但是精準度會下降
        max_batch_size: 最大batch資料，如果有使用動態batch這裡可以隨意填，如果是靜態batch就需要寫上
        trt_engine_path: 如果已經有經過序列化保存的TensorRT引擎資料就直接提供，可以透過反序列化直接實例化對象
        save_trt_engine_path: 如果有需要將TensorRT引擎序列化後保存，提供下次使用可以指定路徑
        dynamic_shapes: 如果有設定動態輸入資料就需要到這裡指定
            dict = {
                'input_name': (min_shape, usual_shape, max_shape)
            Ex: 'image1': ((1, 3, 512, 512), (2, 3, 512, 512), (3, 3, 512, 512))
            }
            如果有多組輸入都是變動shape就都放在dict當中，這裡的動態只支援batch方面的，否則會報錯
        trt_logger_level: TensorRT構建以及使用時的logger等級
    Return:
        TensorRT Engine包裝後的對象
    """
    tensorrt_engine = TensorrtBase(onnx_file_path=onnx_file_path, fp16_mode=fp16_mode, max_batch_size=max_batch_size,
                                   dynamic_shapes=dynamic_shapes, save_trt_engine_path=save_trt_engine_path,
                                   trt_engine_path=trt_engine_path, trt_logger_level=trt_logger_level)
    return tensorrt_engine


def tensorrt_engine_detect_image(tensorrt_engine, image_info, threshold=0.7, opacity=0.5, pipeline=None,
                                 input_name='images_seg', output_shapes: Union[str, list] = 'outputs_seg',
                                 input_size=(512, 512), category_info: Union[str, dict] = 'FoodAndNotFood',
                                 with_class=True, with_draw=True, using_dynamic_shape=False):
    if isinstance(image_info, str):
        image = cv2.imread(image_info)
    elif isinstance(image_info, torch.Tensor):
        image = image_info.cpu().numpy()
    elif isinstance(image_info, np.ndarray):
        image = image_info
    else:
        raise NotImplementedError('目前對該圖像格式並不支援')
    origin_image = image.copy()
    support_category_cls = {
        'FoodAndNotFood': FoodAndNotFood
    }
    if isinstance(category_info, dict):
        CLASSES = category_info.get('CLASSES', None)
        PALETTE = category_info.get('PALETTE', None)
    elif isinstance(category_info, str):
        category_cls = support_category_cls.get(category_info, None)
        assert category_cls is not None, f'指定的{category_cls}沒有對應的色盤資料'
        CLASSES = category_cls.get('CLASSES', None)
        PALETTE = category_cls.get('PALETTE', None)
    else:
        raise ValueError('指定標註類別時，只支援使用字典傳入或是指定已有的字典名稱')
    assert CLASSES is not None and PALETTE is not None, '無法獲取對應的調色盤'
    if pipeline is not None:
        # 如果有傳入pipeline就會使用pipeline的圖像處理流，但是需要注意輸出的圖像大小要符合模型的輸入大小
        data = dict(img=image)
        data = pipeline(data)
        image = data['img'][0].unsqueeze(dim=0)
        image = image.cpu().numpy()
    else:
        # 如果沒有傳入pipeline就會使用默認簡單的圖像處理
        image = cv2.resize(image, input_size).astype(np.float32)
        image -= np.array((123.675, 116.28, 103.53))
        image /= np.array((58.395, 57.12, 57.375))
        image = np.expand_dims(image[..., ::-1].transpose(2, 0, 1), axis=0)
    tensorrt_inputs = {input_name: np.ascontiguousarray(image).astype(np.float32)}
    if not isinstance(output_shapes, list):
        output_shapes = [output_shapes]
    tensorrt_preds = tensorrt_engine.inference(input_datas=tensorrt_inputs, output_shapes=output_shapes,
                                               dynamic_shape=using_dynamic_shape)
    tensorrt_preds = tensorrt_preds[0]
    onnx_preds = tensorrt_preds[0].transpose(1, 2, 0)
    seg_pred = cv2.resize(onnx_preds, image.shape[2:][::-1], interpolation=cv2.INTER_LINEAR)
    seg_pred = cv2.resize(seg_pred, origin_image.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
    seg_pred = seg_pred.transpose(2, 0, 1)
    seg_pred -= np.max(seg_pred, axis=0, keepdims=True)
    seg_pred = np.exp(seg_pred) / np.sum(np.exp(seg_pred), axis=0, keepdims=True)
    mask = (seg_pred > threshold)
    seg_pred = np.argmax(seg_pred, axis=0)
    if not with_draw:
        return seg_pred
    draw_image_mix, draw_image = image_draw(origin_image, seg_pred, palette=PALETTE, classes=CLASSES, opacity=opacity,
                                            with_class=with_class, mask=mask)
    return draw_image_mix, draw_image, seg_pred


def test():
    create_onnx_file(model_phi='nano',
                     pretrained=r'C:\DeepLearning\SpecialTopic\SegmentationNet\checkpoint\1115_0_mit_b0_eval.pth',
                     save_path='SegmentationNetNano.onnx')
    simplify_onnx(onnx_path='SegmentationNetNano.onnx', output_path='SegmentationNetNano_Simplify.onnx')
    session = create_onnx_session(onnx_file='SegmentationNetNano_Simplify.onnx')
    image_path = r'C:\Dataset\SegmentationFoodRemain\Donburi\images\training\1.jpg'
    draw_image_mix, draw_image = onnxruntime_detect_image(onnx_model=session, image_info=image_path)
    from PIL import Image
    image = Image.fromarray(cv2.cvtColor(draw_image_mix, cv2.COLOR_BGR2RGB))
    image.show()
    tensorrt_engine = create_tensorrt_engine(onnx_file_path='SegmentationNetNano_Simplify.onnx', fp16_mode=True,
                                             save_trt_engine_path='SegmentationNetNano.trt',
                                             trt_engine_path='SegmentationNetNano.trt')
    draw_image_mix, draw_image = tensorrt_engine_detect_image(tensorrt_engine, image_path)
    image = Image.fromarray(cv2.cvtColor(draw_image_mix, cv2.COLOR_BGR2RGB))
    image.show()


if __name__ == '__main__':
    test()
