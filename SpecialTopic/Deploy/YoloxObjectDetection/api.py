import tensorrt
import torch
from PIL import Image
import numpy as np
import cv2
import PIL
import onnxruntime
import os
from SpecialTopic.Deploy.YoloxObjectDetection.utils import load_pretrained
from SpecialTopic.Deploy.YoloxObjectDetection.YoloxObjectDetection_L import \
    YoloxObjectDetection as YoloxObjectDetectionL
from SpecialTopic.Deploy.OnnxToTensorRT.TensorrtBase import TensorrtBase
from SpecialTopic.YoloxObjectDetection.utils import resize_image, cvtColor, preprocess_input, decode_outputs, \
    non_max_suppression


def create_onnx_file(model_phi='l', num_classes=9, pretrained='/Users/huanghongyan/Downloads/900_yolox_850.25.pth',
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
    Returns:
        None，會直接將onnx檔案保存到指定位置
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
    images = torch.randn(*create_model_cfg['input_shape']).to(device)
    input_names = [input_name]
    output_names = [output_name]
    with torch.no_grad():
        # model_script = torch.jit.script(model)
        torch.onnx.export(model, images, save_path, input_names=input_names,
                          output_names=output_names, opset_version=11, dynamic_axes=dynamic_axes)


def create_onnx_session(onnx_file='YoloxObjectDetectionL.onnx', gpu='auto'):
    """ 創建一個onnxruntime對象
    Args:
        onnx_file: onnx檔案路徑
        gpu: 使用的onnxruntime是否為gpu版本，如果是使用auto就會自動查看
    Return:
        實例化的onnxruntime對象，可以執行的onnx
    """
    assert os.path.exists(onnx_file), '給定的onnx檔案路徑不存在'
    if gpu == 'auto':
        gpu = True if onnxruntime.get_device() == 'GPU' else False
    if not gpu:
        session = onnxruntime.InferenceSession(onnx_file)
    else:
        session = onnxruntime.InferenceSession(onnx_file, providers=['CUDAExecutionProvider'])
    return session


def onnxruntime_detect_image(onnx_model, image, input_shape='Default', num_classes=9,
                             confidence=0.5, nms_iou=0.3, keep_ratio=True,
                             input_name='images', output_name='yolox_outputs'):
    """ 使用onnxruntime執行onnx格式的模型，這裡主要提供測試使用
    Args:
        onnx_model: onnx類型的模型
        image: 圖像資料
        input_shape: 輸入的圖像大小，如果是Default就會是[640, 640]
        num_classes: 類別數量
        confidence: 置信度閾值
        nms_iou: nms閾值
        keep_ratio: 輸入圖像處理時是否需要保持圖像高寬比
        input_name: 輸入到onnx模型的資料名稱，這裡需要與生成onnx資料時相同
        output_name: 從onnx模型輸出的資料名稱，這裡需要與生成onnx資料時相同
    Return:
        top_label: 每個目標的類別
        top_con: 置信度分數
        top_boxes: 目標匡座標位置
    """
    if isinstance(image, str):
        image = Image.open(image)
    elif type(image) is np.ndarray:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        pass
    else:
        raise ValueError('傳入的圖像資料需要是圖像路徑或是已經是ndarray或是PIL格式')
    if input_shape == 'Default':
        input_shape = [640, 640]
    image_shape = np.array(np.shape(image)[0:2])
    image = cvtColor(image)
    image_data = resize_image(image, input_shape, keep_ratio=keep_ratio)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    onnx_inputs = {input_name: image_data}
    onnx_outputs = [output_name]
    onnx_preds = onnx_model.run(onnx_outputs, onnx_inputs)[0]
    outputs = [torch.from_numpy(pred) for pred in onnx_preds]
    outputs = decode_outputs(outputs, input_shape)
    results = non_max_suppression(outputs, num_classes, input_shape, image_shape, keep_ratio, conf_thres=confidence,
                                  nms_thres=nms_iou)
    top_label = np.array(results[0][:, 6], dtype='int32').tolist()
    top_conf = results[0][:, 4] * results[0][:, 5].tolist()
    top_boxes = results[0][:, :4].tolist()
    return top_label, top_conf, top_boxes


def create_tensorrt_engine(onnx_file_path=None, fp16_mode=True, max_batch_size=1, trt_engine_path=None,
                           save_trt_engine_path=None, dynamic_shapes=None, trt_logger_level='INTERNAL_ERROR'):
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
            Ex: 'image1': ((1, 3, 224, 224), (2, 3, 300, 300), (3, 3, 512, 512))
            }
            如果有多組輸入都是變動shape就都放在dict當中
        trt_logger_level: TensorRT構建以及使用時的logger等級
    Return:
        TensorRT Engine包裝後的對象
    """
    tensorrt_engine = TensorrtBase(onnx_file_path=onnx_file_path, fp16_mode=fp16_mode, max_batch_size=max_batch_size,
                                   dynamic_shapes=dynamic_shapes, save_trt_engine_path=save_trt_engine_path,
                                   trt_engine_path=trt_engine_path, trt_logger_level=trt_logger_level)
    return tensorrt_engine


def tensorrt_engine_detect_image(tensorrt_engine, image, input_shape='Default', num_classes=9,
                                 confidence=0.5, nms_iou=0.3, keep_ratio=True,
                                 input_name='images', output_shapes='Default', using_dynamic_shape=False):
    """ TensorRT進行一次圖像檢測
    Args:
        tensorrt_engine: TensorRT引擎對象
        image: 圖像資料
        input_shape: 輸入到網路的圖像大小，Default = [640, 640]
        num_classes: 分類類別數
        confidence: 置信度
        nms_iou: nms閾值
        keep_ratio: 原始圖像縮放時，是否需要保存高寬比
        input_name: 輸入資料的名稱
        output_shapes: 輸出資料的shape，主要是從tensorrt中的輸出都會是一維的，需要透過reshape還原
            [shape1, shape2]
            有幾個輸出就需要提供多少種shape，這裡排放順序就會是輸出資料的順序，無法進行指定，所以要確定onnx建成時的輸出順序
            如果使用Default就會默認Yolox預定的輸出shape
        using_dynamic_shape: 如果有使用到動態shape這裡需要設定成True，否則無法告知引擎資料大小
    Return:
        top_label: 每個目標的類別
        top_con: 置信度分數
        top_boxes: 目標匡座標位置
    """
    if input_shape == 'Default':
        input_shape = [640, 640]
    if output_shapes == 'Default':
        output_shapes = [(1, 14, 80, 80), (1, 14, 40, 40), (1, 14, 20, 20)]
    if isinstance(image, str):
        image = Image.open(image)
    elif type(image) is np.ndarray:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
        pass
    else:
        raise ValueError('傳入的圖像資料需要是圖像路徑或是已經是ndarray或是PIL格式')
    image_shape = np.array(np.shape(image)[0:2])
    image = cvtColor(image)
    image_data = resize_image(image, input_shape, keep_ratio=keep_ratio)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    tensorrt_inputs = {input_name: np.ascontiguousarray(image_data)}
    tensorrt_preds = tensorrt_engine.inference(input_datas=tensorrt_inputs, output_shapes=output_shapes,
                                               dynamic_shape=using_dynamic_shape)
    outputs = [torch.from_numpy(pred) for pred in tensorrt_preds]
    outputs = decode_outputs(outputs, input_shape)
    results = non_max_suppression(outputs, num_classes, input_shape, image_shape, keep_ratio, conf_thres=confidence,
                                  nms_thres=nms_iou)
    top_label = np.array(results[0][:, 6], dtype='int32').tolist()
    top_conf = results[0][:, 4] * results[0][:, 5].tolist()
    top_boxes = results[0][:, :4].tolist()
    return top_label, top_conf, top_boxes


if __name__ == '__main__':
    print('Testing Deploy Yolox object detection')
    create_onnx_file(pretrained=r'C:\Checkpoint\YoloxFoodDetection\900_yolox_850.25.pth')
    image_path = r'C:\Dataset\FoodDetectionDataset\img\2.jpg'

    # 如果需要重新構建TensorRT引擎，需要將最後輸出的list去除，不過這樣會造成與onnxruntime衝突，會導致onnxruntime發生錯誤
    # TODO: 處理TensorRT對Onnx的SequenceConstruct的相容性
    tensorrt_engine = create_tensorrt_engine(onnx_file_path='YoloxObjectDetectionL.onnx', fp16_mode=True,
                                             trt_logger_level='INTERNAL_ERROR',
                                             save_trt_engine_path='YoloxObjectDetectionL.trt')
    trt_image = cv2.imread(image_path)
    results = tensorrt_engine_detect_image(tensorrt_engine=tensorrt_engine, image=trt_image)
    labels, scores, boxes = results
    for score, box in zip(scores, boxes):
        ymin, xmin, ymax, xmax = box
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cv2.rectangle(trt_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.putText(trt_image, str(score), (xmin + 30, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (89, 214, 210), 2, cv2.LINE_AA)
    trt_image = Image.fromarray(cv2.cvtColor(trt_image, cv2.COLOR_BGR2RGB))
    trt_image.show()

    # model = create_onnx_session()
    # image = cv2.imread(image_path)
    # results = onnxruntime_detect_image(model, image)
    # labels, scores, boxes = results
    # for score, box in zip(scores, boxes):
    #     ymin, xmin, ymax, xmax = box
    #     xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    #     cv2.putText(image, str(score), (xmin + 30, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX,
    #                 1, (89, 214, 210), 2, cv2.LINE_AA)
    # img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # img.show()

    # from SpecialTopic.YoloxObjectDetection.api import init_model, detect_image
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # torch_model = init_model(pretrained=r'C:\Checkpoint\YoloxFoodDetection\900_yolox_850.25.pth',
    #                          num_classes=9)
    # torch_model.eval()
    # torch_image = cv2.imread(image_path)
    # results = detect_image(torch_model, device, torch_image, input_shape=[640, 640], num_classes=9)
    # labels, scores, boxes = results
    # for score, box in zip(scores, boxes):
    #     ymin, xmin, ymax, xmax = box
    #     xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
    #     cv2.rectangle(torch_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    #     cv2.putText(torch_image, str(score), (xmin + 30, ymin + 30), cv2.FONT_HERSHEY_SIMPLEX,
    #                 1, (89, 214, 210), 2, cv2.LINE_AA)
    # torch_image = Image.fromarray(cv2.cvtColor(torch_image, cv2.COLOR_BGR2RGB))
    # torch_image.show()
