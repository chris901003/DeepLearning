import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from SpecialTopic.ST.build import build_detector
from SpecialTopic.YoloxObjectDetection.utils import resize_image, cvtColor, preprocess_input, decode_outputs, \
    non_max_suppression


def detect_image(model, device, image_info, input_shape, num_classes, confidence=0.5, nms_iou=0.3, keep_ratio=True):
    if isinstance(image_info, str):
        image = Image.open(image_info)
    elif type(image_info) is np.ndarray:
        image = Image.fromarray(cv2.cvtColor(image_info, cv2.COLOR_BGR2RGB))
    elif isinstance(image_info, PIL.JpegImagePlugin.JpegImageFile):
        image = image_info
    else:
        raise ValueError('傳入的圖像資料需要是圖像路徑或是已經是ndarray或是PIL格式')
    image_shape = np.array(np.shape(image)[0:2])
    image = cvtColor(image)
    image_data = resize_image(image, input_shape, keep_ratio=keep_ratio)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        images = images.to(device)
        outputs = model(images)
        outputs = decode_outputs(outputs, input_shape)
        results = non_max_suppression(outputs, num_classes, input_shape, image_shape, keep_ratio, conf_thres=confidence,
                                      nms_thres=nms_iou)
    top_label = np.array(results[0][:, 6], dtype='int32').tolist()
    top_conf = results[0][:, 4] * results[0][:, 5].tolist()
    top_boxes = results[0][:, :4].tolist()
    return top_label, top_conf, top_boxes


def init_model(cfg='auto', pretrained='none', num_classes=100, phi='l', device='auto'):
    if device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if cfg == 'auto':
        cfg = {
            'type': 'YoloBody',
            'phi': phi,
            'backbone_cfg': {
                'type': 'YOLOPAFPN'
            },
            'head_cfg': {
                'type': 'YOLOXHead',
                'num_classes': num_classes
            }
        }
    model = build_detector(cfg)
    if pretrained != 'none':
        print(f'Load weights {pretrained}')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pretrained, map_location=device)
        if 'model_weight' in pretrained_dict:
            pretrained_dict = pretrained_dict['model_weight']
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        model = model.to(device)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        assert len(no_load_key) == 0, '給定的預訓練權重與模型不匹配'
    else:
        print('未加載預訓練權重，模型工作是幾乎無效的')
    return model
