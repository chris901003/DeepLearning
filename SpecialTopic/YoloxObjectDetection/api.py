import cv2
import numpy as np
import PIL
import torch
from PIL import Image
from utils import resize_image, cvtColor, preprocess_input, decode_outputs, non_max_suppression


def detect_image(model, device, image_info, input_shape, num_classes, confidence=0.5, nms_iou=0.3, keep_ratio=True):
    if isinstance(image_info, str):
        image = Image.open(image_info)
    elif isinstance(image_info, np.array):
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
