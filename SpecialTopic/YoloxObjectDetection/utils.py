import numpy as np
import torch
from torchvision.ops import boxes
from PIL import Image


def get_classes(classes_path):
    with open(classes_path) as f:
        class_name = f.readlines()
    class_names = [c.strip() for c in class_name]
    return class_names, len(class_names)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def decode_outputs(outputs, input_shape):
    grids = list()
    strides = list()
    hw = [x.shape[-2:] for x in outputs]
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    for h, w in hw:
        grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape = grid.shape[:2]
        grids.append(grid)
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
    grids = torch.cat(grids, dim=1).type(outputs.type())
    strides = torch.cat(strides, dim=1).type(outputs.type())
    outputs[..., :2] = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    outputs[..., [0, 2]] = outputs[..., [0, 2]] / input_shape[1]
    outputs[..., [1, 3]] = outputs[..., [1, 3]] / input_shape[0]
    return outputs


def non_max_suppression(prediction, num_classes, input_shape, image_shape, keep_ratio, conf_thres=0.5, nms_thres=0.4):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()
        if not image_pred.size(0):
            continue
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        nms_out_index = boxes.batched_nms(
            detections[:, :4], detections[:, 4] * detections[:, 5], detections[:, 6], nms_thres)
        output[i] = detections[nms_out_index]
        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4] = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, keep_ratio)
    return output


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, keep_ratio):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)
    if keep_ratio:
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size, keep_ratio=True):
    iw, ih = image.size
    h, w = size
    if keep_ratio:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image
