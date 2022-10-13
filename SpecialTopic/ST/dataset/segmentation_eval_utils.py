import torch
import numpy as np
import cv2


def intersect_and_union(pred_label, label, num_classes, ignore_index, label_map='Default', reduce_zero_label=False):
    if label_map == 'Default':
        label_map = dict()
    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy(pred_label)
    if isinstance(label, str):
        label = torch.from_numpy(cv2.imread(label))
    else:
        label = torch.from_numpy(label)
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=num_classes, min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def pre_eval_to_metrics(pre_eval_results, metrics='Default', nan_to_num=None, beta=1):
    if metrics == 'Default':
        metrics = ['mIoU']
    pre_eval_results = tuple(zip(*pre_eval_results))
    assert len(pre_eval_results) == 4
    total_area_intersect = sum(pre_eval_results[0])
    total_area_union = sum(pre_eval_results[1])
    total_area_pred_label = sum(pre_eval_results[2])
    total_area_label = sum(pre_eval_results[3])
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label, total_area_label,
                                        metrics, nan_to_num, beta)
    return ret_metrics


def total_area_to_metrics(total_area_intersect, total_area_union, total_area_pred_label, total_area_label,
                          metrics='Default', nan_to_num=None, beta=1):
    if metrics == 'Default':
        metrics = 'mIoU'
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU']
    if not set(metrics).issubset(set(allowed_metrics)):
        assert KeyError
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = dict(aAcc=all_acc)
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
    ret_metrics = {
        metric: value.numpy() for metric, value in ret_metrics.items()
    }
    return ret_metrics
