import cv2
import numpy as np


def plot_bboxes(image, bboxes, line_thickness=None):
    # 已看過
    # 根據輸入的標註匡訊息，在輸入的圖像上面進行標註

    # Plots one bounding box on image img
    # 如果沒有傳入標註匡寬度就會依據圖像大小決定標註匡寬度
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    list_pts = []
    point_radius = 4

    # 遍歷整個需要標註的標註匡訊息
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        # 依據不同的類別可以給不同的標註匡顏色
        if cls_id in ['car', 'bus', 'truck']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        # check whether hit line
        # 在標註匡上面會有一個點
        check_point_x = x1
        check_point_y = int(y1 + ((y2 - y1) * 0.6))

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        cv2.putText(image, '{}id:{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
        list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

        ndarray_pts = np.array(list_pts, np.int32)
        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))
        list_pts.clear()
    return image
