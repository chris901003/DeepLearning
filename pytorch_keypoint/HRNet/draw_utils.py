import numpy as np
from numpy import ndarray
import PIL
from PIL import ImageDraw, ImageFont
from PIL.Image import Image

# COCO 17 points
point_name = ["nose", "left_eye", "right_eye",
              "left_ear", "right_ear",
              "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow",
              "left_wrist", "right_wrist",
              "left_hip", "right_hip",
              "left_knee", "right_knee",
              "left_ankle", "right_ankle"]

point_color = [(240, 2, 127), (240, 2, 127), (240, 2, 127),
               (240, 2, 127), (240, 2, 127),
               (255, 255, 51), (255, 255, 51),
               (254, 153, 41), (44, 127, 184),
               (217, 95, 14), (0, 0, 255),
               (255, 255, 51), (255, 255, 51), (228, 26, 28),
               (49, 163, 84), (252, 176, 243), (0, 176, 240),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142),
               (255, 255, 0), (169, 209, 142)]


def draw_keypoints(img: Image,
                   keypoints: ndarray,
                   scores: ndarray = None,
                   thresh: float = 0.2,
                   r: int = 2,
                   draw_text: bool = False,
                   font: str = 'arial.ttf',
                   font_size: int = 10):
    """
    :param img: 原始傳入圖像
    :param keypoints: shape [num_kps, 2]
    :param scores: shape [num_kps, 1]
    :param thresh: 閾值，預設為0.2
    :param r: 預設為3
    :param draw_text: 預設為False
    :param font: 字體決定
    :param font_size: 字體大小決定
    :return:
    """
    # 如果img是numpy格式就把它轉成Image格式
    if isinstance(img, ndarray):
        img = PIL.Image.fromarray(img)

    # 如果沒有傳入score，就製造一個長度為關節點數量且全為1的矩陣
    if scores is None:
        scores = np.ones(keypoints.shape[0])

    # 預設為False
    if draw_text:
        try:
            font = ImageFont.truetype(font, font_size)
        except IOError:
            font = ImageFont.load_default()

    # 製造可以畫圖的實例對象
    draw = ImageDraw.Draw(img)
    # 遍歷每一個關節點
    for i, (point, score) in enumerate(zip(keypoints, scores)):
        # 如果預測分數大於閾值且預測的點是合法的話就可以進行標注
        if score > thresh and np.max(point) > 0:
            # 給定四個點的位置可以畫出圓或是橢圓，fill=要填充的顏色，outline=邊界的顏色
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r],
                         fill=point_color[i],
                         outline=(255, 255, 255))
            # 除了標記點以外還寫出點的名稱
            if draw_text:
                draw.text((point[0] + r, point[1] + r), text=point_name[i], font=font)

    # 返回標註好的圖片
    return img
