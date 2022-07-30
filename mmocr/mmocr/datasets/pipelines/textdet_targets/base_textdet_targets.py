# Copyright (c) OpenMMLab. All rights reserved.
import sys

import cv2
import numpy as np
import pyclipper
from mmcv.utils import print_log
from shapely.geometry import Polygon as plg

import mmocr.utils.check_argument as check_argument


class BaseTextDetTargets:
    """Generate text detector ground truths."""

    def __init__(self):
        pass

    def point2line(self, xs, ys, point_1, point_2):
        """Compute the distance from point to a line. This is adapted from
        https://github.com/MhLiao/DB.

        Args:
            xs (ndarray): The x coordinates of size hxw.
            ys (ndarray): The y coordinates of size hxw.
            point_1 (ndarray): The first point with shape 1x2.
            point_2 (ndarray): The second point with shape 1x2.

        Returns:
            result (ndarray): The distance matrix of size hxw.
        """
        # suppose a triangle with three edge abc with c=point_1 point_2
        # a^2
        a_square = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        # b^2
        b_square = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        # c^2
        c_square = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] -
                                                                  point_2[1])
        # -cosC=(c^2-a^2-b^2)/2(ab)
        neg_cos_c = (
            (c_square - a_square - b_square) /
            (np.finfo(np.float32).eps + 2 * np.sqrt(a_square * b_square)))
        # sinC^2=1-cosC^2
        square_sin = 1 - np.square(neg_cos_c)
        square_sin = np.nan_to_num(square_sin)
        # distance=a*b*sinC/c=a*h/c=2*area/c
        result = np.sqrt(a_square * b_square * square_sin /
                         (np.finfo(np.float32).eps + c_square))
        # set result to minimum edge if C<pi/2
        result[neg_cos_c < 0] = np.sqrt(np.fmin(a_square,
                                                b_square))[neg_cos_c < 0]
        return result

    def polygon_area(self, polygon):
        """Compute the polygon area. Please refer to Green's theorem.
        https://en.wikipedia.org/wiki/Green%27s_theorem. This is adapted from
        https://github.com/MhLiao/DB.

        Args:
            polygon (ndarray): The polygon boundary points.
        """

        polygon = polygon.reshape(-1, 2)
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (
                polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.

    def polygon_size(self, polygon):
        """Estimate the height and width of the minimum bounding box of the
        polygon.

        Args:
            polygon (ndarray): The polygon point sequence.

        Returns:
            size (tuple): The height and width of the minimum bounding box.
        """
        poly = polygon.reshape(-1, 2)
        rect = cv2.minAreaRect(poly.astype(np.int32))
        size = rect[1]
        return size

    def generate_kernels(self,
                         img_size,
                         text_polys,
                         shrink_ratio,
                         max_shrink=sys.maxsize,
                         ignore_tags=None):
        """Generate text instance kernels for one shrink ratio.

        Args:
            img_size (tuple(int, int)): The image size of (height, width).
            text_polys (list[list[ndarray]]: The list of text polygons.
            shrink_ratio (float): The shrink ratio of kernel.

        Returns:
            text_kernel (ndarray): The text kernel mask of (height, width).
        """
        # 已看過，構建一個text標註透過給定一個縮放比例
        # img_size = 當前圖像大小
        # test_polys = 標註訊息，list[list[ndarray]]，ndarray裏面的資訊就是標註匡的(x,y)
        # shrink_ratio = 減縮比例
        # max_shrink = 最大減縮量
        # ignore_tags = 忽略的tags

        # 檢查傳入的資料是否有格式錯誤
        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)
        assert isinstance(shrink_ratio, float)

        # 獲取當前圖像高寬
        h, w = img_size
        # 構建一個全為0且shape[h, w]的ndarray
        text_kernel = np.zeros((h, w), dtype=np.float32)

        # 遍歷所有的標註訊息
        for text_ind, poly in enumerate(text_polys):
            # 將x與y分開，instance shape [points, 2]，points就是該標註匡用多少個點構成
            instance = poly[0].reshape(-1, 2).astype(np.int32)
            # 將instance放入到plg中計算匡出的面積
            area = plg(instance).area
            # 透過cv2的arcLength獲取標註匡的周長
            peri = cv2.arcLength(instance, True)
            # 計算縮放距離，同時該距離不會小於最大縮小距離
            distance = min(
                int(area * (1 - shrink_ratio * shrink_ratio) / (peri + 0.001) +
                    0.5), max_shrink)
            # 構建多邊形內縮實例對象
            pco = pyclipper.PyclipperOffset()
            # 準備多邊形內縮的東西
            pco.AddPath(instance, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
            # 將instance進行內縮distance距離，這裡加上負號表示內縮且內縮的距離需要是整數
            # shrunk = 內縮過後的多邊形座標，shape list[ndarray[points, 2]]，points標示該標註匡由多少的點標注而成
            shrunk = np.array(pco.Execute(-distance))

            # check shrunk == [] or empty ndarray
            if len(shrunk) == 0 or shrunk.size == 0:
                # 如果縮放後沒有點就放到ignore_tags當中
                if ignore_tags is not None:
                    ignore_tags[text_ind] = True
                continue
            try:
                # 將外層的list去除
                shrunk = np.array(shrunk[0]).reshape(-1, 2)

            except Exception as e:
                print_log(f'{shrunk} with error {e}')
                if ignore_tags is not None:
                    ignore_tags[text_ind] = True
                continue
            # 透過fillPoly進行填充顏色，將被塗色圖像以及多邊形座標以及要塗的顏色放入
            # 查看:https://shengyu7697.github.io/python-opencv-fillpoly/
            cv2.fillPoly(text_kernel, [shrunk.astype(np.int32)], text_ind + 1)
        # 最終將結果回傳
        return text_kernel, ignore_tags

    def generate_effective_mask(self, mask_size: tuple, polygons_ignore):
        """Generate effective mask by setting the ineffective regions to 0 and
        effective regions to 1.

        Args:
            mask_size (tuple): The mask size.
            polygons_ignore (list[[ndarray]]: The list of ignored text
                polygons.

        Returns:
            mask (ndarray): The effective mask of (height, width).
        """
        # 已看過，處理ignore的mask，會將有被標注道的地方設定成0其餘為1

        # 檢查傳入的polygons_ignore是否為list型態
        assert check_argument.is_2dlist(polygons_ignore)

        # 構建全為1且shape[height, width]的ndarray
        mask = np.ones(mask_size, dtype=np.uint8)

        # 遍歷所有ignore的mask
        for poly in polygons_ignore:
            # instance shape = [points, 2] -> [1, points, 2]
            instance = poly[0].reshape(-1,
                                       2).astype(np.int32).reshape(1, -1, 2)
            # 透過fillPoly將標注區域設定成0
            cv2.fillPoly(mask, instance, 0)

        # 回傳mask
        return mask

    def generate_targets(self, results):
        raise NotImplementedError

    def __call__(self, results):
        # 已看過，主要是構建ground truth圖像
        # 透過generate_targets構建
        results = self.generate_targets(results)
        return results
