# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np
from mmdet.core import BitmapMasks
from mmdet.datasets.builder import PIPELINES
from numpy.linalg import norm

import mmocr.utils.check_argument as check_argument
from . import BaseTextDetTargets


@PIPELINES.register_module()
class TextSnakeTargets(BaseTextDetTargets):
    """Generate the ground truth targets of TextSnake: TextSnake: A Flexible
    Representation for Detecting Text of Arbitrary Shapes.

    [https://arxiv.org/abs/1807.01544]. This was partially adapted from
    https://github.com/princewang1994/TextSnake.pytorch.

    Args:
        orientation_thr (float): The threshold for distinguishing between
            head edge and tail edge among the horizontal and vertical edges
            of a quadrangle.
    """

    def __init__(self,
                 orientation_thr=2.0,
                 resample_step=4.0,
                 center_region_shrink_ratio=0.3):
        """ 已看過，構建TextSnake的標註圖像
        Args:
            orientation_thr: 四邊形水平邊和垂直邊中區分頭邊和尾邊的閾值
        """

        # 繼承自BaseTextDetTargets，將繼承對象進行初始化
        super().__init__()
        # 保存傳入的參數
        self.orientation_thr = orientation_thr
        self.resample_step = resample_step
        self.center_region_shrink_ratio = center_region_shrink_ratio
        # 主要是在除法時避免除到0
        self.eps = 1e-8

    def vector_angle(self, vec1, vec2):
        """ 已看過，計算兩個向量之間的夾角
        Args:
            vec1: 當前向量資訊，ndarray shape [2]
            vec2: 當前邊的兩邊向量，ndarray shape [2, 2]
        """
        if vec1.ndim > 1:
            # 如果vec1是一個以上的向量就會到這裡
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + self.eps).reshape(
                (-1, 1))
        else:
            # 如果vec1只有一個向量就會到這裡
            # 這裡是在將原始向量轉成單位向量，norm就是找到向量的長度，原始向量除以向量長度就會是單位向量，eps是避免向量長度為0
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + self.eps)
        if vec2.ndim > 1:
            # 如果vec2是一個以上的向量就會到這裡
            # 這裡的差別就是最後需要再調整一下通道
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + self.eps).reshape(
                (-1, 1))
        else:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + self.eps)
        # unit_vec1 * unit_vec2 = 在對應位置上做相乘，ndarray shape [2, 2]
        # np.sum() = 在最後一個維度上進行加總，ndarray shape [2]
        # np.clip() = 將值限定在[-1.0, 1.0]當中
        # np.across() = 做外積
        # 所以是當前的向量與兩邊的向量做內積後結果再做外積，最終回傳ndarray shape [2]
        return np.arccos(
            np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

    def vector_slope(self, vec):
        # 已看過，獲取斜率
        assert len(vec) == 2
        return abs(vec[1] / (vec[0] + self.eps))

    def vector_sin(self, vec):
        # 已看過，獲取sin
        assert len(vec) == 2
        return vec[1] / (norm(vec) + self.eps)

    def vector_cos(self, vec):
        # 已看過，獲取cos
        assert len(vec) == 2
        return vec[0] / (norm(vec) + self.eps)

    def find_head_tail(self, points, orientation_thr):
        """Find the head edge and tail edge of a text polygon.

        Args:
            points (ndarray): The points composing a text polygon.
            orientation_thr (float): The threshold for distinguishing between
                head edge and tail edge among the horizontal and vertical edges
                of a quadrangle.

        Returns:
            head_inds (list): The indexes of two points composing head edge.
            tail_inds (list): The indexes of two points composing tail edge.
        """
        # 已看過，主要是在給定的文字團座標當中找到頭的邊以及尾的邊
        # points = 文字團的座標，ndarray [points, 2]
        # orientation_thr = 四邊形水平邊和垂直邊中區分頭邊和尾邊的閾值

        # 檢查傳入的points是否符合規定
        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2
        assert isinstance(orientation_thr, float)

        if len(points) > 4:
            # 如果points當中的點數超過4個點就會到這裡
            # 將頭的point添加到最後端當中，這樣就算成圍城一圈
            pad_points = np.vstack([points, points[0]])
            # 獲取兩個相鄰點之間的向量，edge_vec = nadarray shape [points, 2]
            edge_vec = pad_points[1:] - pad_points[:-1]

            # 角度的綜合
            theta_sum = []
            # 相鄰向量角度
            adjacent_vec_theta = []
            # 遍歷所有的邊向量
            for i, edge_vec1 in enumerate(edge_vec):
                # 獲取當前邊的左右兩條邊在edge_vec當中的index
                adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
                # 獲取當前邊的左右兩條邊的邊向量，adjacent_edge_vec shape = ndarray [2, 2]
                adjacent_edge_vec = edge_vec[adjacent_ind]
                # 將當前的向量與左右兩條邊的向量傳入到vector_angle當中，將結果加總
                temp_theta_sum = np.sum(
                    self.vector_angle(edge_vec1, adjacent_edge_vec))
                # 獲取相鄰之間的夾角角度，ndarray shape [1]
                temp_adjacent_theta = self.vector_angle(
                    adjacent_edge_vec[0], adjacent_edge_vec[1])
                # 將結果進行保存
                theta_sum.append(temp_theta_sum)
                adjacent_vec_theta.append(temp_adjacent_theta)
            # 將theta_sum轉成ndarray shape [edges=points]同時都除以pi
            theta_sum_score = np.array(theta_sum) / np.pi
            # 將adjacent_vec_theta轉成ndarray shape [edges=points]同時都除以pi
            adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
            # 分別對所有的x點去平均以及y點取平均，poly_center shape = ndarray [2]
            poly_center = np.mean(points, axis=0)
            # 兩相鄰點與中心點距離較長的長度，edge_dist shape = ndarray shape [points]
            edge_dist = np.maximum(
                norm(pad_points[1:] - poly_center, axis=-1),
                norm(pad_points[:-1] - poly_center, axis=-1))
            # 距離的分數會是所有距離除以最大距離，也就是最大距離的分數會是1，其他會小於1
            dist_score = edge_dist / (np.max(edge_dist) + self.eps)
            # 位置分數，原先設定全為0，ndarray shape [points]
            position_score = np.zeros(len(edge_vec))
            # score shape = ndarray [points]
            score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
            score += 0.35 * dist_score
            if len(points) % 2 == 0:
                # 如果構成文字團的點是偶數就會到這裡
                position_score[(len(score) // 2 - 1)] += 1
                position_score[-1] += 1
            # 將position_score的分數加上去
            score += 0.1 * position_score
            # pad_score = ndarray shape [points * 2]
            pad_score = np.concatenate([score, score])
            # score_matrix shape = ndarray [point, point - 3]且全為0
            score_matrix = np.zeros((len(score), len(score) - 3))
            x = np.arange(len(score) - 3) / float(len(score) - 4)
            gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
                (x - 0.5) / 0.5, 2.) / 2)
            gaussian = gaussian / np.max(gaussian)
            for i in range(len(score)):
                score_matrix[i, :] = score[i] + pad_score[
                    (i + 2):(i + len(score) - 1)] * gaussian * 0.3

            head_start, tail_increment = np.unravel_index(
                score_matrix.argmax(), score_matrix.shape)
            tail_start = (head_start + tail_increment + 2) % len(points)
            head_end = (head_start + 1) % len(points)
            tail_end = (tail_start + 1) % len(points)

            if head_end > tail_end:
                head_start, tail_start = tail_start, head_start
                head_end, tail_end = tail_end, head_end
            head_inds = [head_start, head_end]
            tail_inds = [tail_start, tail_end]
        else:
            if self.vector_slope(points[1] - points[0]) + self.vector_slope(
                    points[3] - points[2]) < self.vector_slope(
                        points[2] - points[1]) + self.vector_slope(points[0] -
                                                                   points[3]):
                horizontal_edge_inds = [[0, 1], [2, 3]]
                vertical_edge_inds = [[3, 0], [1, 2]]
            else:
                horizontal_edge_inds = [[3, 0], [1, 2]]
                vertical_edge_inds = [[0, 1], [2, 3]]

            vertical_len_sum = norm(points[vertical_edge_inds[0][0]] -
                                    points[vertical_edge_inds[0][1]]) + norm(
                                        points[vertical_edge_inds[1][0]] -
                                        points[vertical_edge_inds[1][1]])
            horizontal_len_sum = norm(
                points[horizontal_edge_inds[0][0]] -
                points[horizontal_edge_inds[0][1]]) + norm(
                    points[horizontal_edge_inds[1][0]] -
                    points[horizontal_edge_inds[1][1]])

            if vertical_len_sum > horizontal_len_sum * orientation_thr:
                head_inds = horizontal_edge_inds[0]
                tail_inds = horizontal_edge_inds[1]
            else:
                head_inds = vertical_edge_inds[0]
                tail_inds = vertical_edge_inds[1]

        return head_inds, tail_inds

    def reorder_poly_edge(self, points):
        """Get the respective points composing head edge, tail edge, top
        sideline and bottom sideline.

        Args:
            points (ndarray): The points composing a text polygon.

        Returns:
            head_edge (ndarray): The two points composing the head edge of text
                polygon.
            tail_edge (ndarray): The two points composing the tail edge of text
                polygon.
            top_sideline (ndarray): The points composing top curved sideline of
                text polygon.
            bot_sideline (ndarray): The points composing bottom curved sideline
                of text polygon.
        """
        # 已看過，獲取組成頭邊、尾邊、頂邊線和底邊線的相應點。
        # points = 一個文字團的標註點

        # 檢查傳入的points通道數是否為2
        assert points.ndim == 2
        # 至少需要4個點以上
        assert points.shape[0] >= 4
        # 這裡會是[x, y]
        assert points.shape[1] == 2

        # 尋找頭尾的index，orientation_thr = 四邊形水平邊和垂直邊中區分頭邊和尾邊的閾值
        # head_inds, tail_inds = ndarray shape [2]，由兩個點之間的編組成的邊
        head_inds, tail_inds = self.find_head_tail(points,
                                                   self.orientation_thr)
        # 獲取指定的index的座標，head_edge與tail_edge shape [2, 2]
        head_edge, tail_edge = points[head_inds], points[tail_inds]

        # pad_points shape [points * 2, 2]，將points複製兩份
        pad_points = np.vstack([points, points])
        if tail_inds[1] < 1:
            # 如果tail_inds[1]是0就將其變成len(points)表示的點是一樣的
            tail_inds[1] = len(points)
        # 這裡會將標註邊分成兩段
        sideline1 = pad_points[head_inds[1]:tail_inds[1]]
        sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
        # 計算中間值的偏移
        sideline_mean_shift = np.mean(
            sideline1, axis=0) - np.mean(
                sideline2, axis=0)

        # 根據偏移會讓分配top_sideline與bot_sideline
        if sideline_mean_shift[1] > 0:
            top_sideline, bot_sideline = sideline2, sideline1
        else:
            top_sideline, bot_sideline = sideline1, sideline2

        # 最終回傳
        return head_edge, tail_edge, top_sideline, bot_sideline

    def cal_curve_length(self, line):
        """Calculate the length of each edge on the discrete curve and the sum.

        Args:
            line (ndarray): The points composing a discrete curve.

        Returns:
            tuple: Returns (edges_length, total_length).

                - | edge_length (ndarray): The length of each edge on the
                    discrete curve.
                - | total_length (float): The total length of the discrete
                    curve.
        """
        # 已看過，計算離散曲線上每條邊的長度和總和
        # line = 一條線上的座標點

        # 檢查傳入資料是否有問題
        assert line.ndim == 2
        assert len(line) >= 2

        # 獲取兩相鄰點之間的距離，shape ndarray [points - 1]
        edges_length = np.sqrt((line[1:, 0] - line[:-1, 0])**2 +
                               (line[1:, 1] - line[:-1, 1])**2)
        # 獲得總長
        total_length = np.sum(edges_length)
        # 將相鄰點距離以及總長回傳
        return edges_length, total_length

    def resample_line(self, line, n):
        """Resample n points on a line.

        Args:
            line (ndarray): The points composing a line.
            n (int): The resampled points number.

        Returns:
            resampled_line (ndarray): The points composing the resampled line.
        """
        # 已看過，對一條線進行重新採樣
        # line = 由點組成的線的點資訊，ndarray [points, 2]
        # n = 重新採樣後的點數

        # 檢查是否合法
        assert line.ndim == 2
        assert line.shape[0] >= 2
        assert line.shape[1] == 2
        assert isinstance(n, int)
        assert n > 2

        # edges_length = 相鄰點之間的距離，ndarray [points - 1]
        # total_length = 距離的總和，ndarray [1]
        edges_length, total_length = self.cal_curve_length(line)
        # 透過cumsum進行由左往右累加，在開頭的地方加上0，t_org shape = ndarray [points]
        t_org = np.insert(np.cumsum(edges_length), 0, 0)
        # unit_t = 取平均距離，對於新採樣後每個點之間的平均距離
        unit_t = total_length / (n - 1)
        # 等距離的累加距離
        t_equidistant = np.arange(1, n - 1, dtype=np.float32) * unit_t
        # 邊的index
        edge_ind = 0
        # 最開始的point座標
        points = [line[0]]
        # 遍歷等距離的list
        for t in t_equidistant:
            # 我們找到當前t會在原始累加邊長的哪兩個index之間，t_org[edge_ind + 1]會是比當前t還要大的值
            # 也就是最後會是，t_org[edge_ind] < t < t_org[edge_ind + 1]
            while edge_ind < len(edges_length) - 1 and t > t_org[edge_ind + 1]:
                edge_ind += 1
            # 將左邊與右邊提取出來
            t_l, t_r = t_org[edge_ind], t_org[edge_ind + 1]
            # 獲取當前點到左端點與右端點的權重比
            weight = np.array([t_r - t, t - t_l], dtype=np.float32) / (
                t_r - t_l + self.eps)
            # 用矩陣乘法找到最後的座標位置，這裡用的就是內插法
            p_coords = np.dot(weight, line[[edge_ind, edge_ind + 1]])
            # 最後將點保存下來
            points.append(p_coords)
        # 將最後結尾點放上
        points.append(line[-1])
        # 堆疊在一起，ndarray shape [points, 2]
        resampled_line = np.vstack(points)

        return resampled_line

    def resample_sidelines(self, sideline1, sideline2, resample_step):
        """Resample two sidelines to be of the same points number according to
        step size.

        Args:
            sideline1 (ndarray): The points composing a sideline of a text
                polygon.
            sideline2 (ndarray): The points composing another sideline of a
                text polygon.
            resample_step (float): The resampled step size.

        Returns:
            resampled_line1 (ndarray): The resampled line 1.
            resampled_line2 (ndarray): The resampled line 2.
        """
        # 已看過，根據步長將兩條邊線重採樣為相同的點數
        # sideline1 = 文字團當中一邊的文字座標
        # sideline2 = 文字團當中另一邊的文字座標
        # resample_step = 重新採樣的步距

        # 檢查傳入的資料
        assert sideline1.ndim == sideline2.ndim == 2
        assert sideline1.shape[1] == sideline2.shape[1] == 2
        assert sideline1.shape[0] >= 2
        assert sideline2.shape[0] >= 2
        assert isinstance(resample_step, float)

        # length1, length2 = 該線段相鄰點的距離總和
        _, length1 = self.cal_curve_length(sideline1)
        _, length2 = self.cal_curve_length(sideline2)

        # 計算平均長度
        avg_length = (length1 + length2) / 2
        # 計算最後在採樣的點數
        resample_point_num = max(int(float(avg_length) / resample_step) + 1, 3)

        # 透過resample_line進行重新採樣，resampled_line1與resampled_line2的shape = [points, 2]
        resampled_line1 = self.resample_line(sideline1, resample_point_num)
        resampled_line2 = self.resample_line(sideline2, resample_point_num)

        return resampled_line1, resampled_line2

    def draw_center_region_maps(self, top_line, bot_line, center_line,
                                center_region_mask, radius_map, sin_map,
                                cos_map, region_shrink_ratio):
        """Draw attributes on text center region.

        Args:
            top_line (ndarray): The points composing top curved sideline of
                text polygon.
            bot_line (ndarray): The points composing bottom curved sideline
                of text polygon.
            center_line (ndarray): The points composing the center line of text
                instance.
            center_region_mask (ndarray): The text center region mask.
            radius_map (ndarray): The map where the distance from point to
                sidelines will be drawn on for each pixel in text center
                region.
            sin_map (ndarray): The map where vector_sin(theta) will be drawn
                on text center regions. Theta is the angle between tangent
                line and vector (1, 0).
            cos_map (ndarray): The map where vector_cos(theta) will be drawn on
                text center regions. Theta is the angle between tangent line
                and vector (1, 0).
            region_shrink_ratio (float): The shrink ratio of text center.
        """
        # 已看過，在文本中心區域繪製
        # top_line = 文字群外匡部分的上邊部分，ndarray shape [top_points, 2] (top_points + bot_points = tot_points)
        # bot_line = 文字群外匡部分的下邊部分，ndarray shape [bot_points, 2] (top_points + bot_points = tot_points)
        # center_line = 中心線部分，這裡是透過上邊線加上下邊線的平均出來的，ndarray shape [tot_points / 2, 2]
        # center_region_mask = 文本中心部分的mask，一開始傳入時全為0，ndarray shape [height, width]
        # radius_map = 將為文本中心區域中的每個像素繪製從點到邊線的距離的map，一開始傳入時全為0，ndarray shape [height, width]
        # sin_map = 文本中心與法線的夾角的sin值 (目前不確定)，一開始傳入時全為0，ndarray shape [height, width]
        # cos_map = 文本中心與法線的夾角的cos值 (目前不確定)，一開始傳入時全為0，ndarray shape [height, width]
        # region_shrink_ratio = 從外圍縮小多少倍會定義成文本中心

        # 檢查傳入資料是否正確
        assert top_line.shape == bot_line.shape == center_line.shape
        assert (center_region_mask.shape == radius_map.shape == sin_map.shape
                == cos_map.shape)
        assert isinstance(region_shrink_ratio, float)
        # 從頭開始遍歷到倒數第二個點
        for i in range(0, len(center_line) - 1):
            # 當前點與當前點的下一個點之間取中間值
            top_mid_point = (top_line[i] + top_line[i + 1]) / 2
            bot_mid_point = (bot_line[i] + bot_line[i + 1]) / 2
            # 半徑就會是兩點中間的距離除以二
            radius = norm(top_mid_point - bot_mid_point) / 2

            # 文字的方向會是下個中間線的點減去當前中間線的點
            text_direction = center_line[i + 1] - center_line[i]
            # sin_theta = dif_x / len, cos_theta = dif_y / len
            sin_theta = self.vector_sin(text_direction)
            cos_theta = self.vector_cos(text_direction)

            # 獲取文字中心區塊，會是原大小進行region_shrink_ratio的縮放
            # 左上方邊界點
            tl = center_line[i] + (top_line[i] -
                                   center_line[i]) * region_shrink_ratio
            # 右上方邊界點
            tr = center_line[i + 1] + (
                top_line[i + 1] - center_line[i + 1]) * region_shrink_ratio
            # 右下方邊界點
            br = center_line[i + 1] + (
                bot_line[i + 1] - center_line[i + 1]) * region_shrink_ratio
            # 左下方邊界點
            bl = center_line[i] + (bot_line[i] -
                                   center_line[i]) * region_shrink_ratio
            # 最後由四個點組成的矩形內就是設定的文字中心位置
            current_center_box = np.vstack([tl, tr, br, bl]).astype(np.int32)

            # 透過fillPoly將內部全部填充上color值
            cv2.fillPoly(center_region_mask, [current_center_box], color=1)
            cv2.fillPoly(sin_map, [current_center_box], color=sin_theta)
            cv2.fillPoly(cos_map, [current_center_box], color=cos_theta)
            cv2.fillPoly(radius_map, [current_center_box], color=radius)

    def generate_center_mask_attrib_maps(self, img_size, text_polys):
        """Generate text center region mask and geometric attribute maps.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            center_region_mask (ndarray): The text center region mask.
            radius_map (ndarray): The distance map from each pixel in text
                center region to top sideline.
            sin_map (ndarray): The sin(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).
            cos_map (ndarray): The cos(theta) map where theta is the angle
                between vector (top point - bottom point) and vector (1, 0).
        """
        # 已看過，構建文字中心實例mask以及幾何屬性圖
        # img_size = 當前圖像大小
        # text_polys = 文字團標註

        # 檢查傳入的img_size的型態是否為tuple
        assert isinstance(img_size, tuple)
        # 檢查文字團標註
        assert check_argument.is_2dlist(text_polys)

        # 獲取當前圖像高寬
        h, w = img_size

        # 構建全為0且shape是[height, width]的ndarray，用來標註文字中心的mask用的
        center_region_mask = np.zeros((h, w), np.uint8)
        # 構建全為0且shape是[height, width]的ndarray，用來標註該位置應該預測的圓圈半徑
        radius_map = np.zeros((h, w), dtype=np.float32)
        # 構建全為0且shape是[height, width]的ndarray，用來標註該位置應該預測的sin值
        sin_map = np.zeros((h, w), dtype=np.float32)
        # 構建全為0且shape是[height, width]的ndarray，用來標註該位置應該預測的cos值
        cos_map = np.zeros((h, w), dtype=np.float32)

        # 遍歷所有的標註文字團
        for poly in text_polys:
            # 檢查poly型態是否合法
            assert len(poly) == 1
            # 將資料處理成(x, y)一組
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            # polygon_points = ndarray shape [points, 2]
            polygon_points = np.array(text_instance).reshape(-1, 2)

            # 獲取總共有多少個膽點組成一個文字團
            n = len(polygon_points)
            # 須保留的點
            keep_inds = []
            # 遍歷所有點
            for i in range(n):
                # 如果當前點與下個點的差值透過norm後大於1e-5就會被保留
                # 這裡用的是numpy的norm，默認是用二范數，也就是會將數入全部平方後開根號
                # 這裡計算的就會是歐式距離，基本上不要太近就都會進行保留
                if norm(polygon_points[i] -
                        polygon_points[(i + 1) % n]) > 1e-5:
                    keep_inds.append(i)
            # 保存一個文字團的點
            polygon_points = polygon_points[keep_inds]

            # 將標著點傳入到recorder_poly_edge當中
            # top_line = 標註圖像上半部的線，ndarray [top_points, 2]
            # bot_line = 標註圖像下半部的線，ndarray [bot_points, 2]
            # top_points + bot_points = tot_points
            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            # resampled_top_line與resampled_bot_line shape = ndarray [points, 2]
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            # 將resampled_bot_line順序進行反轉，這樣可以讓top的最後一個點接到bot的最後一個點
            resampled_bot_line = resampled_bot_line[::-1]
            # 獲取文字團中間線，這裡透過將上方的線與下方的線相加後就可以獲取中心線
            center_line = (resampled_top_line + resampled_bot_line) / 2

            # 將中心線的頭與尾的點相減獲取向量，通過調整後中心線都會是由左下到右上
            if self.vector_slope(center_line[-1] - center_line[0]) > 0.9:
                # 如果斜率大於0.9就會到這裡，這裡計算的是x/y所以斜率大表示的是很平
                if (center_line[-1] - center_line[0])[1] < 0:
                    # 這裡就會是斜向下的平
                    # 就全部翻轉過來
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]
            else:
                # 當中心線不是很平的時候就會到這裏
                if (center_line[-1] - center_line[0])[0] < 0:
                    # 如果中心線是由右到左就會全部翻轉
                    center_line = center_line[::-1]
                    resampled_top_line = resampled_top_line[::-1]
                    resampled_bot_line = resampled_bot_line[::-1]

            # 獲取上邊頭與下邊頭的距離除以4
            line_head_shrink_len = norm(resampled_top_line[0] -
                                        resampled_bot_line[0]) / 4.0
            # 獲取上邊尾與下邊尾的距離除以4
            line_tail_shrink_len = norm(resampled_top_line[-1] -
                                        resampled_bot_line[-1]) / 4.0
            # 將shrink_len除以resample_step，計算出需要縮減多少的頭與尾
            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)

            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                # 如果中心線組成的點數量大於head_shrink_num+tail_shrink_num+2就會進來
                # 這裏我們就會取其中一部分的資料，放棄一些頭尾資料
                center_line = center_line[head_shrink_num:len(center_line) -
                                          tail_shrink_num]
                resampled_top_line = resampled_top_line[
                    head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[
                    head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            # 傳入的center_region_mask與radius_map與sin_map與cos_map都會是全0的
            self.draw_center_region_maps(resampled_top_line,
                                         resampled_bot_line, center_line,
                                         center_region_mask, radius_map,
                                         sin_map, cos_map,
                                         self.center_region_shrink_ratio)

        # 以下的shape皆為ndarray [height, width]
        # center_region_mask = 如果為文本中心區域的地方會是1否則就會是0
        # radius_map = 會在文本中心區域的地方表示希望預測的radius，不同的文字群會有不同的radius值，同一文字群當中也會有不同值
        # sin_map = 跟radius_map相同只是保存的是sin值
        # cos_map = 跟radius_map相同只是保存的是cos值
        return center_region_mask, radius_map, sin_map, cos_map

    def generate_text_region_mask(self, img_size, text_polys):
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """
        # 已看過，構建文字中心實例mask以及幾何屬性圖
        # img_size = 當前圖像的高寬
        # text_polys = 標註訊息

        # 檢查輸入資料
        assert isinstance(img_size, tuple)
        assert check_argument.is_2dlist(text_polys)

        # 獲取當前圖像高寬
        h, w = img_size
        # 構建高寬與當前圖像相同且全為0的ndarray，shape [height, width]
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        # 遍歷所有的標註資料
        for poly in text_polys:
            # 檢查poly是否合法
            assert len(poly) == 1
            # 將(x, y)取出來，text_instance = list[list]，第一個list長度是有多少點，第二個list長度為2存放[x, y]
            text_instance = [[poly[0][i], poly[0][i + 1]]
                             for i in range(0, len(poly[0]), 2)]
            # 轉成numpy，polygon = ndarray shape [1, points, 2]
            polygon = np.array(
                text_instance, dtype=np.int32).reshape((1, -1, 2))
            # 透過fillPoly會將text_region_mask當中被polygon包圍的地方放下1
            cv2.fillPoly(text_region_mask, polygon, 1)

        # 回傳，有文字的地方會是1沒有文字的地方會是0
        return text_region_mask

    def generate_targets(self, results):
        """Generate the gt targets for TextSnake.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """
        # 已看過，構建gt圖像，專門給TextSnake使用的

        # 檢查傳入的results是否為dict格式
        assert isinstance(results, dict)

        # 獲取標註訊息資料
        polygon_masks = results['gt_masks'].masks
        # 獲取ignore的標註資料
        polygon_masks_ignore = results['gt_masks_ignore'].masks

        # 獲取當前圖像的高寬
        h, w, _ = results['img_shape']

        # gt_text_mask = 有文字的地方會是1沒有文字的地方會是0，ndarray shape [height, width]
        gt_text_mask = self.generate_text_region_mask((h, w), polygon_masks)
        # gt_mask = 有需要ignore的地方會是0，其他地方會是1
        gt_mask = self.generate_effective_mask((h, w), polygon_masks_ignore)

        # 構建文字中心的mask
        # 以下的shape皆為ndarray [height, width]
        # gt_center_region_mask = 如果為文本中心區域的地方會是1否則就會是0
        # gt_radius_map = 會在文本中心區域的地方表示希望預測的radius，不同的文字群會有不同的radius值，同一文字群當中也會有不同值
        # gt_sin_map = 跟radius_map相同只是保存的是sin值
        # gt_cos_map = 跟radius_map相同只是保存的是cos值
        (gt_center_region_mask, gt_radius_map, gt_sin_map,
         gt_cos_map) = self.generate_center_mask_attrib_maps((h, w),
                                                             polygon_masks)

        # 將results當中與mask相關的保存資料清除
        results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        # 構建新的mapping關係，這裡都是剛生成的資料
        mapping = {
            'gt_text_mask': gt_text_mask,
            'gt_center_region_mask': gt_center_region_mask,
            'gt_mask': gt_mask,
            'gt_radius_map': gt_radius_map,
            'gt_sin_map': gt_sin_map,
            'gt_cos_map': gt_cos_map
        }
        # 將mapping當中的資料都放入到BitmapMasks類當中進行實例化
        for key, value in mapping.items():
            value = value if isinstance(value, list) else [value]
            results[key] = BitmapMasks(value, h, w)
            # 添加到與mask相關資訊裡
            results['mask_fields'].append(key)

        # 回傳更新好的results
        return results
