# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import numpy as np

from mmpose.core.post_processing import (get_affine_transform, get_warp_matrix,
                                         warp_affine_joints)
from mmpose.datasets.builder import PIPELINES
from .shared_transform import Compose


def _ceil_to_multiples_of(x, base=64):
    """Transform x to the integral multiple of the base."""
    # 獲取離x最近且可以被base整除的值，如果需要調整只能選大於x的值
    return int(np.ceil(x / base)) * base


def _get_multi_scale_size(image,
                          input_size,
                          current_scale,
                          min_scale,
                          use_udp=False):
    """Get the size for multi-scale training.

    Args:
        image: Input image.
        input_size (np.ndarray[2]): Size (w, h) of the image input.
        current_scale (float): Scale factor.
        min_scale (float): Minimal scale.
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing multi-scale sizes.

        - (w_resized, h_resized) (tuple(int)): resized width/height
        - center (np.ndarray)image center
        - scale (np.ndarray): scales wrt width/height
    """
    """ 獲取多尺度圖像大小
    Args:
        image: 圖像資料，ndarray shape [height, width, channel]
        input_size: 輸入到網路的圖像大小
        current_scale: 當前的縮放比例
        min_scale: 最小縮放比例
        use_udp: 是否使用udp
    """
    assert len(input_size) == 2
    h, w, _ = image.shape

    # calculate the size for min_scale
    min_input_w = _ceil_to_multiples_of(min_scale * input_size[0], 64)
    min_input_h = _ceil_to_multiples_of(min_scale * input_size[1], 64)
    if w < h:
        w_resized = int(min_input_w * current_scale / min_scale)
        h_resized = int(
            _ceil_to_multiples_of(min_input_w / w * h, 64) * current_scale /
            min_scale)
        if use_udp:
            scale_w = w - 1.0
            scale_h = (h_resized - 1.0) / (w_resized - 1.0) * (w - 1.0)
        else:
            scale_w = w / 200.0
            scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = int(min_input_h * current_scale / min_scale)
        w_resized = int(
            _ceil_to_multiples_of(min_input_h / h * w, 64) * current_scale /
            min_scale)
        if use_udp:
            scale_h = h - 1.0
            scale_w = (w_resized - 1.0) / (h_resized - 1.0) * (h - 1.0)
        else:
            scale_h = h / 200.0
            scale_w = w_resized / h_resized * h / 200.0
    if use_udp:
        center = (scale_w / 2.0, scale_h / 2.0)
    else:
        center = np.array([round(w / 2.0), round(h / 2.0)])
    # 返回(resize後的高寬, 中心點位置, 高寬縮放比例)
    return (w_resized, h_resized), center, np.array([scale_w, scale_h])


def _resize_align_multi_scale(image, input_size, current_scale, min_scale):
    """Resize the images for multi-scale training.

    Args:
        image: Input image
        input_size (np.ndarray[2]): Size (w, h) of the image input
        current_scale (float): Current scale
        min_scale (float): Minimal scale

    Returns:
        tuple: A tuple containing image info.

        - image_resized (np.ndarray): resized image
        - center (np.ndarray): center of image
        - scale (np.ndarray): scale
    """
    """ 調整大小對齊多尺度
    Args:
        image: 當前圖像資訊，ndarray shape [height, width, channel]
        input_size: 輸入到模型的圖像大小，ndarray shape [2]
        current_scale: 當前的縮放比例
        min_scale: 最小縮放比例
    """
    # 檢查input_size需要長度為2
    assert len(input_size) == 2
    # 將資料通過_get_multi_scale_size，獲取資料為(resize後的高寬, 中心點位置, 高寬縮放比例)
    size_resized, center, scale = _get_multi_scale_size(image, input_size, current_scale, min_scale)

    # 獲取仿射變換矩陣
    trans = get_affine_transform(center, scale, 0, size_resized)
    # 進行仿射變換
    image_resized = cv2.warpAffine(image, trans, size_resized)

    # 將結果回傳(變換後的圖像資料, 中心點, 高寬縮放比例)
    return image_resized, center, scale


def _resize_align_multi_scale_udp(image, input_size, current_scale, min_scale):
    """Resize the images for multi-scale training.

    Args:
        image: Input image
        input_size (np.ndarray[2]): Size (w, h) of the image input
        current_scale (float): Current scale
        min_scale (float): Minimal scale

    Returns:
        tuple: A tuple containing image info.

        - image_resized (np.ndarray): resized image
        - center (np.ndarray): center of image
        - scale (np.ndarray): scale
    """
    assert len(input_size) == 2
    size_resized, _, _ = _get_multi_scale_size(image, input_size,
                                               current_scale, min_scale, True)

    _, center, scale = _get_multi_scale_size(image, input_size, min_scale,
                                             min_scale, True)

    trans = get_warp_matrix(
        theta=0,
        size_input=np.array(scale, dtype=np.float32),
        size_dst=np.array(size_resized, dtype=np.float32) - 1.0,
        size_target=np.array(scale, dtype=np.float32))
    image_resized = cv2.warpAffine(
        image.copy(), trans, size_resized, flags=cv2.INTER_LINEAR)

    return image_resized, center, scale


class HeatmapGenerator:
    """Generate heatmaps for bottom-up models.

    Args:
        num_joints (int): Number of keypoints
        output_size (np.ndarray): Size (w, h) of feature map
        sigma (int): Sigma of the heatmaps.
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, output_size, num_joints, sigma=-1, use_udp=False):
        """ 構建熱力圖資料，專門給bottom-up模型使用
        Args:
            output_size: 輸出的大小
            num_joints: 關節點數量
            sigma: 高斯分佈的超參數
            use_udp: 是否使用udp
        """
        if not isinstance(output_size, np.ndarray):
            # 如果output_size不是ndarray，就將其轉成ndarray
            output_size = np.array(output_size)
        if output_size.size > 1:
            # 如果output_size不是1就會檢查
            assert len(output_size) == 2
            self.output_size = output_size
        else:
            # 將output_size變成[output_size, output_size]
            self.output_size = np.array([output_size, output_size], dtype=np.int)
        # 保存關節點數量
        self.num_joints = num_joints
        if sigma < 0:
            # 獲取sigma值
            sigma = self.output_size.prod()**0.5 / 64
        # 保存sigma值
        self.sigma = sigma
        size = 6 * sigma + 3
        self.use_udp = use_udp
        if use_udp:
            # 如果使用udp就會到這裡
            self.x = np.arange(0, size, 1, np.float32)
            self.y = self.x[:, None]
        else:
            # 如果沒有使用udp就會到這裡
            # 構建x會是全為1，且shape = [size]
            x = np.arange(0, size, 1, np.float32)
            # 構建y
            y = x[:, None]
            x0, y0 = 3 * sigma + 1, 3 * sigma + 1
            self.g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def __call__(self, joints):
        """Generate heatmaps."""
        # 構建熱力圖資料，傳入的是關節點座標位置，joints shape [num_peoples, num_joints, 3]

        # 構建一個全為0且shape為[num_joints, height, width]的ndarray
        hms = np.zeros((self.num_joints, self.output_size[1], self.output_size[0]), dtype=np.float32)

        # 獲取高斯分佈的sigma超參數
        sigma = self.sigma
        # 遍歷每個人的關節點
        for p in joints:
            # 遍歷當中每個座標
            for idx, pt in enumerate(p):
                if pt[2] > 0:
                    # 如果該點不是不可預測點就會到這裡
                    # 將x與y座標提取出來
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or x >= self.output_size[0] or y >= self.output_size[1]:
                        # 如果(x, y)座標在非法位置就會直接continue跳過
                        continue

                    if self.use_udp:
                        # 如果使用udp就會到這裡
                        x0 = 3 * sigma + 1 + pt[0] - x
                        y0 = 3 * sigma + 1 + pt[1] - y
                        g = np.exp(-((self.x - x0)**2 + (self.y - y0)**2) /
                                   (2 * sigma**2))
                    else:
                        # 否則就會到這裡，獲取g參數
                        g = self.g

                    # 以下開始將該點透過高斯分佈均勻的在點的周圍填上值
                    ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                    br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                    c, d = max(0, -ul[0]), min(br[0], self.output_size[0]) - ul[0]
                    a, b = max(0, -ul[1]), min(br[1], self.output_size[1]) - ul[1]

                    cc, dd = max(0, ul[0]), min(br[0], self.output_size[0])
                    aa, bb = max(0, ul[1]), min(br[1], self.output_size[1])
                    hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], g[a:b, c:d])
        # hms = ndarray shape [num_joints, height, width]，每個關節點會有自己的一張圖，會在點的周圍給上準確率，中心點會是1
        # 然後往外擴散，主要是透過高斯分佈進行擴散
        return hms


class JointsEncoder:
    """Encodes the visible joints into (coordinates, score); The coordinate of
    one joint and its score are of `int` type.

    (idx * output_size**2 + y * output_size + x, 1) or (0, 0).

    Args:
        max_num_people(int): Max number of people in an image
        num_joints(int): Number of keypoints
        output_size(np.ndarray): Size (w, h) of feature map
        tag_per_joint(bool):  Option to use one tag map per joint.
    """

    def __init__(self, max_num_people, num_joints, output_size, tag_per_joint):
        """ 將可見的關節點變成(座標, 分數)，這裡座標以及分數會是int格式
        Args:
            max_num_people: 一張圖像最多人數
            num_joints: 關節點數量
            output_size: 輸出的大小
            tag_per_joint: 每個關節點使用一張圖，也就是一張圖就只表示一個關節點
        """
        # 保存傳入的參數
        self.max_num_people = max_num_people
        self.num_joints = num_joints
        # 處理輸入的output_size資料
        if not isinstance(output_size, np.ndarray):
            output_size = np.array(output_size)
        if output_size.size > 1:
            assert len(output_size) == 2
            self.output_size = output_size
        else:
            self.output_size = np.array([output_size, output_size], dtype=np.int)
        self.tag_per_joint = tag_per_joint

    def __call__(self, joints):
        """
        Note:
            - number of people in image: N
            - number of keypoints: K
            - max number of people in an image: M

        Args:
            joints (np.ndarray[N,K,3])

        Returns:
            visible_kpts (np.ndarray[M,K,2]).
        """
        # joints = ndarray shape [num_peoples, num_joints, 3]，標記出每個人物的關節點座標位置
        # 構建全為0的ndarray且shape = [最大偵測人數, 關節點數量, 2]
        visible_kpts = np.zeros((self.max_num_people, self.num_joints, 2), dtype=np.float32)
        # 遍歷圖像當中所有人
        for i in range(len(joints)):
            # 將tot設定成0
            tot = 0
            # 遍歷一個人的所有關節點
            for idx, pt in enumerate(joints[i]):
                # 獲取關節點的x與y座標
                x, y = int(pt[0]), int(pt[1])
                if (pt[2] > 0 and 0 <= y < self.output_size[1]
                        and 0 <= x < self.output_size[0]):
                    # 如果該關節點座標是合法的且該關節點是可預測的就會到這裡
                    if self.tag_per_joint:
                        # 如果有啟用tag_per_joint就會到這裡
                        # 保存第i個人物第tot個關節的座標位置，可是這裡會加上idx*output_size.prod()
                        visible_kpts[i][tot] = \
                            (idx * self.output_size.prod()
                             + y * self.output_size[0] + x, 1)
                    else:
                        # 如果沒有啟用tag_per_joint就會到這裡
                        visible_kpts[i][tot] = (y * self.output_size[0] + x, 1)
                    # 將tot加一
                    tot += 1
        # 回傳
        return visible_kpts


class PAFGenerator:
    """Generate part affinity fields.

    Args:
        output_size (np.ndarray): Size (w, h) of feature map.
        limb_width (int): Limb width of part affinity fields.
        skeleton (list[list]): connections of joints.
    """

    def __init__(self, output_size, limb_width, skeleton):
        if not isinstance(output_size, np.ndarray):
            output_size = np.array(output_size)
        if output_size.size > 1:
            assert len(output_size) == 2
            self.output_size = output_size
        else:
            self.output_size = np.array([output_size, output_size],
                                        dtype=np.int)
        self.limb_width = limb_width
        self.skeleton = skeleton

    def _accumulate_paf_map_(self, pafs, src, dst, count):
        """Accumulate part affinity fields between two given joints.

        Args:
            pafs (np.ndarray[2,H,W]): paf maps (2 dimensions:x axis and
                y axis) for a certain limb connection. This argument will
                be modified inplace.
            src (np.ndarray[2,]): coordinates of the source joint.
            dst (np.ndarray[2,]): coordinates of the destination joint.
            count (np.ndarray[H,W]): count map that preserves the number
                of non-zero vectors at each point. This argument will be
                modified inplace.
        """
        limb_vec = dst - src
        norm = np.linalg.norm(limb_vec)
        if norm == 0:
            unit_limb_vec = np.zeros(2)
        else:
            unit_limb_vec = limb_vec / norm

        min_x = max(np.floor(min(src[0], dst[0]) - self.limb_width), 0)
        max_x = min(
            np.ceil(max(src[0], dst[0]) + self.limb_width),
            self.output_size[0] - 1)
        min_y = max(np.floor(min(src[1], dst[1]) - self.limb_width), 0)
        max_y = min(
            np.ceil(max(src[1], dst[1]) + self.limb_width),
            self.output_size[1] - 1)

        range_x = list(range(int(min_x), int(max_x + 1), 1))
        range_y = list(range(int(min_y), int(max_y + 1), 1))

        mask = np.zeros_like(count, dtype=bool)
        if len(range_x) > 0 and len(range_y) > 0:
            xx, yy = np.meshgrid(range_x, range_y)
            delta_x = xx - src[0]
            delta_y = yy - src[1]
            dist = np.abs(delta_x * unit_limb_vec[1] -
                          delta_y * unit_limb_vec[0])
            mask_local = (dist < self.limb_width)
            mask[yy, xx] = mask_local

        pafs[0, mask] += unit_limb_vec[0]
        pafs[1, mask] += unit_limb_vec[1]
        count += mask

        return pafs, count

    def __call__(self, joints):
        """Generate the target part affinity fields."""
        pafs = np.zeros(
            (len(self.skeleton) * 2, self.output_size[1], self.output_size[0]),
            dtype=np.float32)

        for idx, sk in enumerate(self.skeleton):
            count = np.zeros((self.output_size[1], self.output_size[0]),
                             dtype=np.float32)

            for p in joints:
                src = p[sk[0]]
                dst = p[sk[1]]
                if src[2] > 0 and dst[2] > 0:
                    self._accumulate_paf_map_(pafs[2 * idx:2 * idx + 2],
                                              src[:2], dst[:2], count)

            pafs[2 * idx:2 * idx + 2] /= np.maximum(count, 1)

        return pafs


@PIPELINES.register_module()
class BottomUpRandomFlip:
    """Data augmentation with random image flip for bottom-up.

    Args:
        flip_prob (float): Probability of flip.
    """

    def __init__(self, flip_prob=0.5):
        # BottomUp的隨機翻轉
        # 保存翻轉概率
        self.flip_prob = flip_prob

    def __call__(self, results):
        """Perform data augmentation with random image flip."""
        # 進行BottomUp的隨機翻轉
        image, mask, joints = results['img'], results['mask'], results['joints']
        # 獲取每個關節點對稱的關節點index
        self.flip_index = results['ann_info']['flip_index']
        # 獲取輸出的熱力圖大小
        self.output_size = results['ann_info']['heatmap_size']

        # 獲取mask與joints資料是否合法
        assert isinstance(mask, list)
        assert isinstance(joints, list)
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size)

        # 隨機翻轉
        if np.random.random() < self.flip_prob:
            # 獲取翻轉後的圖像
            image = image[:, ::-1].copy() - np.zeros_like(image)
            # 遍歷輸出的圖像數量
            for i, _output_size in enumerate(self.output_size):
                if not isinstance(_output_size, np.ndarray):
                    # 如果_output_size不是ndarray就轉成ndarray
                    _output_size = np.array(_output_size)
                if _output_size.size > 1:
                    assert len(_output_size) == 2
                else:
                    _output_size = np.array([_output_size, _output_size],
                                            dtype=np.int)
                # 將mask資訊進行選轉
                mask[i] = mask[i][:, ::-1].copy()
                # 先將點的index換成對稱點的index
                joints[i] = joints[i][:, self.flip_index]
                # 將x方向變成左右轉換後的方向
                joints[i][:, :, 0] = _output_size[0] - joints[i][:, :, 0] - 1
        # 更新results當中資料
        results['img'], results['mask'], results['joints'] = image, mask, joints
        # 將更新後的results返回
        return results


@PIPELINES.register_module()
class BottomUpRandomAffine:
    """Data augmentation with random scaling & rotating.

    Args:
        rot_factor (int): Rotating to [-rotation_factor, rotation_factor]
        scale_factor (float): Scaling to [1-scale_factor, 1+scale_factor]
        scale_type: wrt ``long`` or ``short`` length of the image.
        trans_factor: Translation factor.
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self,
                 rot_factor,
                 scale_factor,
                 scale_type,
                 trans_factor,
                 use_udp=False):
        """ 進行圖像隨機旋轉以及縮放
        Args:
            rot_factor: 旋轉的角度範圍，會在[-rot_factor, rot_factor]之間
            scale_factor: 縮放比例範圍
            scale_type: 這裡會是long或是short
            trans_factor:
            use_udp: 是否使用使用無偏數據處理
        """
        # 保存傳入資料
        self.max_rotation = rot_factor
        # 將縮放比例提取出來
        self.min_scale = scale_factor[0]
        self.max_scale = scale_factor[1]
        self.scale_type = scale_type
        self.trans_factor = trans_factor
        self.use_udp = use_udp

    def _get_scale(self, image_size, resized_size):
        """ 獲取縮放比例
        Args:
            image_size: 原圖像大小
            resized_size: 透過resize後的圖像大小
        """
        # 將當前高寬提取出來
        w, h = image_size
        # 將resize後圖像的高寬提取出來
        w_resized, h_resized = resized_size
        if w / w_resized < h / h_resized:
            # 如果高度方向的縮小量較大就會到這裡
            if self.scale_type == 'long':
                # 如果scale_type是long就會到這裡，這裡會以h為主，將h調整到h_resized時計算w的長度
                w_pad = h / h_resized * w_resized
                h_pad = h
            elif self.scale_type == 'short':
                # 如果scale_type是short就會到這裡，這裡會以為主，將w調整到w_resized時計算h的長度
                w_pad = w
                h_pad = w / w_resized * h_resized
            else:
                # 其他的scale_type就會報錯
                raise ValueError(f'Unknown scale type: {self.scale_type}')
        else:
            # 如果寬度方向的縮小量較大就會到這裡
            if self.scale_type == 'long':
                # 如果scale_type是short就會到這裡，這裡會以為主，將w調整到w_resized時計算h的長度
                w_pad = w
                h_pad = w / w_resized * h_resized
            elif self.scale_type == 'short':
                # 如果scale_type是long就會到這裡，這裡會以h為主，將h調整到h_resized時計算w的長度
                w_pad = h / h_resized * w_resized
                h_pad = h
            else:
                # 其他的scale_type就會報錯
                raise ValueError(f'Unknown scale type: {self.scale_type}')

        # 將w_pad與h_pad包裝
        scale = np.array([w_pad, h_pad], dtype=np.float32)

        # 回傳scale
        return scale

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        # 進行圖像隨機旋轉以及縮放
        # 獲取results當中的img以及mask以及joints參數
        image, mask, joints = results['img'], results['mask'], results['joints']

        # 獲取anno_info當中指定的圖像大小
        self.input_size = results['ann_info']['image_size']
        if not isinstance(self.input_size, np.ndarray):
            # 如果input_size不是ndarray型態就會轉成ndarray型態
            self.input_size = np.array(self.input_size)
        if self.input_size.size > 1:
            # 如果input_size的size大於1就會是要2
            assert len(self.input_size) == 2
        else:
            # 否則就擴大成[input_size, input_size]
            self.input_size = [self.input_size, self.input_size]
        # 獲取heatmap_size作為output_size
        self.output_size = results['ann_info']['heatmap_size']

        # 檢查mask需要是list
        assert isinstance(mask, list)
        # 檢查joints是list
        assert isinstance(joints, list)
        # 檢查mask的數量要與joints相同
        assert len(mask) == len(joints)
        assert len(mask) == len(self.output_size), (len(mask),
                                                    len(self.output_size),
                                                    self.output_size)

        # 獲取當前圖像高寬
        height, width = image.shape[:2]
        if self.use_udp:
            # 如果使用upd就會到這裡
            center = np.array(((width - 1.0) / 2, (height - 1.0) / 2))
        else:
            # 如果沒有使用upd就會到這裡
            # 獲取中心座標
            center = np.array((width / 2, height / 2))

        # 構建img_scale會是ndarray且shape是[2]
        img_scale = np.array([width, height], dtype=np.float32)
        # 隨機獲取數據增強的縮放比例，這裡會是在[min_scale, max_scale]之間
        aug_scale = np.random.random() * (self.max_scale - self.min_scale) \
            + self.min_scale
        # 將原始圖像大小乘上aug_scale獲取縮放後的比例
        img_scale *= aug_scale
        # 獲取數據增強使用的旋轉角度
        aug_rot = (np.random.random() * 2 - 1) * self.max_rotation

        if self.trans_factor > 0:
            # 如果trans_factor大於0就會到這裡
            # 這裡會隨機獲取dx與dy的值
            dx = np.random.randint(-self.trans_factor * img_scale[0] / 200.0,
                                   self.trans_factor * img_scale[0] / 200.0)
            dy = np.random.randint(-self.trans_factor * img_scale[1] / 200.0,
                                   self.trans_factor * img_scale[1] / 200.0)

            # 將中心點加上dx與dy
            center[0] += dx
            center[1] += dy
        if self.use_udp:
            # 如果有使用udp就會到這裡
            for i, _output_size in enumerate(self.output_size):
                if not isinstance(_output_size, np.ndarray):
                    _output_size = np.array(_output_size)
                if _output_size.size > 1:
                    assert len(_output_size) == 2
                else:
                    _output_size = [_output_size, _output_size]

                scale = self._get_scale(img_scale, _output_size)

                trans = get_warp_matrix(
                    theta=aug_rot,
                    size_input=center * 2.0,
                    size_dst=np.array(
                        (_output_size[0], _output_size[1]), dtype=np.float32) -
                    1.0,
                    size_target=scale)
                mask[i] = cv2.warpAffine(
                    (mask[i] * 255).astype(np.uint8),
                    trans, (int(_output_size[0]), int(_output_size[1])),
                    flags=cv2.INTER_LINEAR) / 255
                mask[i] = (mask[i] > 0.5).astype(np.float32)
                joints[i][:, :, 0:2] = \
                    warp_affine_joints(joints[i][:, :, 0:2].copy(), trans)
                if results['ann_info']['scale_aware_sigma']:
                    joints[i][:, :, 3] = joints[i][:, :, 3] / aug_scale
            scale = self._get_scale(img_scale, self.input_size)
            mat_input = get_warp_matrix(
                theta=aug_rot,
                size_input=center * 2.0,
                size_dst=np.array((self.input_size[0], self.input_size[1]),
                                  dtype=np.float32) - 1.0,
                size_target=scale)
            image = cv2.warpAffine(
                image,
                mat_input, (int(self.input_size[0]), int(self.input_size[1])),
                flags=cv2.INTER_LINEAR)
        else:
            # 沒有使用udp就會到這裡，遍歷output_size的大小，這裡獲取的是mask以及joints的資料
            for i, _output_size in enumerate(self.output_size):
                if not isinstance(_output_size, np.ndarray):
                    # 如果當前的_output_size不是ndarray就會到這裡，將_output_size轉成ndarray
                    _output_size = np.array(_output_size)
                if _output_size.size > 1:
                    # 如果_output_size的size不是1就要是2
                    assert len(_output_size) == 2
                else:
                    # 將_output_size變成list且分成高寬兩個
                    _output_size = [_output_size, _output_size]
                # 透過_get_scale獲取縮放比例，img_scale是透過隨機縮放後的圖像大小，_output_size是希望的圖像大小
                # scale = 依據scale_type獲取最後的高寬
                scale = self._get_scale(img_scale, _output_size)
                # 獲取仿射變換矩陣
                mat_output = get_affine_transform(
                    center=center,
                    scale=scale / 200.0,
                    rot=aug_rot,
                    output_size=_output_size)
                # 透過cv2的warpAffine進行仿射變換，這裡會是對於mask進行
                mask[i] = cv2.warpAffine(
                    (mask[i] * 255).astype(np.uint8), mat_output,
                    (int(_output_size[0]), int(_output_size[1]))) / 255
                # 上面會先將True以及False轉成數字，這裡再轉回來
                mask[i] = (mask[i] > 0.5).astype(np.float32)

                # 對關節點進行仿射變換
                joints[i][:, :, 0:2] = warp_affine_joints(joints[i][:, :, 0:2], mat_output)
                if results['ann_info']['scale_aware_sigma']:
                    # 調整joints[:, :, 3]的值
                    joints[i][:, :, 3] = joints[i][:, :, 3] / aug_scale

            # 獲取img_scale與input_size的縮放比例
            scale = self._get_scale(img_scale, self.input_size)
            # 獲取仿射變換矩陣，這裡獲取的是傳入網路的圖像資料
            mat_input = get_affine_transform(
                center=center,
                scale=scale / 200.0,
                rot=aug_rot,
                output_size=self.input_size)
            # 對圖像進行仿射變換，最終圖像會調整到self.input_size大小
            image = cv2.warpAffine(image, mat_input, (int(
                self.input_size[0]), int(self.input_size[1])))

        # 更新資料
        results['img'], results['mask'], results[
            'joints'] = image, mask, joints

        return results


@PIPELINES.register_module()
class BottomUpGenerateHeatmapTarget:
    """Generate multi-scale heatmap target for bottom-up.

    Args:
        sigma (int): Sigma of heatmap Gaussian
        max_num_people (int): Maximum number of people in an image
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, sigma, use_udp=False):
        self.sigma = sigma
        self.use_udp = use_udp

    def _generate(self, num_joints, heatmap_size):
        """Get heatmap generator."""
        heatmap_generator = [
            HeatmapGenerator(output_size, num_joints, self.sigma, self.use_udp)
            for output_size in heatmap_size
        ]
        return heatmap_generator

    def __call__(self, results):
        """Generate multi-scale heatmap target for bottom-up."""
        heatmap_generator = \
            self._generate(results['ann_info']['num_joints'],
                           results['ann_info']['heatmap_size'])
        target_list = list()
        joints_list = results['joints']

        for scale_id in range(results['ann_info']['num_scales']):
            heatmaps = heatmap_generator[scale_id](joints_list[scale_id])
            target_list.append(heatmaps.astype(np.float32))
        results['target'] = target_list

        return results


@PIPELINES.register_module()
class BottomUpGenerateTarget:
    """Generate multi-scale heatmap target for associate embedding.

    Args:
        sigma (int): Sigma of heatmap Gaussian
        max_num_people (int): Maximum number of people in an image
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, sigma, max_num_people, use_udp=False):
        """ 構建多尺度熱力圖標註，專門給關聯嵌入使用
        Args:
            sigma: 在熱力圖上的高斯分佈超參數
            max_num_people: 最多一張圖像當中的人數
            use_udp: 是否使用upd技術
        """
        # 保存傳入參數
        self.sigma = sigma
        self.max_num_people = max_num_people
        self.use_udp = use_udp

    def _generate(self, num_joints, heatmap_size):
        """Get heatmap generator and joint encoder."""
        # num_joints = 該數據集總共有多少個關節點
        # heatmap_size = 熱力圖大小，ndarray shape [2]

        # 構建熱力圖資料
        heatmap_generator = [
            # 使用HeatmapGenerator創建熱力圖資料
            HeatmapGenerator(output_size, num_joints, self.sigma, self.use_udp)
            for output_size in heatmap_size
        ]
        # 構建關節點資料
        joints_encoder = [
            # 使用JointsEncoder構建關節點資料
            JointsEncoder(self.max_num_people, num_joints, output_size, True)
            for output_size in heatmap_size
        ]
        # 構建的熱力圖以及關節點實例對象
        return heatmap_generator, joints_encoder

    def __call__(self, results):
        """Generate multi-scale heatmap target for bottom-up."""
        # 生成多尺度熱力圖標註，專門給bottom-up模型使用

        # 透過_generate構建heatmap_generator以及joints_encoder
        # 構建的熱力圖以及關節點實例對象
        heatmap_generator, joints_encoder = \
            self._generate(results['ann_info']['num_joints'], results['ann_info']['heatmap_size'])
        # 構建target_list，存放熱力圖資料的
        target_list = list()
        # 取出results當中的mask以及joints資料，joints_list shape = [peoples, num_joints, 3]
        mask_list, joints_list = results['mask'], results['joints']

        # 遍歷總共有多少總縮放比例
        for scale_id in range(results['ann_info']['num_scales']):
            # 將joints_list放入到構建熱力圖的實例化對象當中，target_t shape = [num_joints, height, width]
            target_t = heatmap_generator[scale_id](joints_list[scale_id])
            # 將joints_list放入到構建關節點的實例化對象當中，joints_t shape = [最大偵測人數, 關節點數量, 2]
            joints_t = joints_encoder[scale_id](joints_list[scale_id])

            # 將target_t保存到target_list當中，這會是我們希望模型預測出來的熱力圖
            target_list.append(target_t.astype(np.float32))
            # 將mask對應的scale_id變成float32型態
            mask_list[scale_id] = mask_list[scale_id].astype(np.float32)
            # 將joints_list的對應scale_id變成joints_t的int32型態
            joints_list[scale_id] = joints_t.astype(np.int32)

        # 將masks與joints與targets更新到results當中
        results['masks'], results['joints'] = mask_list, joints_list
        results['targets'] = target_list

        # 將更新後的results返回
        return results


@PIPELINES.register_module()
class BottomUpGeneratePAFTarget:
    """Generate multi-scale heatmaps and part affinity fields (PAF) target for
    bottom-up. Paper ref: Cao et al. Realtime Multi-Person 2D Human Pose
    Estimation using Part Affinity Fields (CVPR 2017).

    Args:
        limb_width (int): Limb width of part affinity fields
    """

    def __init__(self, limb_width, skeleton=None):
        self.limb_width = limb_width
        self.skeleton = skeleton

    def _generate(self, heatmap_size, skeleton):
        """Get PAF generator."""
        paf_generator = [
            PAFGenerator(output_size, self.limb_width, skeleton)
            for output_size in heatmap_size
        ]
        return paf_generator

    def __call__(self, results):
        """Generate multi-scale part affinity fields for bottom-up."""
        if self.skeleton is None:
            assert results['ann_info']['skeleton'] is not None
            self.skeleton = results['ann_info']['skeleton']

        paf_generator = \
            self._generate(results['ann_info']['heatmap_size'],
                           self.skeleton)
        target_list = list()
        joints_list = results['joints']

        for scale_id in range(results['ann_info']['num_scales']):
            pafs = paf_generator[scale_id](joints_list[scale_id])
            target_list.append(pafs.astype(np.float32))

        results['target'] = target_list

        return results


@PIPELINES.register_module()
class BottomUpGetImgSize:
    """Get multi-scale image sizes for bottom-up, including base_size and
    test_scale_factor. Keep the ratio and the image is resized to
    `results['ann_info']['image_size']×current_scale`.

    Args:
        test_scale_factor (List[float]): Multi scale
        current_scale (int): default 1
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, test_scale_factor, current_scale=1, use_udp=False):
        """ 獲取多尺度圖像專門給bottom-up模型，保括基礎尺度以及測試的縮放尺度
        Args:
            test_scale_factor: 多尺度
            current_scale: 預設為1
            use_udp: 是否使用使用無偏置數據處理
        """
        # 將傳入的資料進行保存
        self.test_scale_factor = test_scale_factor
        # 獲取test_scale_factor當中最小的倍率值
        self.min_scale = min(test_scale_factor)
        self.current_scale = current_scale
        self.use_udp = use_udp

    def __call__(self, results):
        """Get multi-scale image sizes for bottom-up."""
        # 獲取多尺度圖像專門給bottom-up模型，保括基礎尺度以及測試的縮放尺度
        # input_size = 輸入到模型當中的圖像大小
        input_size = results['ann_info']['image_size']
        if not isinstance(input_size, np.ndarray):
            # 如果input_size不是ndarray就會到這裡，將input_size轉成ndarray型態
            input_size = np.array(input_size)
        if input_size.size > 1:
            # 如果input_size的size大於1就一定要是2
            assert len(input_size) == 2
        else:
            # 將input_size變成[input_size, input_size]型態
            input_size = np.array([input_size, input_size], dtype=np.int)
        # 獲取當前圖像資料
        img = results['img']

        # 獲取當前圖像高寬
        h, w, _ = img.shape

        # calculate the size for min_scale
        # 獲取最小輸入高寬值，_ceil_to_multiples_of是獲取可以被64整除且離輸入的數字最近的函數
        min_input_w = _ceil_to_multiples_of(self.min_scale * input_size[0], 64)
        min_input_h = _ceil_to_multiples_of(self.min_scale * input_size[1], 64)
        if w < h:
            # 如果寬度比高度短就會到這裡
            w_resized = int(min_input_w * self.current_scale / self.min_scale)
            h_resized = int(
                _ceil_to_multiples_of(min_input_w / w * h, 64) *
                self.current_scale / self.min_scale)
            if self.use_udp:
                scale_w = w - 1.0
                scale_h = (h_resized - 1.0) / (w_resized - 1.0) * (w - 1.0)
            else:
                scale_w = w / 200.0
                scale_h = h_resized / w_resized * w / 200.0
        else:
            # 如果高度比寬度短就會到這裡
            # 獲取resize後的高寬值，這裡會將h_resize往min_input_h縮放
            h_resized = int(min_input_h * self.current_scale / self.min_scale)
            w_resized = int(
                _ceil_to_multiples_of(min_input_h / h * w, 64) *
                self.current_scale / self.min_scale)
            if self.use_udp:
                # 如果有使用udp就會到這裡
                scale_h = h - 1.0
                scale_w = (w_resized - 1.0) / (h_resized - 1.0) * (h - 1.0)
            else:
                # 如果沒有使用udp就會到這裡，計算scale_h與scale_w的值
                scale_h = h / 200.0
                scale_w = w_resized / h_resized * h / 200.0
        if self.use_udp:
            # 如果有使用udp就會到這裡，獲取精確的center
            center = (scale_w / 2.0, scale_h / 2.0)
        else:
            # 如果沒有使用udp就會到這裡
            center = np.array([round(w / 2.0), round(h / 2.0)])
        # 將資料保存到ann_info當中
        results['ann_info']['test_scale_factor'] = self.test_scale_factor
        results['ann_info']['base_size'] = (w_resized, h_resized)
        results['ann_info']['center'] = center
        results['ann_info']['scale'] = np.array([scale_w, scale_h])

        # 最終將results回傳
        return results


@PIPELINES.register_module()
class BottomUpResizeAlign:
    """Resize multi-scale size and align transform for bottom-up.

    Args:
        transforms (List): ToTensor & Normalize
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, transforms, use_udp=False):
        """ 進行多尺度變換，同時進行對齊變換，這裡是專門給bottom-up模型
        Args:
            transforms: 一系列處理流
            use_udp: 是否使用udp
        """
        # 將圖像處理流用Compose包裝
        self.transforms = Compose(transforms)
        if use_udp:
            # 如果使用udp就會到這裏
            self._resize_align_multi_scale = _resize_align_multi_scale_udp
        else:
            # 否則就會到這裡
            self._resize_align_multi_scale = _resize_align_multi_scale

    def __call__(self, results):
        """Resize multi-scale size and align transform for bottom-up."""
        # 進行多尺度變換，同時進行對齊變換，這裡是專門給bottom-up模型
        # 獲取輸入到模型當中的圖像大小
        input_size = results['ann_info']['image_size']
        # 處理input_size資料
        if not isinstance(input_size, np.ndarray):
            input_size = np.array(input_size)
        if input_size.size > 1:
            assert len(input_size) == 2
        else:
            input_size = np.array([input_size, input_size], dtype=np.int)
        # 獲取test的縮放比例
        test_scale_factor = results['ann_info']['test_scale_factor']
        # 保存數據增強的資料
        aug_data = []

        # 遍歷縮放比例，這裡會由大到小排序
        for _, s in enumerate(sorted(test_scale_factor, reverse=True)):
            # 拷貝一份results到_results當中
            _results = results.copy()
            # 將資料傳到_resize_align_multi_scale當中，image_resized會是變換後的圖像，高寬會發生變化
            image_resized, _, _ = self._resize_align_multi_scale(_results['img'], input_size, s, min(test_scale_factor))
            # 將img資料更新到results當中
            _results['img'] = image_resized
            # 將results通過一系列流
            _results = self.transforms(_results)
            # 將_results當中圖像資料提取出來，並且在第0個維度擴圍
            transformed_img = _results['img'].unsqueeze(0)
            # 保存到aug_data當中
            aug_data.append(transformed_img)

        # 將經過數據增強的圖像放到results當中
        results['ann_info']['aug_data'] = aug_data

        # 將更新好的results返回
        return results
