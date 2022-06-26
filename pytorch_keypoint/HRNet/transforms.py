import math
import random
from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


def flip_images(img):
    # 已看過
    assert len(img.shape) == 4, 'images has to be [batch_size, channels, height, width]'
    # 對維度3的部分進行翻轉
    img = torch.flip(img, dims=[3])
    return img


def flip_back(output_flipped, matched_parts):
    """
    :param output_flipped: 預測的結果 shape [batch_size, num_pks, height, width]
    :param matched_parts: 左右對照的index
    :return:
    """
    # 已看過
    # 檢查傳入資料是否有誤
    assert len(output_flipped.shape) == 4, 'output_flipped has to be [batch_size, num_joints, height, width]'
    # 在最後一個維度進行翻轉
    output_flipped = torch.flip(output_flipped, dims=[3])

    # 除了要將左右翻轉回來，原先代表左半部分的熱力圖也要跟右邊的熱力圖翻轉，這樣才是完整的翻轉回來
    for pair in matched_parts:
        tmp = output_flipped[:, pair[0]].clone()
        output_flipped[:, pair[0]] = output_flipped[:, pair[1]]
        output_flipped[:, pair[1]] = tmp

    return output_flipped


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    # 已看過
    # batch_heatmaps shape [batch_size, num_kps, height, width]，為tensor格式
    # 檢查是否符合格式
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    # 獲取資訊
    batch_size, num_joints, h, w = batch_heatmaps.shape
    # 將高寬維度進行壓縮，heatmaps_reshaped shape [batch_size, num_kps, height * width]
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    # 在第二維度上面做取最大值，這裡會返回最大值的值以及最大值的index
    # maxvals, idx shape [batch_size, num_kps]
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)

    # maxvals shape [batch_size, num_kps, 1]
    maxvals = maxvals.unsqueeze(dim=-1)
    # idx shape [batch_size, num_kps]，且為float
    idx = idx.float()

    # 構建一個全為0的tensor shape [batch_size, num_kps, 2]
    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps)

    # 一個keypoint對應到熱力圖上的座標(x, y)
    preds[:, :, 0] = idx % w  # column 对应最大值的x坐标
    preds[:, :, 1] = torch.floor(idx / w)  # row 对应最大值的y坐标

    # torch.gt的用法，傳入兩個tensor，比較兩個tensor相同位置上的值，如果tensor1的值比tensor2的值大就會是1否則就是0
    # 所以torch.gt回傳的shape會與傳入的相同
    # 首先先看最大值是否有比0還要大，如果大於0就會在該位置是True表示有效點，最後會進行擴維，後面的2表示(x, y)都是True
    # pred_mask shape [batch_size, num_kps, 2]
    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    # 如果預測出來的最大值是0會被過濾掉
    preds *= pred_mask
    # preds shape [batch_size, num_kps, 2]
    # maxvlas shape [batch_size, num_kps, 1]
    return preds, maxvals


def affine_points(pt, t):
    """
    :param pt: 關節點
    :param t: 放射變換參數
    :return:
    """
    # 已看過
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt = np.concatenate([pt, ones], axis=1).T
    new_pt = np.dot(t, pt)
    # 回傳新的關節點位置
    return new_pt.T


def get_final_preds(batch_heatmaps: torch.Tensor,
                    trans: list = None,
                    post_processing: bool = False):
    """
    :param batch_heatmaps: 從模型中預測出來的結果 shape [batch_size, num_kps, height, width]
    :param trans: 將熱力圖轉換回原圖的參數
    :param post_processing: 預設會是True
    :return:
    """
    # 已看過
    # 檢查是否有傳入轉換參數
    assert trans is not None
    # coords shape [batch_size, num_kps, 2]
    # maxvlas shape [batch_size, num_kps, 1]
    coords, maxvals = get_max_preds(batch_heatmaps)

    # 取得熱力圖高寬
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    # 熱裡預設為True
    if post_processing:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                # 獲取熱力圖hm shape [height, width]
                hm = batch_heatmaps[n][p]
                # 獲取該熱力圖中最大值的座標位置
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                # 我們會對於最大值地方進行上下左右微調，會依據最大值附近的值去調整
                # 這裡如果不調整對於結果也不會有太大影響，但是調整過後準確率會高一點
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = torch.tensor(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ]
                    ).to(batch_heatmaps.device)
                    coords[n][p] += torch.sign(diff) * .25

    # 將最後座標位置轉換成numpy格式
    # preds shape numpy [batch_size, num_kps, 2]
    preds = coords.clone().cpu().numpy()

    # Transform back
    # 將座標轉換回原始圖片上
    for i in range(coords.shape[0]):
        preds[i] = affine_points(preds[i], trans[i])

    # 回傳回去
    # preds shape [batch_size, num_kps, 2]
    # maxvals shape [batch_size, num_kps, 1]
    return preds, maxvals.cpu().numpy()


def decode_keypoints(outputs, origin_hw, num_joints: int = 17):
    keypoints = []
    scores = []
    heatmap_h, heatmap_w = outputs.shape[-2:]
    for i in range(num_joints):
        pt = np.unravel_index(np.argmax(outputs[i]), (heatmap_h, heatmap_w))
        score = outputs[i, pt[0], pt[1]]
        keypoints.append(pt[::-1])  # hw -> wh(xy)
        scores.append(score)

    keypoints = np.array(keypoints, dtype=float)
    scores = np.array(scores, dtype=float)
    # convert to full image scale
    keypoints[:, 0] = np.clip(keypoints[:, 0] / heatmap_w * origin_hw[1],
                              a_min=0,
                              a_max=origin_hw[1])
    keypoints[:, 1] = np.clip(keypoints[:, 1] / heatmap_h * origin_hw[0],
                              a_min=0,
                              a_max=origin_hw[0])
    return keypoints, scores


def resize_pad(img: np.ndarray, size: tuple):
    h, w, c = img.shape
    src = np.array([[0, 0],       # 原坐标系中图像左上角点
                    [w - 1, 0],   # 原坐标系中图像右上角点
                    [0, h - 1]],  # 原坐标系中图像左下角点
                   dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    if h / w > size[0] / size[1]:
        # 需要在w方向padding
        wi = size[0] * (w / h)
        pad_w = (size[1] - wi) / 2
        dst[0, :] = [pad_w - 1, 0]            # 目标坐标系中图像左上角点
        dst[1, :] = [size[1] - pad_w - 1, 0]  # 目标坐标系中图像右上角点
        dst[2, :] = [pad_w - 1, size[0] - 1]  # 目标坐标系中图像左下角点
    else:
        # 需要在h方向padding
        hi = size[1] * (h / w)
        pad_h = (size[0] - hi) / 2
        dst[0, :] = [0, pad_h - 1]            # 目标坐标系中图像左上角点
        dst[1, :] = [size[1] - 1, pad_h - 1]  # 目标坐标系中图像右上角点
        dst[2, :] = [0, size[0] - pad_h - 1]  # 目标坐标系中图像左下角点

    trans = cv2.getAffineTransform(src, dst)  # 计算正向仿射变换矩阵
    # 对图像进行仿射变换
    resize_img = cv2.warpAffine(img,
                                trans,
                                size[::-1],  # w, h
                                flags=cv2.INTER_LINEAR)
    # import matplotlib.pyplot as plt
    # plt.imshow(resize_img)
    # plt.show()

    dst /= 4  # 网络预测的heatmap尺寸是输入图像的1/4
    reverse_trans = cv2.getAffineTransform(dst, src)  # 计算逆向仿射变换矩阵，方便后续还原

    return resize_img, reverse_trans


def adjust_box(xmin: float, ymin: float, w: float, h: float, fixed_size: Tuple[float, float]):
    """通过增加w或者h的方式保证输入图片的长宽比固定"""
    # 已看過
    # 透過左上角點加上寬高可以獲得右下角點
    xmax = xmin + w
    ymax = ymin + h

    # 計算高寬比例
    hw_ratio = fixed_size[0] / fixed_size[1]
    if h / w > hw_ratio:
        # 需要在w方向padding
        # 將w padding成wi寬度就可以讓比例正確
        wi = h / hw_ratio
        # 計算左邊需要padding多少空間
        pad_w = (wi - w) / 2
        # 進行padding，也就是對x方向的左右兩邊進行延伸
        xmin = xmin - pad_w
        xmax = xmax + pad_w
    else:
        # 需要在h方向padding
        # 將h padding成hi的高度就可以讓比例正確
        hi = w * hw_ratio
        # 計算上方需要padding多少空間
        pad_h = (hi - h) / 2
        # 進行padding，也就是對y方向的上下兩邊進行延伸
        ymin = ymin - pad_h
        ymax = ymax + pad_h

    # 回傳新的左上角座標以及右下角座標
    return xmin, ymin, xmax, ymax


def scale_box(xmin: float, ymin: float, w: float, h: float, scale_ratio: Tuple[float, float]):
    # 已看過
    """根据传入的h、w缩放因子scale_ratio，重新计算xmin，ymin，w，h"""
    # 計算新的高寬
    s_h = h * scale_ratio[0]
    s_w = w * scale_ratio[1]
    # 更新左上角點
    xmin = xmin - (s_w - w) / 2.
    ymin = ymin - (s_h - h) / 2.
    # 回傳
    return xmin, ymin, s_w, s_h


def plot_heatmap(image, heatmap, kps, kps_weights):
    for kp_id in range(len(kps_weights)):
        if kps_weights[kp_id] > 0:
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.plot(*kps[kp_id].tolist(), "ro")
            plt.title("image")
            plt.subplot(1, 2, 2)
            plt.imshow(heatmap[kp_id], cmap=plt.cm.Blues)
            plt.colorbar(ticks=[0, 1])
            plt.title(f"kp_id: {kp_id}")
            plt.show()


class Compose(object):
    """组合多个transform函数"""
    # 已看過
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    # 已看過
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        # 已看過
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        # 已看過
        # 將圖像標準化
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class HalfBody(object):
    def __init__(self, p: float = 0.3, upper_body_ids=None, lower_body_ids=None):
        """
        :param p: 啟用HalfBody的概率，預設概率為0.3
        :param upper_body_ids: 上半身的index
        :param lower_body_ids: 下半身的index
        """
        # 已看過
        assert upper_body_ids is not None
        assert lower_body_ids is not None
        # 賦值
        self.p = p
        self.upper_body_ids = upper_body_ids
        self.lower_body_ids = lower_body_ids

    def __call__(self, image, target):
        # 已看過
        # 有機率觸發
        if random.random() < self.p:
            # kps = numpy shape [17, 2]
            kps = target["keypoints"]
            # vis = numpy shape [17]
            vis = target["visible"]
            upper_kps = []
            lower_kps = []

            # 对可见的keypoints进行归类
            for i, v in enumerate(vis):
                # 如果是不可見的會是0
                if v > 0.5:
                    # 判斷是上半身或是下半身，保存該部分的座標
                    if i in self.upper_body_ids:
                        upper_kps.append(kps[i])
                    else:
                        lower_kps.append(kps[i])

            # 50%的概率选择上或下半身
            if random.random() < 0.5:
                selected_kps = upper_kps
            else:
                selected_kps = lower_kps

            # 如果点数太少就不做任何处理
            if len(selected_kps) > 2:
                selected_kps = np.array(selected_kps, dtype=np.float32)
                # 找到所選點中左上角以及右下角
                xmin, ymin = np.min(selected_kps, axis=0).tolist()
                xmax, ymax = np.max(selected_kps, axis=0).tolist()
                # 計算出寬高
                w = xmax - xmin
                h = ymax - ymin
                # 至少要有一定大小
                if w > 1 and h > 1:
                    # 把w和h适当放大点，要不然关键点处于边缘位置
                    # 傳入左上角寬高以及要擴充的大小，就會返回新的左上角位置以及寬高
                    xmin, ymin, w, h = scale_box(xmin, ymin, w, h, (1.5, 1.5))
                    # 更新box
                    target["box"] = [xmin, ymin, w, h]

        return image, target


class AffineTransform(object):
    """scale+rotation"""
    def __init__(self,
                 scale: Tuple[float, float] = None,  # e.g. (0.65, 1.35)
                 rotation: Tuple[int, int] = None,   # e.g. (-45, 45)
                 fixed_size: Tuple[int, int] = (256, 192)):
        # 已看過
        # 將圖像進行縮放以及旋轉還有最後輸入的大小
        self.scale = scale
        self.rotation = rotation
        self.fixed_size = fixed_size

    def __call__(self, img, target):
        # 已看過
        # 將target中的box訊息傳遞到adjust_box當中，同時也將指定輸出大小傳入
        # 回傳左上角座標以及右下角座標(xmin, ymin, xmax, ymax)，輸出的高寬比會跟fixed_size比例相同，但大小不一定會是fixed_size
        src_xmin, src_ymin, src_xmax, src_ymax = adjust_box(*target["box"], self.fixed_size)
        # 計算寬高
        src_w = src_xmax - src_xmin
        src_h = src_ymax - src_ymin
        # 計算中心點位置
        src_center = np.array([(src_xmin + src_xmax) / 2, (src_ymin + src_ymax) / 2])
        # 中間偏上部分
        src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
        # 中間偏右部分
        src_p3 = src_center + np.array([src_w / 2, 0])   # right middle

        # 不確定是做什麼用的
        dst_center = np.array([(self.fixed_size[1] - 1) / 2, (self.fixed_size[0] - 1) / 2])
        dst_p2 = np.array([(self.fixed_size[1] - 1) / 2, 0])  # top middle
        dst_p3 = np.array([self.fixed_size[1] - 1, (self.fixed_size[0] - 1) / 2])  # right middle

        if self.scale is not None:
            # 隨機縮放，從一個範圍內隨機挑出縮放係數
            scale = random.uniform(*self.scale)
            # 進行縮放
            src_w = src_w * scale
            src_h = src_h * scale
            # 計算新的中間偏上部分
            src_p2 = src_center + np.array([0, -src_h / 2])  # top middle
            # 計算新的中間偏右部分
            src_p3 = src_center + np.array([src_w / 2, 0])   # right middle

        if self.rotation is not None:
            # 進行隨機旋轉
            # 在給定的範圍指定一個旋轉角度
            angle = random.randint(*self.rotation)  # 角度制
            angle = angle / 180 * math.pi  # 弧度制
            # 重新計算位置找到座標位置
            src_p2 = src_center + np.array([src_h / 2 * math.sin(angle), -src_h / 2 * math.cos(angle)])
            src_p3 = src_center + np.array([src_w / 2 * math.cos(angle), src_w / 2 * math.sin(angle)])

        # src, dst = numpy shape [3, 2]
        src = np.stack([src_center, src_p2, src_p3]).astype(np.float32)
        dst = np.stack([dst_center, dst_p2, dst_p3]).astype(np.float32)

        # 給定3個點就可以進行放射變換
        trans = cv2.getAffineTransform(src, dst)  # 计算正向仿射变换矩阵
        dst /= 4  # 网络预测的heatmap尺寸是输入图像的1/4
        reverse_trans = cv2.getAffineTransform(dst, src)  # 计算逆向仿射变换矩阵，方便后续还原

        # 对图像进行仿射变换
        # 這裡輸出的圖像就是要輸入到網路的大小，所以跟fixed_size一樣大
        resize_img = cv2.warpAffine(img,
                                    trans,
                                    tuple(self.fixed_size[::-1]),  # [w, h]
                                    flags=cv2.INTER_LINEAR)

        # 對關節點也需要調整
        if "keypoints" in target:
            kps = target["keypoints"]
            # 構建出一個mask表示哪些關節點都不是0，mask = [True, True, False, ..., True]，長度為關節點數量
            mask = np.logical_and(kps[:, 0] != 0, kps[:, 1] != 0)
            # 將過濾後的關節點以及放射變換參數帶入
            kps[mask] = affine_points(kps[mask], trans)
            # 更新關節點
            target["keypoints"] = kps

        # import matplotlib.pyplot as plt
        # from draw_utils import draw_keypoints
        # resize_img = draw_keypoints(resize_img, target["keypoints"])
        # plt.imshow(resize_img)
        # plt.show()

        # 記錄下放射變換方式，以及變回來的方法
        target["trans"] = trans
        target["reverse_trans"] = reverse_trans
        # 輸出的image是已經符合輸入網路大小
        return resize_img, target


class RandomHorizontalFlip(object):
    """随机对输入图片进行水平翻转，注意该方法必须接在 AffineTransform 后"""
    def __init__(self, p: float = 0.5, matched_parts: list = None):
        # 已看過
        # 水平翻轉，傳入資料有左右對應的index，例如左手以及右手會是一組
        assert matched_parts is not None
        self.p = p
        self.matched_parts = matched_parts

    def __call__(self, image, target):
        # 已看過
        if random.random() < self.p:
            # [h, w, c]
            # 有機率的隨機水平翻轉
            image = np.ascontiguousarray(np.flip(image, axis=[1]))
            # 將關節點資料拿出來
            keypoints = target["keypoints"]
            visible = target["visible"]
            width = image.shape[1]

            # Flip horizontal
            # 最後面需要扣ㄧ，是因為是0~寬度減一
            keypoints[:, 0] = width - keypoints[:, 0] - 1

            # Change left-right parts
            # 經過翻轉後左右會相反
            for pair in self.matched_parts:
                keypoints[pair[0], :], keypoints[pair[1], :] = \
                    keypoints[pair[1], :], keypoints[pair[0], :].copy()

                visible[pair[0]], visible[pair[1]] = \
                    visible[pair[1]], visible[pair[0]].copy()

            # 將新的資料更新上去
            target["keypoints"] = keypoints
            target["visible"] = visible

        return image, target


class KeypointToHeatMap(object):
    def __init__(self,
                 heatmap_hw: Tuple[int, int] = (256 // 4, 192 // 4),
                 gaussian_sigma: int = 2,
                 keypoints_weights=None):
        """
        :param heatmap_hw: 熱力圖大小
        :param gaussian_sigma: 超參數
        :param keypoints_weights: 關鍵點權重
        """
        # 已看過
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel_radius = self.sigma * 3
        # 預設會是True，因為會傳入keypoints_weights
        self.use_kps_weights = False if keypoints_weights is None else True
        self.kps_weights = keypoints_weights

        # generate gaussian kernel(not normalized)
        # kernel_size預設為7
        kernel_size = 2 * self.kernel_radius + 1
        # kernel = 7*7全為零的矩陣
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        # x_center = y_center = 3
        x_center = y_center = kernel_size // 2
        # 透過公式將值放入kernel中
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))
        # print(kernel)

        self.kernel = kernel

    def __call__(self, image, target):
        # 已看過
        # 這裡是要讓正樣本不是只有一個點，要讓正樣本是一個從準確的點往外擴散出去
        # 取出關節點
        kps = target["keypoints"]
        # 取出有多少個關節點
        num_kps = kps.shape[0]
        # 構建一個全為一且長度等於關節點數量的陣列
        kps_weights = np.ones((num_kps,), dtype=np.float32)
        # kps_weights裡面的值會等於該關節點是屬於哪中類型(0或1或2)
        if "visible" in target:
            visible = target["visible"]
            kps_weights = visible

        # heatmap = numpy格式shape [可偵測關節點數量, 熱力圖高, 熱力圖寬]，初始化都為0
        heatmap = np.zeros((num_kps, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        # 將關節點位置除以4再加0.5後取上界，因為heatmap比輸入圖像小了4倍所以target中的kps也要除以4
        heatmap_kps = (kps / 4 + 0.5).astype(np.int)  # round
        for kp_id in range(num_kps):
            v = kps_weights[kp_id]
            if v < 0.5:
                # 如果该点的可见度很低，则直接忽略
                continue

            # 取出(x, y)資訊
            x, y = heatmap_kps[kp_id]
            # 左上角以及右下角
            ul = [x - self.kernel_radius, y - self.kernel_radius]  # up-left x,y
            br = [x + self.kernel_radius, y + self.kernel_radius]  # bottom-right x,y
            # 如果以xy为中心kernel_radius为半径的辐射范围内与heatmap没交集，则忽略该点(该规则并不严格)
            # 確認範圍沒有越界，理論上這裡是不會進去到continue除非標記有問題
            if ul[0] > self.heatmap_hw[1] - 1 or \
                    ul[1] > self.heatmap_hw[0] - 1 or \
                    br[0] < 0 or \
                    br[1] < 0:
                # If not, just return the image as is
                # 將有問題的設定成0表示該點無法偵測
                kps_weights[kp_id] = 0
                continue

            # Usable gaussian range
            # 计算高斯核有效区域（高斯核坐标系）
            g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
            g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])
            # image range
            # 计算heatmap中的有效区域（heatmap坐标系）
            img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
            img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

            if kps_weights[kp_id] > 0.5:
                # 将高斯核有效区域复制到heatmap对应区域
                heatmap[kp_id][img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] = \
                    self.kernel[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]

        if self.use_kps_weights:
            # self.kps_weight = 每一個關節點對應的權重
            kps_weights = np.multiply(kps_weights, self.kps_weights)

        # plot_heatmap(image, heatmap, kps, kps_weights)

        # 轉成tensor格式並且放入target當中，如此就可以得到高斯分佈的熱力圖，就可以擴充正樣本範圍讓模型更容易學習
        target["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)
        target["kps_weights"] = torch.as_tensor(kps_weights, dtype=torch.float32)

        return image, target
