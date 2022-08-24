# ------------------------------------------------------------------------------
# Adapted from https://github.com/princeton-vl/pose-ae-train/
# Original licence: Copyright (c) 2017, umich-vl, under BSD 3-Clause License.
# ------------------------------------------------------------------------------

import numpy as np
import torch
from munkres import Munkres

from mmpose.core.evaluation import post_dark_udp


def _py_max_match(scores):
    """Apply munkres algorithm to get the best match.

    Args:
        scores(np.ndarray): cost matrix.

    Returns:
        np.ndarray: best match.
    """
    # 利用匈牙利算法獲取最佳匹配方式
    # 構建匈牙利實例對象
    m = Munkres()
    # 將scores進行計算進行計算
    # Munkres輸入為[工人數量, 工作數量]值表示需要的工資，最後分配的會是最小工資花費，這裡的工人需要小於等於工作數量
    # tmp = list[tuple]，tuple(第index工人, 第index工作)
    tmp = m.compute(scores)
    # 將tmp轉成ndarray並且轉成int格式
    tmp = np.array(tmp).astype(int)
    # 將分配結果回傳
    return tmp


def _match_by_tag(inp, params):
    """Match joints by tags. Use Munkres algorithm to calculate the best match
    for keypoints grouping.

    Note:
        number of keypoints: K
        max number of people in an image: M (M=30 by default)
        dim of tags: L
            If use flip testing, L=2; else L=1.

    Args:
        inp(tuple):
            tag_k (np.ndarray[KxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[KxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[KxM]): top k value of the
                feature maps per keypoint.
        params(Params): class Params().

    Returns:
        np.ndarray: result of pose groups.
    """
    """ 透過tag進行配對
    Args:
        inp: (tag_k: 標籤資訊，ndarray shape [num_joints, max_people, scale * 2],
              loc_k: 在熱力圖上獲取前max_people大的值的座標訊息，ndarray shape [num_joints, max_people, 2],
              val_k: 在熱力圖上前max_people大的值，ndarray shape [num_joints, max_people])
        params: 某個class類
    """
    # 檢查params是否為_Params類型
    assert isinstance(params, _Params), 'params should be class _Params()'

    # 將inp資料提取出來
    tag_k, loc_k, val_k = inp

    # 構建全0的ndarray且shape [num_joints, 3 + scale * 2]
    default_ = np.zeros((params.num_joints, 3 + tag_k.shape[2]), dtype=np.float32)

    # 構建joint與tag字典資料
    joint_dict = {}
    tag_dict = {}
    # 遍歷關節點數量
    for i in range(params.num_joints):
        # 獲取當前關節點對應的index
        idx = params.joint_order[i]

        # 將當前關節點的tag資訊取出來，tags shape = [max_people, scale * 2]
        tags = tag_k[idx]
        # 將loc_k與val_k與tags資訊全部concat起來作為joints，joints shape [max_people, 2 + 1 + (scale * 2)]
        joints = np.concatenate((loc_k[idx], val_k[idx, :, None], tags), 1)
        # 獲取mask資料，這裡會將置信度低於閾值的會設定成False其他會是True，mask shape [max_people]
        mask = joints[:, 2] > params.detection_threshold
        # 將tags與joints進行調整，M的大小會是有多少個True在mask當中，也就是有多少人的當前關節點是有被檢測到的
        tags = tags[mask]  # shape: [M, L]
        joints = joints[mask]  # shape: [M, 3 + L], 3: x, y, val

        if joints.shape[0] == 0:
            # 如果沒有剩下半個人的當前關節點就直接continue
            continue

        if i == 0 or len(joint_dict) == 0:
            # 如果當前i是0或是joint_dict是空的就會到這裡
            # 將tags與joints打包進行遍歷，也就是遍歷有檢測出的人數
            for tag, joint in zip(tags, joints):
                # 將tag[0]取出作為key值
                key = tag[0]
                # 使用setdefault查看joint_dict當中有沒有key，如果沒有就會生成key且value是default_
                # 會在對應的idx上將joint資料放入
                joint_dict.setdefault(key, np.copy(default_))[idx] = joint
                # 構建key在tag_dict上並且將tag存入到value當中
                tag_dict[key] = [tag]
        else:
            # 其他情況就會到這裡
            # shape: [M]
            # 獲取當前joint_dict有的key值，grouped_keys = list[float]，list長度會是M
            grouped_keys = list(joint_dict.keys())
            if params.ignore_too_much:
                # 如果有啟用ignore_too_much就會到這裡，將超過的key值刪除
                grouped_keys = grouped_keys[:params.max_num_people]
            # shape: [M, L]
            # 提取對應的tag資料，grouped_tags = list[ndarray]，ndarray shape [scale * 2]
            grouped_tags = [np.mean(tag_dict[i], axis=0) for i in grouped_keys]

            # shape: [M, M, L]
            # diff = 當前joints與已經存在的grouped_tags計算差值，shape [M, M, L]表示的是兩兩的差
            diff = joints[:, None, 3:] - np.array(grouped_tags)[None, :, :]
            # shape: [M, M]
            # 對於diff在維度2上取2泛數，也就是歐式距離，最終就可以知道差距值，diff_normed shape [M, M]
            diff_normed = np.linalg.norm(diff, ord=2, axis=2)
            # 將差距值保存到diff_saved當中
            diff_saved = np.copy(diff_normed)

            if params.use_detection_val:
                # 如果有設定使用use_detection_val就會到這裡
                diff_normed = np.round(diff_normed) * 100 - joints[:, 2:3]

            # 獲取當前關節點有偵測到的人數
            num_added = diff.shape[0]
            # 獲取當前已經有保存的人數
            num_grouped = diff.shape[1]

            if num_added > num_grouped:
                # 如果當前有偵測到的人數大於已經保存的人數就會到這裡
                # 將diff_normed在axis=1地方添加ndarray shape [num_added, num_added - num_grouped]且值為inf
                diff_normed = np.concatenate(
                    (diff_normed, np.zeros((num_added, num_added - num_grouped), dtype=np.float32) + 1e10), axis=1)

            # pairs = list[tuple]，tuple(第index工人, 第index工作)
            pairs = _py_max_match(diff_normed)
            # 開始對於分配進行遍歷
            for row, col in pairs:
                if (row < num_added and col < num_grouped
                        and diff_saved[row][col] < params.tag_threshold):
                    # 如果row與col在合法範圍且diff_saved的值小於tag_threshold就會到這裡
                    # 獲取對應的key值
                    key = grouped_keys[col]
                    # 將joint_dict對應的位置將joints資訊放入
                    joint_dict[key][idx] = joints[row]
                    # 將tag_dict對應key值添加tags資訊上去
                    tag_dict[key].append(tags[row])
                else:
                    # 其他情況，也就是要新增人的關節點
                    # 獲取新的key值
                    key = tags[row][0]
                    # 新增key直到joint_dict當中並且將joints資訊放入
                    joint_dict.setdefault(key, np.copy(default_))[idx] = joints[row]
                    # 新增tag_dict資訊
                    tag_dict[key] = [tags[row]]

    # 獲取joint_dict_keys當中所有的key值
    joint_dict_keys = list(joint_dict.keys())
    if params.ignore_too_much:
        # 如果有啟用ignore_too_much就會到這裡，只會取出前max_people個key值
        # The new person joints beyond the params.max_num_people will be
        # ignored, for the dict is in ordered when python > 3.6 version.
        joint_dict_keys = joint_dict_keys[:params.max_num_people]
    # 將結果提取出來變成results，這裡只會拿joint_dict資訊
    results = np.array([joint_dict[i] for i in joint_dict_keys]).astype(np.float32)
    # results shape = [people, num_joints, (x, y, score, tag[0], tag[1])]
    return results


class _Params:
    """A class of parameter.

    Args:
        cfg(Config): config.
    """

    def __init__(self, cfg):
        """ 一個clas的參數
        Args:
            cfg: 設定檔資料
        """
        # 獲取總共有多少關節點
        self.num_joints = cfg['num_joints']
        # 總共可以偵測多少個人物
        self.max_num_people = cfg['max_num_people']

        # 偵測的閾值
        self.detection_threshold = cfg['detection_threshold']
        # 標註的閾值
        self.tag_threshold = cfg['tag_threshold']
        # 是否使用偵測的值
        self.use_detection_val = cfg['use_detection_val']
        # 當過多時是否忽略
        self.ignore_too_much = cfg['ignore_too_much']

        if self.num_joints == 17:
            # 如果關節點數量是17就會到這裡，構建關節點的順序，這裡比較亂
            self.joint_order = [
                i - 1 for i in
                [1, 2, 3, 4, 5, 6, 7, 12, 13, 8, 9, 10, 11, 14, 15, 16, 17]
            ]
        else:
            # 其他的話就會是依照順序來
            self.joint_order = list(np.arange(self.num_joints))


class HeatmapParser:
    """The heatmap parser for post processing."""

    def __init__(self, cfg):
        """ 構建熱力圖
        Args:
            cfg: 設定參數值
        """
        # 透過Params將cfg整理
        self.params = _Params(cfg)
        # 獲取是否tag每個關節點
        self.tag_per_joint = cfg['tag_per_joint']
        # 構建池化層
        self.pool = torch.nn.MaxPool2d(cfg['nms_kernel'], 1,
                                       cfg['nms_padding'])
        # 是否使用udp
        self.use_udp = cfg.get('use_udp', False)
        # 是否使用score_per_joint
        self.score_per_joint = cfg.get('score_per_joint', False)

    def nms(self, heatmaps):
        """Non-Maximum Suppression for heatmaps.

        Args:
            heatmaps(torch.Tensor): Heatmaps before nms.

        Returns:
            torch.Tensor: Heatmaps after nms.
        """
        # 進行NMS處理，heatmaps就是熱力圖資料

        # 將heatmaps通過pool最大池化
        maxm = self.pool(heatmaps)
        # 獲取哪些地方的值與池化後的值相同，那些就會是需要保存下的
        maxm = torch.eq(maxm, heatmaps).float()
        # 熱力圖乘上mask就可以獲取最後熱力圖
        heatmaps = heatmaps * maxm

        # 回傳通過mask後的熱力圖
        return heatmaps

    def match(self, tag_k, loc_k, val_k):
        """Group keypoints to human poses in a batch.

        Args:
            tag_k (np.ndarray[NxKxMxL]): tag corresponding to the
                top k values of feature map per keypoint.
            loc_k (np.ndarray[NxKxMx2]): top k locations of the
                feature maps for keypoint.
            val_k (np.ndarray[NxKxM]): top k value of the
                feature maps per keypoint.

        Returns:
            list
        """
        """ 進行配對
        Args:
            tag_k: 標籤資訊，ndarray shape [batch_size, num_joints, max_people, scale * 2]
            loc_k: 在熱力圖上獲取前max_people大的值的座標訊息，ndarray shape [batch_size, num_joints, max_people, 2]
            val_k: 在熱力圖上前max_people大的值，ndarray shape [batch_size, num_joints, max_people]
        """

        def _match(x):
            # x = zip(tag_k, loc_k, val_k)
            # 返回值，results shape = [people, num_joints, (x, y, score, tag[0], tag[1])]
            return _match_by_tag(x, self.params)

        # 將tag_k與loc_k與val_k與_match放到map當中進行處理，map的第一個參數是函數後面的是要可以iterate的
        # 最後list長度會是batch_size裡面的資訊就會是_match_by_tag的返回值
        return list(map(_match, zip(tag_k, loc_k, val_k)))

    def top_k(self, heatmaps, tags):
        """Find top_k values in an image.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            max number of people: M
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (torch.Tensor[NxKxHxW])
            tags (torch.Tensor[NxKxHxWxL])

        Returns:
            dict: A dict containing top_k values.

            - tag_k (np.ndarray[NxKxMxL]):
                tag corresponding to the top k values of
                feature map per keypoint.
            - loc_k (np.ndarray[NxKxMx2]):
                top k location of feature map per keypoint.
            - val_k (np.ndarray[NxKxM]):
                top k value of feature map per keypoint.
        """
        """ 獲取前k大的值
        Args:
            heatmaps: 熱力圖資料，tensor shape [batch_size, num_joints, height, width]
            tags: 標籤資料，tensor shape [batch_size, num_joints, height, width, scale * 2]
        """
        # 將熱力圖資料通過nms處理，會將經過抑制的部分變成0其他維持原本
        heatmaps = self.nms(heatmaps)
        # 獲取熱力圖的size(batch_size, num_joints, height, width)
        N, K, H, W = heatmaps.size()
        # 將熱力圖進行通道調整 [batch_size, num_joints, height * width]
        heatmaps = heatmaps.view(N, K, -1)
        # 獲取前max_num_people的值，這裡回傳的會有值以及index，val_k與ind的shape [batch_size, num_joints, max_people]
        val_k, ind = heatmaps.topk(self.params.max_num_people, dim=2)

        # 調整tags通道 [batch_size, num_joints, height * width, scale * 2]
        tags = tags.view(tags.size(0), tags.size(1), W * H, -1)
        if not self.tag_per_joint:
            # 如果沒有tag_per_joint就會到這裡
            tags = tags.expand(-1, self.params.num_joints, -1, -1)

        # 使用gather獲取指定位置的tag資料，會遍歷tags最後一個維度的值，之後根據ind獲取需要的值
        # tag_k shape [batch_size, num_joints, max_people]
        tag_k = torch.stack([torch.gather(tags[..., i], 2, ind) for i in range(tags.size(3))], dim=3)

        # 因為我們將高寬壓平，所以先在的ind會是(x, y)融合結果
        # 透過%W獲取x的值，透過//W獲取對應的y值
        x = ind % W
        y = ind // W

        # 將(x, y)在dim=3進行stack，ind_k shape = [batch_size, num_joints, max_people, scale * 2]
        ind_k = torch.stack((x, y), dim=3)

        # 將tag_k與loc_k與val_k轉換成ndarray格式後用dict包裝
        results = {
            'tag_k': tag_k.cpu().numpy(),
            'loc_k': ind_k.cpu().numpy(),
            'val_k': val_k.cpu().numpy()
        }

        # 回傳包裝結果
        return results

    @staticmethod
    def adjust(results, heatmaps):
        """Adjust the coordinates for better accuracy.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W

        Args:
            results (list(np.ndarray)): Keypoint predictions.
            heatmaps (torch.Tensor[NxKxHxW]): Heatmaps.
        """
        """ 調整座標獲取更好的準確率
        Args:
            results: list[ndarray]，list長度會是batch_size，ndarray [people, num_joints, (x, y, score, tag[0], tag[1])]
            heatmaps: 熱力圖資料，tensor shape [batch_size, num_joints, height, width]
        """
        # 獲取熱力圖的shape資訊
        _, _, H, W = heatmaps.shape
        # 遍歷整個batch圖像
        for batch_id, people in enumerate(results):
            # 遍歷一張圖像當中的人物資料
            for people_id, people_i in enumerate(people):
                # 遍歷人物當中的關節點資料
                for joint_id, joint in enumerate(people_i):
                    if joint[2] > 0:
                        # 如果該關節點置信度大於0就會到這裡
                        # 獲取指定的(x, y)座標
                        x, y = joint[0:2]
                        # 將座標點直接變成int格式
                        xx, yy = int(x), int(y)
                        # 獲取對應人物對應關節點的預測熱力圖，tmp shape [height, width]
                        tmp = heatmaps[batch_id][joint_id]
                        # 進行x與y的微調
                        if tmp[min(H - 1, yy + 1), xx] > tmp[max(0, yy - 1), xx]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[yy, min(W - 1, xx + 1)] > tmp[yy, max(0, xx - 1)]:
                            x += 0.25
                        else:
                            x -= 0.25
                        # 最後更新調整後的(x, y)座標
                        results[batch_id][people_id, joint_id, 0:2] = (x + 0.5, y + 0.5)
        # 回傳調整後的results
        return results

    @staticmethod
    def refine(heatmap, tag, keypoints, use_udp=False):
        """Given initial keypoint predictions, we identify missing joints.

        Note:
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmap: np.ndarray(K, H, W).
            tag: np.ndarray(K, H, W) |  np.ndarray(K, H, W, L)
            keypoints: np.ndarray of size (K, 3 + L)
                        last dim is (x, y, score, tag).
            use_udp: bool-unbiased data processing

        Returns:
            np.ndarray: The refined keypoints.
        """
        """ 對於一個某個人物的關節點如果置信度是0就會找該關節點的熱力圖當中置信度最大的值作為該關節點的座標
        Args:
            heatmap: 熱力圖資料，ndarray shape [num_joints, height, width]
            tag: 標籤資料，ndarray shape [num_joints, height, width, scale * 2]
            keypoints: 關節點資料，ndarray shape [num_joints, (x, y, val, tag[0], tag[1])]
            use_udp: 是否使用udp
        """

        # 獲取熱力圖的shape資訊(num_joints, height, width)
        K, H, W = heatmap.shape
        if len(tag.shape) == 3:
            # 如果tag的shape是3就會在最後添加一個維度
            tag = tag[..., None]

        # tags的保存list
        tags = []
        # 遍歷所有關節點
        for i in range(K):
            if keypoints[i, 2] > 0:
                # 如果該關節點的置信度超過0就會到這裡
                # save tag value of detected keypoint
                # 獲取(x, y)座標並且轉成int格式
                x, y = keypoints[i][:2].astype(int)
                # 限制(x, y)的範圍
                x = np.clip(x, 0, W - 1)
                y = np.clip(y, 0, H - 1)
                # 將指定的tag保存到tags當中
                tags.append(tag[i, y, x])

        # mean tag of current detected people
        # 取平均標籤，prev_tag shape [scale * 2]
        prev_tag = np.mean(tags, axis=0)
        # 最終回傳的results
        results = []

        # 將熱力圖以及標籤一起遍歷，這裡遍歷長度會是關節點數量
        for _heatmap, _tag in zip(heatmap, tag):
            # distance of all tag values with mean tag of
            # current detected people
            # 計算標籤距離，distance_tag shape = [height, width]
            distance_tag = (((_tag - prev_tag[None, None, :])**2).sum(axis=2)**0.5)
            # 獲取標準化後的熱力圖資料，norm_heatmap shape = [height, width]
            norm_heatmap = _heatmap - np.round(distance_tag)

            # find maximum position
            # 透過unravel_index獲取最大值的位置
            y, x = np.unravel_index(np.argmax(norm_heatmap), _heatmap.shape)
            # 將座標拷貝一份到(xx, yy)當中
            xx = x.copy()
            yy = y.copy()
            # detection score at maximum position
            # 獲取在熱力圖上的值
            val = _heatmap[y, x]
            if not use_udp:
                # 如果沒有使用udp就會到這裡
                # offset by 0.5
                # 將x與y偏移0.5
                x += 0.5
                y += 0.5

            # add a quarter offset
            # 進行(x, y)的偏移
            if _heatmap[yy, min(W - 1, xx + 1)] > _heatmap[yy, max(0, xx - 1)]:
                x += 0.25
            else:
                x -= 0.25

            if _heatmap[min(H - 1, yy + 1), xx] > _heatmap[max(0, yy - 1), xx]:
                y += 0.25
            else:
                y -= 0.25

            # 將結果保存到results當中
            results.append((x, y, val))
        # 將results轉成ndarray
        results = np.array(results)

        if results is not None:
            # 如果results當中有資料就會到這裡，遍歷當中的關節點位置
            for i in range(K):
                # add keypoint if it is not detected
                if results[i, 2] > 0 and keypoints[i, 2] == 0:
                    # 如果置信度大於0且關節點的置信度是0就會到這裡，將keypoint座標更新
                    keypoints[i, :3] = results[i, :3]

        # 回傳更新後的值
        return keypoints

    def parse(self, heatmaps, tags, adjust=True, refine=True):
        """Group keypoints into poses given heatmap and tag.

        Note:
            batch size: N
            number of keypoints: K
            heatmap height: H
            heatmap width: W
            dim of tags: L
                If use flip testing, L=2; else L=1.

        Args:
            heatmaps (torch.Tensor[NxKxHxW]): model output heatmaps.
            tags (torch.Tensor[NxKxHxWxL]): model output tagmaps.

        Returns:
            tuple: A tuple containing keypoint grouping results.

            - results (list(np.ndarray)): Pose results.
            - scores (list/list(np.ndarray)): Score of people.
        """
        """ 將關節點進行分組，根據給定的熱力圖以及標籤
        Args:
            heatmaps: 熱力圖資料，tensor shape [batch_size, num_joints, height, width]
            tags: 標籤資料，tensor shape [batch_size, num_joints, height, width, num_scale * 2]
            adjust: 細微調整關節點座標位置
            refine:
        """
        # 先經過top_k後進行match
        # results = list[ndarray]，list長度會是batch_size，ndarray = [people, num_joints, (x, y, score, tag[0], tag[1])]
        results = self.match(**self.top_k(heatmaps, tags))

        if adjust:
            # 如果有需要進行adjust就會到這裡
            if self.use_udp:
                # 如果使用udp就會到這裡
                for i in range(len(results)):
                    if results[i].shape[0] > 0:
                        results[i][..., :2] = post_dark_udp(
                            results[i][..., :2].copy(), heatmaps[i:i + 1, :])
            else:
                # 沒有使用udp就會到這裡，透過adjust進行調整
                results = self.adjust(results, heatmaps)

        if self.score_per_joint:
            # 如果需要每個關節點的置信度分數就會到這裡，scores shape [people, num_joints]
            scores = [i[:, 2] for i in results[0]]
        else:
            # 如果只需要整體的置信度分數就會到這裡，scores shape [people]
            scores = [i[:, 2].mean() for i in results[0]]

        if refine:
            # 如果有使用refine就會到這裡
            # 提取出results資訊
            results = results[0]
            # for every detected person
            # 遍歷有檢測到的人物
            for i in range(len(results)):
                # 將熱力圖轉成ndarray，這裡會把batch_size去除
                heatmap_numpy = heatmaps[0].cpu().numpy()
                # 將tag資訊轉成ndarray，這裡會把batch_size去除
                tag_numpy = tags[0].cpu().numpy()
                if not self.tag_per_joint:
                    # 如果沒有設定tag_per_joint就會到這裡
                    tag_numpy = np.tile(tag_numpy, (self.params.num_joints, 1, 1, 1))
                # 將資料放到refine當中進行處理
                results[i] = self.refine(heatmap_numpy, tag_numpy, results[i], use_udp=self.use_udp)
            # 最後將results用list包裝
            results = [results]

        # 回傳results以及scores資訊
        return results, scores
