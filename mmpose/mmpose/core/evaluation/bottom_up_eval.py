# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

from mmpose.core.post_processing import (get_warp_matrix, transform_preds,
                                         warp_affine_joints)


def split_ae_outputs(outputs, num_joints, with_heatmaps, with_ae,
                     select_output_index):
    """Split multi-stage outputs into heatmaps & tags.

    Args:
        outputs (list(Tensor)): Outputs of network
        num_joints (int): Number of joints
        with_heatmaps (list[bool]): Option to output
            heatmaps for different stages.
        with_ae (list[bool]): Option to output
            ae tags for different stages.
        select_output_index (list[int]): Output keep the selected index

    Returns:
        tuple: A tuple containing multi-stage outputs.

        - list[Tensor]: multi-stage heatmaps.
        - list[Tensor]: multi-stage tags.
    """
    """ 將輸出結果分成熱力圖資料以及標籤資料
    Args:
        outputs: 預測輸出資料，list[tensor]，tensor shape [batch_size, channel, height, width]
        num_joints: 總共有多少個關節點
        with_heatmaps: 該輸出tensor是否包含熱力圖輸出
        with_ae: 該輸出tensor是否包含ae輸出
        select_output_index: 選擇輸出的index
    """

    # 保存熱力圖的list
    heatmaps = []
    # 保存tags的list
    tags = []

    # aggregate heatmaps from different stages
    # 遍歷輸出的特徵圖數量
    for i, output in enumerate(outputs):
        if i not in select_output_index:
            # 如果該index不需要回傳就會跳過
            continue
        # staring index of the associative embeddings
        # channel切割點，如果當前層tensor當中沒有包含熱力圖就會是0否則就會是關節點數量
        offset_feat = num_joints if with_heatmaps[i] else 0
        if with_heatmaps[i]:
            # 如果當前層有熱力圖就會到這裡，取出前關節點數量的channel作為熱力圖資料
            heatmaps.append(output[:, :num_joints])
        if with_ae[i]:
            # 如果ae輸出就會到這裡，接續offset_feat後提取
            tags.append(output[:, offset_feat:])

    # 回傳熱力圖以及tags資料
    # heatmaps shape = tensor [batch_size, num_joints, height, width]
    # tags shape = tensor [batch_size, num_joints, height, width]
    return heatmaps, tags


def flip_feature_maps(feature_maps, flip_index=None):
    """Flip the feature maps and swap the channels.

    Args:
        feature_maps (list[Tensor]): Feature maps.
        flip_index (list[int] | None): Channel-flip indexes.
            If None, do not flip channels.

    Returns:
        list[Tensor]: Flipped feature_maps.
    """
    """ 將特徵圖進行翻轉
    Args:
        feature_maps: 需要進行翻轉的特徵圖，list[tensor]，tensor shape [batch_size, channel, height, width]
        flip_index: 關節點對稱關節點index
    """

    # 翻轉後的特徵圖保存list
    flipped_feature_maps = []
    # 遍歷所有特徵圖
    for feature_map in feature_maps:
        # 將特徵圖進行翻轉
        feature_map = torch.flip(feature_map, [3])
        if flip_index is not None:
            # 如果有傳入flip_index就會在channel維度上進行排序，這樣就可以變回正常的關節點名稱
            flipped_feature_maps.append(feature_map[:, flip_index, :, :])
        else:
            # 沒有就不需要對channel維度進行調整
            flipped_feature_maps.append(feature_map)

    # 將flipped_feature_maps回傳
    return flipped_feature_maps


def _resize_average(feature_maps, align_corners, index=-1, resize_size=None):
    """Resize the feature maps and compute the average.

    Args:
        feature_maps (list[Tensor]): Feature maps.
        align_corners (bool): Align corners when performing interpolation.
        index (int): Only used when `resize_size' is None.
            If `resize_size' is None, the target size is the size
            of the indexed feature maps.
        resize_size (list[int, int]): The target size [w, h].

    Returns:
        list[Tensor]: Averaged feature_maps.
    """
    """ 將特徵圖進行resize並且計算平均值
    Args:
        feature_maps: 特徵圖資料，list[tensor]，tensor shape [batch_size, channel, height, width]
        align_corners: 執行插值時對齊角
        index: 當resize_size是None時會使用，將傳入特徵圖都變成指定傳入特徵圖的index
        resize_size: 指定resize大小
    """

    if feature_maps is None:
        # 如果feature_maps是None就直接回傳None
        return None
    # 保存均值的地方，初始化成0
    feature_maps_avg = 0

    # 進行特徵圖resize
    feature_map_list = _resize_concate(feature_maps, align_corners, index=index, resize_size=resize_size)
    # 將所有值加起來
    for feature_map in feature_map_list:
        feature_maps_avg += feature_map

    # 取平均
    feature_maps_avg /= len(feature_map_list)
    # 將取平均後的結果回傳
    return [feature_maps_avg]


def _resize_unsqueeze_concat(feature_maps,
                             align_corners,
                             index=-1,
                             resize_size=None):
    """Resize, unsqueeze and concatenate the feature_maps.

    Args:
        feature_maps (list[Tensor]): Feature maps.
        align_corners (bool): Align corners when performing interpolation.
        index (int): Only used when `resize_size' is None.
            If `resize_size' is None, the target size is the size
            of the indexed feature maps.
        resize_size (list[int, int]): The target size [w, h].

    Returns:
        list[Tensor]: Averaged feature_maps.
    """
    """ 進行resize後用concat進行拼接
    Args:
        feature_maps: 特徵圖資料，list[tensor]，tensor shape [batch_size, num_joints, height, width]
        align_corners: 差值算法的參數
        index: 如果沒有設定resize_size就會根據feature_maps[index]決定resize後的大小
        resize_size: 直接指定resize後的大小
    """
    if feature_maps is None:
        # 如果傳入的feature_maps是None就會直接回傳None
        return None
    # 進行_resize_concat部分
    feature_map_list = _resize_concate(feature_maps, align_corners, index=index, resize_size=resize_size)

    # 獲取特徵圖的dim
    feat_dim = len(feature_map_list[0].shape) - 1
    # 將feature_map_list在最後維度進行stack，output_feature_maps shape [batch_size, num_joints, height, width, len(list)]
    output_feature_maps = torch.cat(
        [torch.unsqueeze(fmap, dim=feat_dim + 1) for fmap in feature_map_list],
        dim=feat_dim + 1)
    return [output_feature_maps]


def _resize_concate(feature_maps, align_corners, index=-1, resize_size=None):
    """Resize and concatenate the feature_maps.

    Args:
        feature_maps (list[Tensor]): Feature maps.
        align_corners (bool): Align corners when performing interpolation.
        index (int): Only used when `resize_size' is None.
            If `resize_size' is None, the target size is the size
            of the indexed feature maps.
        resize_size (list[int, int]): The target size [w, h].

    Returns:
        list[Tensor]: Averaged feature_maps.
    """
    """ 將特徵圖進行resize後concat
    Args:
        feature_maps: 特徵圖資料，list[tensor]，tensor shape [batch_size, channel, height, width]
        align_corners: 差值時的設定資料
        resize_size: 要resize的指定大小
    """
    if feature_maps is None:
        # 如果傳入的feature_maps是None就直接回傳None
        return None

    # 保存特徵圖的list
    feature_map_list = []

    if index < 0:
        # 如果index設定為-1就會到這裡，將index設定成最後一個特徵圖的index
        index += len(feature_maps)

    if resize_size is None:
        # 如果沒有傳入resize_size就會到這裡，根據index找到指定的大小
        resize_size = (feature_maps[index].size(2),
                       feature_maps[index].size(3))

    # 遍歷所有特徵圖
    for feature_map in feature_maps:
        # 獲取當前特徵圖大小
        ori_size = (feature_map.size(2), feature_map.size(3))
        if ori_size != resize_size:
            # 如果當前大小與指定大小不同就會透過差值算法進行縮放
            feature_map = torch.nn.functional.interpolate(
                feature_map,
                size=resize_size,
                mode='bilinear',
                align_corners=align_corners)

        # 將結果保存
        feature_map_list.append(feature_map)

    # 回傳結果
    return feature_map_list


def aggregate_stage_flip(feature_maps,
                         feature_maps_flip,
                         index=-1,
                         project2image=True,
                         size_projected=None,
                         align_corners=False,
                         aggregate_stage='concat',
                         aggregate_flip='average'):
    """Inference the model to get multi-stage outputs (heatmaps & tags), and
    resize them to base sizes.

    Args:
        feature_maps (list[Tensor]): feature_maps can be heatmaps,
            tags, and pafs.
        feature_maps_flip (list[Tensor] | None): flipped feature_maps.
            feature maps can be heatmaps, tags, and pafs.
        project2image (bool): Option to resize to base scale.
        size_projected (list[int, int]): Base size of heatmaps [w, h].
        align_corners (bool): Align corners when performing interpolation.
        aggregate_stage (str): Methods to aggregate multi-stage feature maps.
            Options: 'concat', 'average'. Default: 'concat.

            - 'concat': Concatenate the original and the flipped feature maps.
            - 'average': Get the average of the original and the flipped
                feature maps.
        aggregate_flip (str): Methods to aggregate the original and
            the flipped feature maps. Options: 'concat', 'average', 'none'.
            Default: 'average.

            - 'concat': Concatenate the original and the flipped feature maps.
            - 'average': Get the average of the original and the flipped
                feature maps..
            - 'none': no flipped feature maps.

    Returns:
        list[Tensor]: Aggregated feature maps with shape [NxKxWxH].
    """
    """ 進行融合
    Args:
        feature_maps: 原始圖像預測資料，list[tensor]，tensor shape [batch_size, num_joints, height, width]
        feature_maps_flip: 翻轉圖像預測資料，list[tensor]，tensor shape [batch_size, num_joints, height, width]
        index:
        project2image: 是否需要將結果縮放回原始圖像大小
        size_projected: 原始圖像大小
        align_corners: 是否要用align_corners
        aggregate_stage: 聚合多階段特徵圖的方法
        aggregate_flip: 聚合原始特徵圖和翻轉特徵圖的方法
    """

    if feature_maps_flip is None:
        # 如果翻轉特徵圖是None就將融合翻轉的方式設定成none
        aggregate_flip = 'none'

    # 最終輸出特徵圖的保存list
    output_feature_maps = []

    if aggregate_stage == 'average':
        # 如果融合stage的方式是average就會到這裡
        _aggregate_stage_func = _resize_average
    elif aggregate_stage == 'concat':
        # 如果融合stage的方式是concat就會到這裡
        _aggregate_stage_func = _resize_concate
    else:
        # 其他融合方式就會直接報錯
        NotImplementedError()

    # 如果有需要縮放回原圖且有給定原圖大小就會進入
    if project2image and size_projected:
        # 將原始的特徵圖進行縮放，同時將特徵圖融合
        _origin = _aggregate_stage_func(
            feature_maps,
            align_corners,
            index=index,
            resize_size=(size_projected[1], size_projected[0]))

        # 將翻轉的特徵圖進行縮放，同時將特徵圖融合
        _flipped = _aggregate_stage_func(
            feature_maps_flip,
            align_corners,
            index=index,
            resize_size=(size_projected[1], size_projected[0]))
    else:
        _origin = _aggregate_stage_func(
            feature_maps, align_corners, index=index, resize_size=None)
        _flipped = _aggregate_stage_func(
            feature_maps_flip, align_corners, index=index, resize_size=None)

    if aggregate_flip == 'average':
        # 如果融合翻轉以及不翻轉的特徵圖是average就會到這裡
        # 檢查是否傳入翻轉的特徵圖
        assert feature_maps_flip is not None
        # 將原始特徵圖與翻轉特徵圖包裝
        for _ori, _fli in zip(_origin, _flipped):
            # 進行相加取平均
            output_feature_maps.append((_ori + _fli) / 2.0)

    elif aggregate_flip == 'concat':
        # 如果是透過concat就會到這裡
        assert feature_maps_flip is not None
        # 直接添加到output_feature_maps當中
        output_feature_maps.append(*_origin)
        output_feature_maps.append(*_flipped)

    elif aggregate_flip == 'none':
        # 如果是none就會到這裡
        if isinstance(_origin, list):
            output_feature_maps.append(*_origin)
        else:
            output_feature_maps.append(_origin)
    else:
        # 其他就會報錯
        NotImplementedError()

    # 將結果進行回傳
    return output_feature_maps


def aggregate_scale(feature_maps_list,
                    align_corners=False,
                    aggregate_scale='average'):
    """Aggregate multi-scale outputs.

    Note:
        batch size: N
        keypoints num : K
        heatmap width: W
        heatmap height: H

    Args:
        feature_maps_list (list[Tensor]): Aggregated feature maps.
        project2image (bool): Option to resize to base scale.
        align_corners (bool): Align corners when performing interpolation.
        aggregate_scale (str): Methods to aggregate multi-scale feature maps.
            Options: 'average', 'unsqueeze_concat'.

            - 'average': Get the average of the feature maps.
            - 'unsqueeze_concat': Concatenate the feature maps along new axis.
                Default: 'average.

    Returns:
        Tensor: Aggregated feature maps.
    """
    """ 將多尺度的輸出進行融合
    Args:
        feature_maps_list: 特徵圖資料，list[tensor]，tensor shape [batch_size, channel, height, width]
        align_corners: 差值算法的參數
        aggregate_scale: 融合方式
    """

    if aggregate_scale == 'average':
        # 如果融合方式是averager就會到這裡
        output_feature_maps = _resize_average(feature_maps_list, align_corners, index=0, resize_size=None)

    elif aggregate_scale == 'unsqueeze_concat':
        # 如果是用concat就會到這裡
        output_feature_maps = _resize_unsqueeze_concat(feature_maps_list, align_corners, index=0, resize_size=None)
    else:
        # 其他選擇就會直接報錯
        NotImplementedError()

    # 回傳融合後的結果
    return output_feature_maps[0]


def get_group_preds(grouped_joints,
                    center,
                    scale,
                    heatmap_size,
                    use_udp=False):
    """Transform the grouped joints back to the image.

    Args:
        grouped_joints (list): Grouped person joints.
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        heatmap_size (np.ndarray[2, ]): Size of the destination heatmaps.
        use_udp (bool): Unbiased data processing.
             Paper ref: Huang et al. The Devil is in the Details: Delving into
             Unbiased Data Processing for Human Pose Estimation (CVPR'2020).

    Returns:
        list: List of the pose result for each person.
    """
    """ 將分組好的關節點映射回圖像當中
    Args:
        grouped_joints: 分組後的關節點資訊，list[ndarray]
                        ndarray shape [num_people, num_joints, (x, y, score, tag[0], tag[1]]，list長度會是batch_size
        center: 標註匡中心位置，ndarray shape[2]
        scale: 邊界匡比例
        heatmap_size: 熱力圖大小
        use_udp: 是否使用udp
    """
    if len(grouped_joints) == 0:
        # 如果沒有檢查到任何關節點資訊就直接回傳空
        return []

    if use_udp:
        # 如果有使用udp就會到這裡
        if grouped_joints[0].shape[0] > 0:
            heatmap_size_t = np.array(heatmap_size, dtype=np.float32) - 1.0
            trans = get_warp_matrix(
                theta=0,
                size_input=heatmap_size_t,
                size_dst=scale,
                size_target=heatmap_size_t)
            grouped_joints[0][..., :2] = \
                warp_affine_joints(grouped_joints[0][..., :2], trans)
        results = [person for person in grouped_joints[0]]
    else:
        # 沒有使用udp就會到這裡
        # 構建最後回傳的list
        results = []
        # 遍歷一個人的所有關節點資訊
        for person in grouped_joints[0]:
            # 透過transform_preds獲取最終的關節點位置
            joints = transform_preds(person, center, scale, heatmap_size)
            # 將結果放到results當中
            results.append(joints)

    # 回傳results資訊
    return results
