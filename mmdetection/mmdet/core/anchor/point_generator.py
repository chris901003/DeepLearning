# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch.nn.modules.utils import _pair

from .builder import PRIOR_GENERATORS


@PRIOR_GENERATORS.register_module()
class PointGenerator:

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_points(self, featmap_size, stride=16, device='cuda'):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0., feat_w, device=device) * stride
        shift_y = torch.arange(0., feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        stride = shift_x.new_full((shift_xx.shape[0], ), stride)
        shifts = torch.stack([shift_xx, shift_yy, stride], dim=-1)
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid


@PRIOR_GENERATORS.register_module()
class MlvlPointGenerator:
    """Standard points generator for multi-level (Mlvl) feature maps in 2D
    points-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        offset (float): The offset of points, the value is normalized with
            corresponding stride. Defaults to 0.5.
    """

    def __init__(self, strides, offset=0.5):
        """ 對於多尺度特徵圖進行生成標準點
        Args:
            strides: 與原圖的縮小倍率
            offset: 偏移量
        """
        # 將傳入的strides保存且這裡會是[(stride, stride)]進行保存
        self.strides = [_pair(stride) for stride in strides]
        self.offset = offset

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (points) at a point
        on the feature grid"""
        return [1 for _ in range(len(self.strides))]

    def _meshgrid(self, x, y, row_major=True):
        # 使用meshgrid獲取二維座標
        yy, xx = torch.meshgrid(y, x)
        # 壓平後回傳
        if row_major:
            # warning .flatten() would cause error in ONNX exporting
            # have to use reshape here
            return xx.reshape(-1), yy.reshape(-1)

        else:
            return yy.reshape(-1), xx.reshape(-1)

    def grid_priors(self,
                    featmap_sizes,
                    dtype=torch.float32,
                    device='cuda',
                    with_stride=False):
        """Generate grid points of multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str): The device where the anchors will be put on.
            with_stride (bool): Whether to concatenate the stride to
                the last dimension of points.

        Return:
            list[torch.Tensor]: Points of  multiple feature levels.
            The sizes of each tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        """ 生成多尺度特徵圖的網格點
        Args:
            featmap_sizes: 特徵圖大小，list[torch.Size]，list長度就是不同尺度的大小
            dtype: 數據類型
            device: 訓練設備
            with_stride: 是否將步幅連接到點的最後一個維度
        """

        # 初始化時設定的層數要與傳入的相同
        assert self.num_levels == len(featmap_sizes)
        # 保存結果的list
        multi_level_priors = []
        # 遍歷多層的特徵圖
        for i in range(self.num_levels):
            # 透過single_level_grid_priors構建
            priors = self.single_level_grid_priors(
                featmap_sizes[i],
                level_idx=i,
                dtype=dtype,
                device=device,
                with_stride=with_stride)
            # 將結果保存
            multi_level_priors.append(priors)
        # 回傳結果，list[tensor]，list長度就會是不同尺度的特徵圖，tensor shape [feat_height * feat_height, 2或是4]
        return multi_level_priors

    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda',
                                 with_stride=False):
        """Generate grid Points of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps, arrange as
                (h, w).
            level_idx (int): The index of corresponding feature map level.
            dtype (:obj:`dtype`): Dtype of priors. Default: torch.float32.
            device (str, optional): The device the tensor will be put on.
                Defaults to 'cuda'.
            with_stride (bool): Concatenate the stride to the last dimension
                of points.

        Return:
            Tensor: Points of single feature levels.
            The shape of tensor should be (N, 2) when with stride is
            ``False``, where N = width * height, width and height
            are the sizes of the corresponding feature level,
            and the last dimension 2 represent (coord_x, coord_y),
            otherwise the shape should be (N, 4),
            and the last dimension 4 represent
            (coord_x, coord_y, stride_w, stride_h).
        """
        """ 對於單一一個特徵圖尺度構建網格圖
        Args:
            featmap_size: 特徵圖大小
            level_idx: 層數的index
            dtype: 數據類型
            device: 訓練設備
            with_stride: 是否將步幅連接到點的最後一個維度
        """
        # 獲取特徵圖的高寬資訊
        feat_h, feat_w = featmap_size
        # 獲取該層的特徵圖對於原圖的縮放比率
        stride_w, stride_h = self.strides[level_idx]
        # 構建範圍[0, feat_w]的數組並且放大stride_w倍，這樣就可以映射回原圖上的座標點
        shift_x = (torch.arange(0, feat_w, device=device) + self.offset) * stride_w
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        # 將數值型態進行轉換
        shift_x = shift_x.to(dtype)

        # 構建範圍[0, feat_h]的數組並且放大stride_h倍，這樣就可以映射回原圖上的座標點
        shift_y = (torch.arange(0, feat_h, device=device) + self.offset) * stride_h
        # keep featmap_size as Tensor instead of int, so that we
        # can convert to ONNX correctly
        # 將數值進行轉換
        shift_y = shift_y.to(dtype)
        # 使用meshgrid變成2維座標，shift_xx與shift_yy shape [width * height]，對應index配對可以獲取從左至右從上往下的座標
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        if not with_stride:
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)
        else:
            # use `shape[0]` instead of `len(shift_xx)` for ONNX export
            # 構建與shift_xx相同的tensor且全部填充為stride_w
            stride_w = shift_xx.new_full((shift_xx.shape[0], ), stride_w).to(dtype)
            # 構建與shift_yy相同的tensor且全部填充為stride_h
            stride_h = shift_xx.new_full((shift_yy.shape[0], ), stride_h).to(dtype)
            # 將資料進行拼接，這裡會在最後一個維度進行拼接，tensor shape [height * width, 4]
            shifts = torch.stack([shift_xx, shift_yy, stride_w, stride_h], dim=-1)
        # 將資料放到設備上
        all_points = shifts.to(device)
        return all_points

    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
        """Generate valid flags of points of multiple feature levels.

        Args:
            featmap_sizes (list(tuple)): List of feature map sizes in
                multiple feature levels, each size arrange as
                as (h, w).
            pad_shape (tuple(int)): The padded shape of the image,
                 arrange as (h, w).
            device (str): The device where the anchors will be put on.

        Return:
            list(torch.Tensor): Valid flags of points of multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            point_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / point_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / point_stride[0])), feat_w)
            flags = self.single_level_valid_flags((feat_h, feat_w),
                                                  (valid_feat_h, valid_feat_w),
                                                  device=device)
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(self,
                                 featmap_size,
                                 valid_size,
                                 device='cuda'):
        """Generate the valid flags of points of a single feature map.

        Args:
            featmap_size (tuple[int]): The size of feature maps, arrange as
                as (h, w).
            valid_size (tuple[int]): The valid size of the feature maps.
                The size arrange as as (h, w).
            device (str, optional): The device where the flags will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: The valid flags of each points in a single level \
                feature map.
        """
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        return valid

    def sparse_priors(self,
                      prior_idxs,
                      featmap_size,
                      level_idx,
                      dtype=torch.float32,
                      device='cuda'):
        """Generate sparse points according to the ``prior_idxs``.

        Args:
            prior_idxs (Tensor): The index of corresponding anchors
                in the feature map.
            featmap_size (tuple[int]): feature map size arrange as (w, h).
            level_idx (int): The level index of corresponding feature
                map.
            dtype (obj:`torch.dtype`): Date type of points. Defaults to
                ``torch.float32``.
            device (obj:`torch.device`): The device where the points is
                located.
        Returns:
            Tensor: Anchor with shape (N, 2), N should be equal to
            the length of ``prior_idxs``. And last dimension
            2 represent (coord_x, coord_y).
        """
        height, width = featmap_size
        x = (prior_idxs % width + self.offset) * self.strides[level_idx][0]
        y = ((prior_idxs // width) % height +
             self.offset) * self.strides[level_idx][1]
        prioris = torch.stack([x, y], 1).to(dtype)
        prioris = prioris.to(device)
        return prioris
