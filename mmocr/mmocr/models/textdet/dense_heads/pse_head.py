# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import HEADS
from . import PANHead


@HEADS.register_module()
class PSEHead(PANHead):
    """The class for PSENet head.

    Args:
        in_channels (list[int]): A list of 4 numbers of input channels.
        out_channels (int): Number of output channels.
        downsample_ratio (float): Downsample ratio.
        loss (dict): Configuration dictionary for loss type. Supported loss
            types are "PANLoss" and "PSELoss".
        postprocessor (dict): Config of postprocessor for PSENet.
        train_cfg, test_cfg (dict): Depreciated.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample_ratio=0.25,
                 loss=dict(type='PSELoss'),
                 postprocessor=dict(
                     type='PSEPostprocessor', text_repr_type='poly'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        """ 已看過，PSENet的預測頭
        Args:
            in_channels: 輸入的channel深度，會是list且list長度就會是輸入的特徵圖數量
            out_channels: 輸出的channel深度
            downsample_ratio: 下採樣倍率
            loss: 損失函數的設定
            postprocessor: 預處理方式
            train_cfg: train的設定
            test_cfg: test的設定
            init_cfg: 初始化方式
        """

        # 繼承自PANHead，將繼承對象進行初始化
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            downsample_ratio=downsample_ratio,
            loss=loss,
            postprocessor=postprocessor,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs)
