# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn

from .. import builder


class BaseGCN(nn.Module, metaclass=ABCMeta):
    """Base class for GCN-based action recognition.

    All GCN-based recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict | None): Classification head to process feature.
            Default: None.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
    """

    def __init__(self, backbone, cls_head=None, train_cfg=None, test_cfg=None):
        """ 所有基於骨架預測行為的模型都繼承與該類別
        Args:
            backbone: 特徵提取網路設定資料
            cls_head: 分類頭設定資料
            train_cfg: 在train模式下的額外設定
            test_cfg: 在test模式下的額外設定
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        # record the source of the backbone
        # 預設將backbone來源設定成mmaction2
        self.backbone_from = 'mmaction2'
        # 透過build_backbone構建backbone實例化對象
        self.backbone = builder.build_backbone(backbone)
        # 透過build_head構建分類頭實例化對象
        self.cls_head = builder.build_head(cls_head) if cls_head else None

        # 保存傳入參數
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # 初始化權重
        self.init_weights()

    @property
    def with_cls_head(self):
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        """Initialize the model network weights."""
        if self.backbone_from in ['mmcls', 'mmaction2']:
            self.backbone.init_weights()
        else:
            raise NotImplementedError('Unsupported backbone source '
                                      f'{self.backbone_from}!')

        if self.with_cls_head:
            self.cls_head.init_weights()

    @abstractmethod
    def forward_train(self, *args, **kwargs):
        """Defines the computation performed at training."""

    @abstractmethod
    def forward_test(self, *args):
        """Defines the computation performed at testing."""

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, keypoint, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        # 進行正向傳遞
        if return_loss:
            # 如果需要計算損失就會到這裡
            if label is None:
                # 如果沒有標籤訊息就會報錯
                raise ValueError('Label should not be None.')
            # 進行正向傳遞並且計算損失
            return self.forward_train(keypoint, label, **kwargs)

        # 如果不需計算損失就會到這裡
        return self.forward_test(keypoint, **kwargs)

    def extract_feat(self, skeletons):
        """Extract features through a backbone.

        Args:
            skeletons (torch.Tensor): The input skeletons.

        Returns:
            torch.tensor: The extracted features.
        """
        # 透過backbone進行特徵提取，x shape [batch_size * people, channel, frames, num_node]
        x = self.backbone(skeletons)
        # 將提取結果回傳
        return x

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        # GCN模型最底層的類的前向訓練函數，在複雜的模型當中會包含反向傳遞處理
        # data_batch = dataloader的輸出當中會有關節點資訊或是圖像資訊以及標註訊息
        # 將keypoint資訊提取出來，tensor shape [batch_size, channel(x, y, score), frames, num_node, people]
        skeletons = data_batch['keypoint']
        # 將label資訊提取出來，tensor shape [batch_size]
        label = data_batch['label']
        # 如果label的shape是[batch_size, 1]就會進行調整成[batch_size]
        label = label.squeeze(-1)

        # 進行正向傳遞並且計算損失
        losses = self(skeletons, label, return_loss=True)

        # 將loss值進行解析
        loss, log_vars = self._parse_losses(losses)
        # 最後用字典進行包裝
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(skeletons.data))

        # 輸出整理過後的損失字典
        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        skeletons = data_batch['keypoint']
        label = data_batch['label']

        losses = self(skeletons, label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(skeletons.data))

        return outputs
