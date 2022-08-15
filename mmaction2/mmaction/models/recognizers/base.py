# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16

from .. import builder


class BaseRecognizer(nn.Module, metaclass=ABCMeta):
    """Base class for recognizers.

    All recognizers should subclass it.
    All subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.

    Args:
        backbone (dict): Backbone modules to extract feature.
        cls_head (dict | None): Classification head to process feature.
            Default: None.
        neck (dict | None): Neck for feature fusion. Default: None.
        train_cfg (dict | None): Config for training. Default: None.
        test_cfg (dict | None): Config for testing. Default: None.
    """

    def __init__(self,
                 backbone,
                 cls_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None):
        """ 已看過，動作判別模型的基底，初始化函數
        Args:
            backbone: 骨幹特徵提取構建資料
            cls_head: 分類頭設定資料
            neck: 從backbone輸出的特徵有需要可以通過neck進行調整，neck設定資料
            train_cfg: 在train當中的額外設定資料
            test_cfg: 在test當中的額外設定資料
        """
        # 繼承自nn.Module，對繼承對象進行初始化
        super().__init__()
        # record the source of the backbone
        # 記錄下backbone的來源，這裡預設會是mmaction2
        self.backbone_from = 'mmaction2'

        if backbone['type'].startswith('mmcls.'):
            # 如果backbone的type是mmcls.為開頭的就會到這裡
            try:
                # 嘗試導入mmcls模組
                import mmcls.models.builder as mmcls_builder
            except (ImportError, ModuleNotFoundError):
                # 如果沒有辦法導入mmcls模組就會到這裡報錯，提示需要安裝mmcls來使用指定的backbone模塊
                # 這裡官方沒有直接給出mmcls的源碼，須透過pip install mmcls進行安裝
                raise ImportError('Please install mmcls to use this backbone.')
            # 更新backbone的type，將前面的mmcls.部分拿掉
            backbone['type'] = backbone['type'][6:]
            # 透過mmcls_builder進行構建backbone實例化對象
            self.backbone = mmcls_builder.build_backbone(backbone)
            # 將backbone的來源指定到mmcls上
            self.backbone_from = 'mmcls'
        elif backbone['type'].startswith('torchvision.'):
            # 如果backbone的type是torchvision.為開頭的就會到這裡
            try:
                # 嘗試導入torchvision模組
                import torchvision.models
            except (ImportError, ModuleNotFoundError):
                # 如果無法導入torchvision就會報錯，這裡會需要安裝torchvision
                raise ImportError('Please install torchvision to use this '
                                  'backbone.')
            # 將type當中的torchvision.去除，獲取真正需要的模型名稱
            backbone_type = backbone.pop('type')[12:]
            # 直接透過torchvision.models獲取模型實例化對象
            self.backbone = torchvision.models.__dict__[backbone_type](
                **backbone)
            # disable the classifier
            # 這裡將最終的分類部分使用Identity進行替代
            self.backbone.classifier = nn.Identity()
            # 全連接層部分也進行替代
            self.backbone.fc = nn.Identity()
            # 將backbone的來源指定到torchvision上
            self.backbone_from = 'torchvision'
        elif backbone['type'].startswith('timm.'):
            # 如果backbone的type是timm.為開頭的就會到這裡
            try:
                # 嘗試導入timm模組
                import timm
            except (ImportError, ModuleNotFoundError):
                # 如果無法導入timm模組就會報錯，這裡會需要安裝timm
                raise ImportError('Please install timm to use this '
                                  'backbone.')
            # 將type當中的timm.去除，獲取真正需要的模型迷稱
            backbone_type = backbone.pop('type')[5:]
            # disable the classifier
            # 將backbone當中的num_classes設定成0，這樣在構建時就不會有分類頭
            backbone['num_classes'] = 0
            # 透過timm.create_model構建模型實例化對象
            self.backbone = timm.create_model(backbone_type, **backbone)
            # 將backbone的來源指定到timm上
            self.backbone_from = 'timm'
        else:
            # 其他backbone的type就是使用mmaction2的模型，透過mmaction2的builder進行構建模型實例化對象
            self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            # 如果有指定的neck模塊就會到這裡進行neck模塊的實例化
            self.neck = builder.build_neck(neck)

        # 構建分類頭實例對象
        self.cls_head = builder.build_head(cls_head) if cls_head else None

        # 保存train_cfg與test_cfg資料
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # aux_info is the list of tensor names beyond 'imgs' and 'label' which
        # will be used in train_step and val_step, data_batch should contain
        # these tensors
        # 構建一個aux_info的空間
        self.aux_info = []
        if train_cfg is not None and 'aux_info' in train_cfg:
            # 如果有給定train_cfg且當中有aux_info就會將當中的內容直接貼上
            self.aux_info = train_cfg['aux_info']
        # max_testing_views should be int
        # 將max_testing_views設定成None
        self.max_testing_views = None
        if test_cfg is not None and 'max_testing_views' in test_cfg:
            # 如果有給定test_cfg且當中有max_testing_views就會複製過去
            self.max_testing_views = test_cfg['max_testing_views']
            # 檢查max_testing_views需要是int格式
            assert isinstance(self.max_testing_views, int)

        if test_cfg is not None and 'feature_extraction' in test_cfg:
            # 如果test_cfg當中有feature_extraction就會將內容貼上去
            self.feature_extraction = test_cfg['feature_extraction']
        else:
            # 否則就將feature_extraction默認設定成False
            self.feature_extraction = False

        # mini-batch blending, e.g. mixup, cutmix, etc.
        # 將混合設定成None
        self.blending = None
        if train_cfg is not None and 'blending' in train_cfg:
            # 如果train_cfg當中有設定blending就會到這裡
            # 將build_from_cfg導入近來
            from mmcv.utils import build_from_cfg

            from mmaction.datasets.builder import BLENDINGS
            # 構建blending實例化對象
            self.blending = build_from_cfg(train_cfg['blending'], BLENDINGS)

        # 進行權重初始化
        self.init_weights()

        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the recognizer has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_cls_head(self):
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self):
        """Initialize the model network weights."""
        if self.backbone_from in ['mmcls', 'mmaction2']:
            self.backbone.init_weights()
        elif self.backbone_from in ['torchvision', 'timm']:
            warnings.warn('We do not initialize weights for backbones in '
                          f'{self.backbone_from}, since the weights for '
                          f'backbones in {self.backbone_from} are initialized'
                          'in their __init__ functions.')
        else:
            raise NotImplementedError('Unsupported backbone source '
                                      f'{self.backbone_from}!')

        if self.with_cls_head:
            self.cls_head.init_weights()
        if self.with_neck:
            self.neck.init_weights()

    @auto_fp16()
    def extract_feat(self, imgs):
        """Extract features through a backbone.

        Args:
            imgs (torch.Tensor): The input images.

        Returns:
            torch.tensor: The extracted features.
        """
        # 已看過，進行特徵提取
        # imgs = 一個batch的圖像資料，tensor shape [batch_size * num_clips, channel, clip_len, height, width]
        if (hasattr(self.backbone, 'features')
                and self.backbone_from == 'torchvision'):
            # 如果backbone當中有features層結構且backbone是由torchvision提供的就會到這裡
            x = self.backbone.features(imgs)
        elif self.backbone_from == 'timm':
            # 如果backbone是由timm提供的就會到這裡
            x = self.backbone.forward_features(imgs)
        elif self.backbone_from == 'mmcls':
            # 如果backbone是由mmcls提供的就會到這裡
            x = self.backbone(imgs)
            if isinstance(x, tuple):
                assert len(x) == 1
                x = x[0]
        else:
            # 其他的也就是mmaction2提供的就會到這裡，x shape = [batch_size * num_clips, channel, clip_len, height, width]
            # 這裡的channel與clip_len與height與width會與傳入時有所不同，因為透過提取會進行下採樣
            x = self.backbone(imgs)
        # 回傳提取好的特徵
        return x

    def average_clip(self, cls_score, num_segs=1):
        """Averaging class score over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score. Only called in test mode.

        Args:
            cls_score (torch.Tensor): Class score to be averaged.
            num_segs (int): Number of clips for each input sample.

        Returns:
            torch.Tensor: Averaged class score.
        """
        if 'average_clips' not in self.test_cfg.keys():
            raise KeyError('"average_clips" must defined in test_cfg\'s keys')

        average_clips = self.test_cfg['average_clips']
        if average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        if average_clips is None:
            return cls_score

        batch_size = cls_score.shape[0]
        cls_score = cls_score.view(batch_size // num_segs, num_segs, -1)

        if average_clips == 'prob':
            cls_score = F.softmax(cls_score, dim=2).mean(dim=1)
        elif average_clips == 'score':
            cls_score = cls_score.mean(dim=1)

        return cls_score

    @abstractmethod
    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""

    @abstractmethod
    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""

    @abstractmethod
    def forward_gradcam(self, imgs):
        """Defines the computation performed at every all when using gradcam
        utils."""

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

    def forward(self, imgs, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        # 已看過，準備進行正向傳遞
        if kwargs.get('gradcam', False):
            # 如果kwargs當中有gradcam就會到這裡
            del kwargs['gradcam']
            # 使用gradcam的forward函數
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            # 如果需要計算loss就會到這裡
            if label is None:
                # 如果沒有給正確的label就會報錯
                raise ValueError('Label should not be None.')
            if self.blending is not None:
                # 如果有設定blending就會到這裡
                imgs, label = self.blending(imgs, label)
            # 透過forward_train函數進行正向傳播
            return self.forward_train(imgs, label, **kwargs)

        # 如果是驗證或是測試就會走這裡，不去計算損失值也不用提供標註訊息
        return self.forward_test(imgs, **kwargs)

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
        # 已看過，開始進行一個batch的訓練
        # data_batch = 一個batch的訓練資料，dict型態
        # optimizer = 優化器實例對象
        # kwargs = 其他參數，通常為空

        # 將data_batch當中圖像資料提取出來，imgs shape = [batch_size, num_clip, channel, clip_len, height, width]
        imgs = data_batch['imgs']
        # 將data_batch當中每個影像資料的標註訊息提取出來，label shape = [batch_size, labels(一段影片可能會有多個標註)]
        label = data_batch['label']

        # 如果有aux_info資訊就會保存在這裡
        aux_info = {}
        # 遍歷aux_info資訊
        for item in self.aux_info:
            # 檢查data_batch當中需要有item資訊
            assert item in data_batch
            # 將data_batch的item資訊放到aux_info當中
            aux_info[item] = data_batch[item]

        # 進行正向傳遞與計算損失
        losses = self(imgs, label, return_loss=True, **aux_info)

        # 將獲取的損失進行整理壓縮
        loss, log_vars = self._parse_losses(losses)

        # 構建損失的dict
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        # 最後回傳損失值進行反向傳遞
        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        imgs = data_batch['imgs']
        label = data_batch['label']

        aux_info = {}
        for item in self.aux_info:
            aux_info[item] = data_batch[item]

        losses = self(imgs, label, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
