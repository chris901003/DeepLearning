# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16


class BaseSegmentor(BaseModule, metaclass=ABCMeta):
    """Base class for segmentors."""
    # 繼承於BaseModule，同時BaseModule是MMCV共同繼承的Module，也就是pytorch當中的nn.Module一樣
    # BaseModule最後也是繼承torch.nn.Module，只是BaseModule有多了一些功能

    def __init__(self, init_cfg=None):
        """
        :param init_cfg: model config裏面的init_cfg，控制初始化用的
        """
        # 已看過
        # EncoderDecoder繼承於這個class
        # 初始化繼承的class
        super(BaseSegmentor, self).__init__(init_cfg)
        # fp16_enabled = 目前不確定是做什麼用的
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, img, img_metas):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """Placeholder for Forward function for training."""
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        """Placeholder for single image test."""
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        # 已看過，實際測試時會進入到這裡
        # imgs = 輸入圖像的tensor，shape [batch_size, channel, height, width]
        # img_metas = 圖像的相關資訊
        # kwargs = 裡面有rescale參數，這裡會是True

        # 遍歷傳入資料
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                # 檢查var是否為list格式
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        # num_augs = 有多少個batch
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            # 如果資訊不匹配這裡就會報錯
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        # 遍歷圖像資訊
        for img_meta in img_metas:
            # 獲取圖像原始大小
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            # 獲取當前圖像大小
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            # 獲取padding後的圖像大小
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            # 如果只有一張圖像會到這裡，我們只會把list的第一個傳入就可以了
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            # 在進行真正使用時我們不需要損失值，所以會往這裡
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
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

        """
        :param data_batch: 一個batch的資料，裡面包括訓練圖片的tensor以及標註的tensor 
        :param optimizer: 優化器
        :param kwargs: 其他參數通常都是空的
        :return: loss值
        """

        # 將data_batch傳入，這裡就會到層結構當中的forward當中
        # losses = dict格式，裏面就會有主損失以及主分支正確率，也會有輔助損失以及輔助正確率
        losses = self(**data_batch)
        # 將losses傳入到parse_losses進行解析
        # loss = 總損失值(float)
        # log_vars = 損失值以及正確率(dict{str:float})
        loss, log_vars = self._parse_losses(losses)

        # 再將outputs進行包裝，num_samples=batch_size
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']))

        return outputs

    def val_step(self, data_batch, optimizer=None, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        # 已看過，這裡是驗證集的部分
        # 進行向前傳遞並且將結果進行損失計算
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        log_vars_ = dict()
        for loss_name, loss_value in log_vars.items():
            k = loss_name + '_val'
            log_vars_[k] = loss_value

        outputs = dict(
            loss=loss,
            log_vars=log_vars_,
            num_samples=len(data_batch['img_metas']))

        return outputs

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
        # 已看過，整理loss字典當中的數值
        # losses = 損失的字典，裡面包括損失值以及正確率
        log_vars = OrderedDict()
        # 遍歷losses當中的資訊
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                # value是tensor格式就直接取均值
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                # value是list格式就取均值後相加
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                # 其他就直接報錯
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        # 計算總損失，這裡會將key當中含有loss的value全部相加，也就是正確率的部分不會被加進來
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, raise assertion error
        # to prevent GPUs from infinite waiting.
        if dist.is_available() and dist.is_initialized():
            # 分布式訓練會用到的
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()) + '\n')
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        # 將loss保存起來
        log_vars['loss'] = loss
        # 遍歷log_vars當中的內容
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            # 這裡將tensor的外皮去除，我們只需要當中的值就可以
            log_vars[loss_name] = loss_value.item()

        # loss = 總損失值(float)
        # log_vars = 損失值以及正確率(dict{str:float})
        return loss, log_vars

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        # 已看過，將圖像傳入可以將mask的進行調色後加到原圖上
        # img = 原始圖像，可以是ndarray或是圖像路徑
        # result = 通過模型預測出來的結果
        # palette = 著色用的
        # win_name = 視窗的名稱
        # wait_time = 等待的時間
        # out_file = 將結果保存到哪裏
        # opacity = 不透明度

        # 透過mmcv.imread讀取原始圖像，img(ndarray) shape = [height, width, channel=3]
        img = mmcv.imread(img)
        img = img.copy()
        # seg(ndarray) shape = [height, width]，這裡的高寬會與上面的img相同
        seg = result[0]
        if palette is None:
            # 如果沒有傳入palette就會自動去找
            if self.PALETTE is None:
                # 如果self當中沒有就會透過亂數進行調色
                # Get random state before set seed,
                # and restore random state later.
                # It will prevent loss of randomness, as the palette
                # may be different in each iteration if not specified.
                # See: https://github.com/open-mmlab/mmdetection/issues/5844
                state = np.random.get_state()
                np.random.seed(42)
                # random palette
                palette = np.random.randint(
                    0, 255, size=(len(self.CLASSES), 3))
                np.random.set_state(state)
            else:
                # 從self當中拿取
                palette = self.PALETTE
        # 將palette轉成ndarray型態
        palette = np.array(palette)
        # 檢查一些東西
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        # 構建一個全為0且shape [height, width, 3]的空間
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        # 將顏色填充上去
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR，從RGB轉成BGR
        color_seg = color_seg[..., ::-1]

        # 將原圖與新圖進行疊合
        img = img * (1 - opacity) + color_seg * opacity
        # 轉成uint8格式
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            # 如果有設定輸出檔案位置我們就不會直接show
            show = False

        if show:
            # 進行展示
            mmcv.imshow(img, win_name, wait_time)
        if out_file is not None:
            # 將圖像導出
            mmcv.imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only '
                          'result image will be returned')
            # 將結果回傳
            return img
