# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import MODULES


@MODULES.register_module()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge',
            'wgan-logistic-ns'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        """ 實例化GAN損失計算
        Args:
            gan_type: GAN的類型
            real_label_val: 真實圖像的標籤值
            fake_label_val: 假圖像的標籤值
            loss_weight: 損失權重
        """
        # 繼承自nn.Module，將繼承對象進行初始化
        super().__init__()
        # 保存傳入的參數
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        # 根據不同的GAN類型會有不同的損失計算方式
        if self.gan_type == 'vanilla':
            # 這裡如果使用的是vanilla就會用BCE二值交叉熵損失
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan-logistic-ns':
            self.loss = self._wgan_logistic_ns_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            # 其他就會直接報錯
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_logistic_ns_loss(self, input, target):
        """WGAN loss in logistically non-saturating mode.

        This loss is widely used in StyleGANv2.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """

        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise, \
                return Tensor.
        """
        # 根據傳入的input以及是否為真實圖像進行構建真實標籤

        if self.gan_type in ['wgan', 'wgan-logistic-ns']:
            # 如果gan的類型是以上兩種就會直接返回target_is_real
            return target_is_real
        # 根據是否為真實圖像獲取對應的標籤值
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        # 會將標籤值重複batch_size次
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        """ 計算損失值
        Args:
            input: 輸入要進行損失計算的結果
            target_is_real: 是否為真實圖像
            is_disc: 計算的是否為鑑別器的損失，在計算生成器的損失時會是False
        """
        # 根據input以及是否為真實圖像構建目標標籤，tensor shape [batch_size, 1]
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            # 如果gan的類型是hinge就會到這裡
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            # 直接進行損失計算
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        # 如果有需要的話會將損失再乘上權重
        return loss if is_disc else loss * self.loss_weight
