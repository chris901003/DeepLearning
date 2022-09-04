# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import _find_tensors

from ..builder import MODELS, build_module
from ..common import set_requires_grad
from .base_gan import BaseGAN

# _SUPPORT_METHODS_ = ['DCGAN', 'STYLEGANv2']


# @MODELS.register_module(_SUPPORT_METHODS_)
@MODELS.register_module()
class StaticUnconditionalGAN(BaseGAN):
    """Unconditional GANs with static architecture in training.

    This is the standard GAN model containing standard adversarial training
    schedule. To fulfill the requirements of various GAN algorithms,
    ``disc_auxiliary_loss`` and ``gen_auxiliary_loss`` are provided to
    customize auxiliary losses, e.g., gradient penalty loss, and discriminator
    shift loss. In addition, ``train_cfg`` and ``test_cfg`` aims at setuping
    training schedule.

    Args:
        generator (dict): Config for generator.
        discriminator (dict): Config for discriminator.
        gan_loss (dict): Config for generative adversarial loss.
        disc_auxiliary_loss (dict): Config for auxiliary loss to
            discriminator.
        gen_auxiliary_loss (dict | None, optional): Config for auxiliary loss
            to generator. Defaults to None.
        train_cfg (dict | None, optional): Config for training schedule.
            Defaults to None.
        test_cfg (dict | None, optional): Config for testing schedule. Defaults
            to None.
    """

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 disc_auxiliary_loss=None,
                 gen_auxiliary_loss=None,
                 train_cfg=None,
                 test_cfg=None):
        """ 最基礎的GAN模型
        Args:
            generator: 生成器相關配置設定
            discriminator: 鑑別器相關配置設定
            gan_loss: 損失計算相關配置設定
            disc_auxiliary_loss: 鑑別器的輔助訓練設定
            gen_auxiliary_loss: 生成器的輔助訓練設定
            train_cfg: 訓練模式下的流程設定
            test_cfg: 測試模式下的流程設定
        """
        # 繼承自BaseGAN，對繼承對象進行初始化
        super().__init__()
        # 深拷貝一份生成器的config資料到self當中
        self._gen_cfg = deepcopy(generator)
        # 實例化生成器實例對象
        self.generator = build_module(generator)

        # 在驗證模式下可以不用指定鑑別器，因為我們也不會需要用到
        # support no discriminator in testing
        if discriminator is not None:
            # 實例化鑑別器實例對象
            self.discriminator = build_module(discriminator)
        else:
            # 如果沒有指定配置文件就會是None
            self.discriminator = None

        # support no gan_loss in testing
        # 在測試模式下可以不用構建損失計算方式
        if gan_loss is not None:
            # 如果有指定損失計算方式就會到這裡進行實例化
            self.gan_loss = build_module(gan_loss)
        else:
            # 如果沒有指定就會是None
            self.gan_loss = None

        if disc_auxiliary_loss:
            # 如果有使用鑑別器輔助訓練就會到這裡進行實例化
            self.disc_auxiliary_losses = build_module(disc_auxiliary_loss)
            if not isinstance(self.disc_auxiliary_losses, nn.ModuleList):
                # 因為輔助方始可能會有多個，所以會用ModuleList進行包裝
                self.disc_auxiliary_losses = nn.ModuleList(
                    [self.disc_auxiliary_losses])
        else:
            # 如果沒有指定就會設定成None
            self.disc_auxiliary_loss = None

        if gen_auxiliary_loss:
            # 如果有使用生成器輔助訓練就會到這裡進行實例化
            self.gen_auxiliary_losses = build_module(gen_auxiliary_loss)
            if not isinstance(self.gen_auxiliary_losses, nn.ModuleList):
                # 因為輔助方始可能會有多個，所以會用ModuleList進行包裝
                self.gen_auxiliary_losses = nn.ModuleList([self.gen_auxiliary_losses])
        else:
            # 如果沒有指定就會設定成None
            self.gen_auxiliary_losses = None

        # 拷貝train以及test的config資料
        self.train_cfg = deepcopy(train_cfg) if train_cfg else None
        self.test_cfg = deepcopy(test_cfg) if test_cfg else None

        # 解析train_cfg資料
        self._parse_train_cfg()
        if test_cfg is not None:
            # 如果有設定test_cfg就進行解析
            self._parse_test_cfg()

    def _parse_train_cfg(self):
        """Parsing train config and set some attributes for training."""
        # 解析train_cfg當中資料
        if self.train_cfg is None:
            # 如果train_cfg是None就構建一個空的dict
            self.train_cfg = dict()
        # control the work flow in train step
        # 獲取train_cfg當中的disc_steps資料，如果沒有就是預設為1
        self.disc_steps = self.train_cfg.get('disc_steps', 1)

        # whether to use exponential moving average for training
        # 獲取train_cfg當中的use_ema資料，如果沒有就會是False
        self.use_ema = self.train_cfg.get('use_ema', False)
        if self.use_ema:
            # 如果使用ema就會到這裡
            # use deepcopy to guarantee the consistency
            self.generator_ema = deepcopy(self.generator)

        # 獲取real_img_key資料
        self.real_img_key = self.train_cfg.get('real_img_key', 'real_img')

    def _parse_test_cfg(self):
        """Parsing test config and set some attributes for testing."""
        if self.test_cfg is None:
            self.test_cfg = dict()

        # basic testing information
        self.batch_size = self.test_cfg.get('batch_size', 1)

        # whether to use exponential moving average for testing
        self.use_ema = self.test_cfg.get('use_ema', False)
        # TODO: finish ema part

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   loss_scaler=None,
                   use_apex_amp=False,
                   running_status=None):
        """Train step function.

        This function implements the standard training iteration for
        asynchronous adversarial training. Namely, in each iteration, we first
        update discriminator and then compute loss for generator with the newly
        updated discriminator.

        As for distributed training, we use the ``reducer`` from ddp to
        synchronize the necessary params in current computational graph.

        Args:
            data_batch (dict): Input data from dataloader.
            optimizer (dict): Dict contains optimizer for generator and
                discriminator.
            ddp_reducer (:obj:`Reducer` | None, optional): Reducer from ddp.
                It is used to prepare for ``backward()`` in ddp. Defaults to
                None.
            loss_scaler (:obj:`torch.cuda.amp.GradScaler` | None, optional):
                The loss/gradient scaler used for auto mixed-precision
                training. Defaults to ``None``.
            use_apex_amp (bool, optional). Whether to use apex.amp. Defaults to
                ``False``.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.

        Returns:
            dict: Contains 'log_vars', 'num_samples', and 'results'.
        """
        # get data from data_batch
        # 獲取真實圖像資料，tensor shape [batch_size, channel, height, width]
        real_imgs = data_batch[self.real_img_key]
        # If you adopt ddp, this batch size is local batch size for each GPU.
        # If you adopt dp, this batch size is the global batch size as usual.
        # 獲取圖像的batch_size
        batch_size = real_imgs.shape[0]

        # get running status
        if running_status is not None:
            # 如果有傳入running_status就會獲取當前是第幾個batch
            curr_iter = running_status['iteration']
        else:
            # 如果沒有給定running_status就自己瞎搞一個
            # dirty walk round for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        # disc training
        # 將self.discriminator的反向傳遞開啟
        set_requires_grad(self.discriminator, True)
        # 將鑑別器的優化器進行反向清空
        optimizer['discriminator'].zero_grad()
        # TODO: add noise sampler to customize noise sampling

        # pass model specific training kwargs
        # 保存生成器的相關參數
        g_training_kwargs = {}
        if hasattr(self.generator, 'get_training_kwargs'):
            # 如果在generator當中有get_training_kwargs就會到這裡，將資料放到g_training_kwargs當中
            g_training_kwargs.update(self.generator.get_training_kwargs(phase='disc'))

        with torch.no_grad():
            # 進行生成圖像，這裡會將反向傳遞進行關閉，fake_imgs = tensor shape [batch_size, channel, height, width]
            fake_imgs = self.generator(None, num_batches=batch_size, **g_training_kwargs)

        # disc pred for fake imgs and real_imgs
        # 將假的圖像放到鑑別器模型當中進行正向推理，disc_pred_fake = tensor shape [batch_size, channel=1]
        disc_pred_fake = self.discriminator(fake_imgs)
        # 將資料集的圖像放到鑑別器當中進行正向推理，disc_pred_real = tensor shape [batch_size, channel=1]
        disc_pred_real = self.discriminator(real_imgs)
        # get data dict to compute losses for disc
        # 構建一個data_dict_用來計算損失用的
        data_dict_ = dict(
            # 將生成器模型
            gen=self.generator,
            # 將鑑別器模型
            disc=self.discriminator,
            # 預測生成圖結果資料
            disc_pred_fake=disc_pred_fake,
            # 預測真實圖結果資料
            disc_pred_real=disc_pred_real,
            # 假圖資料
            fake_imgs=fake_imgs,
            # 真圖資料
            real_imgs=real_imgs,
            # 當前是第幾個batch
            iteration=curr_iter,
            # batch大小
            batch_size=batch_size,
            loss_scaler=loss_scaler)

        # 將data_dict_傳入進行計算損失值
        # loss_disc = 損失總和，tensor float
        # log_vars_disc = 個別損失資料，dict
        loss_disc, log_vars_disc = self._get_disc_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            # 準備進行反向傳遞，在使用分布式訓練時會需要的動作
            ddp_reducer.prepare_for_backward(_find_tensors(loss_disc))

        if loss_scaler:
            # add support for fp16
            loss_scaler.scale(loss_disc).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(
                    loss_disc, optimizer['discriminator'],
                    loss_id=0) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            # 進行loss反向傳遞
            loss_disc.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['discriminator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['discriminator'])
            # loss_scaler.update will be called in runner.train()
        else:
            # 將優化器進行step操作，到這裡就算將鑑別器進行一次訓練
            optimizer['discriminator'].step()

        # skip generator training if only train discriminator for current iteration
        if (curr_iter + 1) % self.disc_steps != 0:
            # 可以透過設定disc_steps讓某幾個batch只會訓練鑑別器
            # 構建results，裡面就會有創建的假圖以及真圖
            results = dict(fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
            # 構建最終回傳的資料
            outputs = dict(
                log_vars=log_vars_disc,
                num_samples=batch_size,
                results=results)
            if hasattr(self, 'iteration'):
                # 將iteration加一
                self.iteration += 1
            # 最終回傳結果
            return outputs

        # generator training
        # 將鑑別器的反向傳遞關閉
        set_requires_grad(self.discriminator, False)
        # 將生成器的優化器反向傳遞值進行清空
        optimizer['generator'].zero_grad()

        # TODO: add noise sampler to customize noise sampling

        # pass model specific training kwargs
        g_training_kwargs = {}
        if hasattr(self.generator, 'get_training_kwargs'):
            # 如果generator當中有get_trainging_kwargs就會存到g_training_kwargs當中
            g_training_kwargs.update(self.generator.get_training_kwargs(phase='gen'))

        # 透過generator創造出假圖，tensor shape [batch_size, channel, height, width]
        fake_imgs = self.generator(None, num_batches=batch_size, **g_training_kwargs)
        # 對假圖進行預測
        disc_pred_fake_g = self.discriminator(fake_imgs)

        # 構建計算損失需要使用的字典
        data_dict_ = dict(
            # 生成器模型
            gen=self.generator,
            # 鑑別器模型
            disc=self.discriminator,
            # 假圖資料
            fake_imgs=fake_imgs,
            # 鑑別器辨識結果
            disc_pred_fake_g=disc_pred_fake_g,
            # 當前第幾個batch
            iteration=curr_iter,
            # batch大小
            batch_size=batch_size,
            loss_scaler=loss_scaler)

        # 進行損失計算
        loss_gen, log_vars_g = self._get_gen_loss(data_dict_)

        # prepare for backward in ddp. If you do not call this function before
        # back propagation, the ddp will not dynamically find the used params
        # in current computation.
        if ddp_reducer is not None:
            # 如果在分布式訓練上會需要
            ddp_reducer.prepare_for_backward(_find_tensors(loss_gen))

        if loss_scaler:
            loss_scaler.scale(loss_gen).backward()
        elif use_apex_amp:
            from apex import amp
            with amp.scale_loss(
                    loss_gen, optimizer['generator'],
                    loss_id=1) as scaled_loss_disc:
                scaled_loss_disc.backward()
        else:
            # 進行損失的反向傳遞
            loss_gen.backward()

        if loss_scaler:
            loss_scaler.unscale_(optimizer['generator'])
            # note that we do not contain clip_grad procedure
            loss_scaler.step(optimizer['generator'])
            # loss_scaler.update will be called in runner.train()
        else:
            # 透過優化器進行權重更新
            optimizer['generator'].step()

        # update ada p
        if hasattr(self.discriminator, 'with_ada') and self.discriminator.with_ada:
            # 如果discriminator有with_ada就會到這裡
            self.discriminator.ada_aug.log_buffer[0] += batch_size
            self.discriminator.ada_aug.log_buffer[1] += disc_pred_real.sign().sum()
            self.discriminator.ada_aug.update(iteration=curr_iter, num_batches=batch_size)
            log_vars_disc['augment'] = (self.discriminator.ada_aug.aug_pipeline.p.data.cpu())

        # 構建log所需的字典
        log_vars = {}
        # 生成器的損失資訊
        log_vars.update(log_vars_g)
        # 鑑別器的損失資訊
        log_vars.update(log_vars_disc)

        # 最終的results資訊，這裡會有假圖以及真圖的資訊
        results = dict(fake_imgs=fake_imgs.cpu(), real_imgs=real_imgs.cpu())
        # 將資料用dict包裝其來放到outputs當中
        outputs = dict(log_vars=log_vars, num_samples=batch_size, results=results)

        if hasattr(self, 'iteration'):
            # 更新iteration
            self.iteration += 1
        # 回傳outputs
        return outputs
