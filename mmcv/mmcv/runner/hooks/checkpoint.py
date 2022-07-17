# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional

from mmcv.fileio import FileClient
from ..dist_utils import allreduce_params, master_only
from .hook import HOOKS, Hook


@HOOKS.register_module()
class CheckpointHook(Hook):
    """Save checkpoints periodically.

    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The root directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default. If
            specified, the ``out_dir`` will be the concatenation of ``out_dir``
            and the last level directory of ``runner.work_dir``.
            `Changed in version 1.3.16.`
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
        save_last (bool, optional): Whether to force the last checkpoint to be
            saved regardless of interval. Default: True.
        sync_buffer (bool, optional): Whether to synchronize buffers in
            different gpus. Default: False.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmcv.fileio.FileClient` for details.
            Default: None.
            `New in version 1.3.16.`

    .. warning::
        Before v1.3.16, the ``out_dir`` argument indicates the path where the
        checkpoint is stored. However, since v1.3.16, ``out_dir`` indicates the
        root directory and the final path to save checkpoint is the
        concatenation of ``out_dir`` and the last level directory of
        ``runner.work_dir``. Suppose the value of ``out_dir`` is "/path/of/A"
        and the value of ``runner.work_dir`` is "/path/of/B", then the final
        path will be "/path/of/A/B".
    """

    def __init__(self,
                 interval: int = -1,
                 by_epoch: bool = True,
                 save_optimizer: bool = True,
                 out_dir: Optional[str] = None,
                 max_keep_ckpts: int = -1,
                 save_last: bool = True,
                 sync_buffer: bool = False,
                 file_client_args: Optional[dict] = None,
                 **kwargs):
        """
        :param interval: 多少個迭代會進行一次保存，如果by_epoch為True那麼就是多少個epoch會保存，否則就是多少個batch會保存
        :param by_epoch: 決定一個迭代是以epoch為準或是以一個batch為準
        :param save_optimizer: 是否只保存最佳的結果
        :param out_dir: 保存檔案的位置
        :param max_keep_ckpts: 最多只保存多少個紀錄檔，如果超過了會從最舊的開始刪除
        :param save_last: 是否強制保存最後一個epoch輸出的結果
        :param sync_buffer: 在多gpu當中是否同步buffer
        :param file_client_args:
        :param kwargs:
        """
        # 已看過

        # 進行一系列的傳入值保存
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self.file_client_args = file_client_args

    def before_run(self, runner):
        # 在開始訓練前會進入到這裡進行訓練前的最後動作
        if not self.out_dir:
            # 如果沒有特別指定輸出的路徑，就會與runner當中的路徑相同
            self.out_dir = runner.work_dir

        self.file_client = FileClient.infer_client(self.file_client_args,
                                                   self.out_dir)

        # if `self.out_dir` is not equal to `runner.work_dir`, it means that
        # `self.out_dir` is set so the final `self.out_dir` is the
        # concatenation of `self.out_dir` and the last level directory of
        # `runner.work_dir`
        if self.out_dir != runner.work_dir:
            basename = osp.basename(runner.work_dir.rstrip(osp.sep))
            self.out_dir = self.file_client.join_path(self.out_dir, basename)

        # 將保存到的資料夾位置打印出來
        runner.logger.info(f'Checkpoints will be saved to {self.out_dir} by '
                           f'{self.file_client.name}.')

        # disable the create_symlink option because some file backends do not
        # allow to create a symlink
        # 一些符號連接的設定，這個會與資料儲存的設備會有關係，這部分不用去理解，因為對於學習並不重要
        if 'create_symlink' in self.args:
            if self.args[
                    'create_symlink'] and not self.file_client.allow_symlink:
                self.args['create_symlink'] = False
                warnings.warn(
                    'create_symlink is set as True by the user but is changed'
                    'to be False because creating symbolic link is not '
                    f'allowed in {self.file_client.name}')
        else:
            self.args['create_symlink'] = self.file_client.allow_symlink

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(
                runner, self.interval) or (self.save_last
                                           and self.is_last_epoch(runner)):
            runner.logger.info(
                f'Saving checkpoint at {runner.epoch + 1} epochs')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)

    @master_only
    def _save_checkpoint(self, runner):
        # 已看過，保存模型資訊生成pkt檔，同時如果以達到保存上限會將較舊的模型資料刪除
        """Save the current checkpoint and delete unwanted checkpoint."""
        # 透過save_checkpoint函數進行保存
        runner.save_checkpoint(
            self.out_dir, save_optimizer=self.save_optimizer, **self.args)
        if runner.meta is not None:
            # 如果有meta就將一些資訊保存在其中
            if self.by_epoch:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'epoch_{}.pth').format(runner.epoch + 1)
            else:
                cur_ckpt_filename = self.args.get(
                    'filename_tmpl', 'iter_{}.pth').format(runner.iter + 1)
            runner.meta.setdefault('hook_msgs', dict())
            runner.meta['hook_msgs']['last_ckpt'] = self.file_client.join_path(
                self.out_dir, cur_ckpt_filename)
        # remove other checkpoints，移除較舊的保存資料
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = runner.epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = runner.iter + 1
            redundant_ckpts = range(
                current_ckpt - self.max_keep_ckpts * self.interval, 0,
                -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = self.file_client.join_path(
                    self.out_dir, filename_tmpl.format(_step))
                if self.file_client.isfile(ckpt_path):
                    self.file_client.remove(ckpt_path)
                else:
                    break

    def after_train_iter(self, runner):
        # 已看過
        if self.by_epoch:
            # 如果是以epoch為主的就直接跳過
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        # 只有在指定的迭代數量後會進行保存或是最後一次迭代
        if self.every_n_iters(
                runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):
            # 保存到logger當中
            runner.logger.info(
                f'Saving checkpoint at {runner.iter + 1} iterations')
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            # 進入保存pkt資料
            self._save_checkpoint(runner)
