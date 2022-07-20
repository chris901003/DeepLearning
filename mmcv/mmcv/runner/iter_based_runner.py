# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, no_type_check

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mmcv
from .base_runner import BaseRunner
from .builder import RUNNERS
from .checkpoint import save_checkpoint
from .hooks import IterTimerHook
from .utils import get_host_info


class IterLoader:

    def __init__(self, dataloader: DataLoader):
        # 已看過

        self._dataloader = dataloader
        # iter_loader = 將dataloader當中的迭代器拿出來
        self.iter_loader = iter(self._dataloader)
        # 將epoch設定成0
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            # 對迭代器使用next可以獲取下個batch資訊
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


@RUNNERS.register_module()
class IterBasedRunner(BaseRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

    def train(self, data_loader, **kwargs):
        """
        :param data_loader: IterLoader的實例化對象，裏面有資料集
        :param kwargs: 其他參數，通常為空
        :return:
        """
        # 已看過

        # 將模型設定成訓練模式
        self.model.train()
        # 將當前的狀態設定成train
        self.mode = 'train'
        # 將data_loader保存下來
        self.data_loader = data_loader
        # 保存下當前的epoch
        self._epoch = data_loader.epoch
        # 獲取當前batch的資訊，詳細資訊會因為不同的資料集有不同的內容，這裡只是舉個例子
        # data_batch = {
        #   'img_metas': 一個batch的詳細資訊，主要是紀錄用的
        #   'img': 原始圖像，裏面的data會有一個batch的圖像已經堆疊好的tensor，shape [batch_size, channel, width, height]
        #   'gt_semantic_seg': 標註圖像，裏面也是一個batch堆疊好的tensor
        # }
        # 這裡會呼叫IterLoader類的__next__函數
        data_batch = next(data_loader)
        # 將資訊儲存起來
        self.data_batch = data_batch
        # 使用before_train_iter的鉤子
        self.call_hook('before_train_iter')
        # 將訓練資料丟入到模型當中進行正向傳播
        # outputs = 損失計算後的結果
        # outputs = dict{
        #   'loss': float[總損失值],
        #   'log_vars': OrderedDict({str: float})[各種損失係項以及正確率資訊],
        #   'num_samples': int[batch_size]
        # }
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            # outputs需要是字典格式
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            # 將log_vars當中的資訊保存到logger當中
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        # 保存outputs資訊
        self.outputs = outputs
        # 呼叫after_train_iter的鉤子函數
        self.call_hook('after_train_iter')
        # 將這個batch的圖像以及標註資訊清除
        del self.data_batch
        # iter計數增加一
        self._inner_iter += 1
        self._iter += 1

    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        # 已看過，驗證模式會進入到這裡
        # 將模型設定成驗證模式
        self.model.eval()
        # 將當前模型狀態設定成驗證模式
        self.mode = 'val'
        # 保存dataloader
        self.data_loader = data_loader
        # 獲取一個batch的資訊，這裡獲取過程與train完全相同，只是dataset不同而已
        data_batch = next(data_loader)
        self.data_batch = data_batch
        # 呼叫before_val_iter鉤子函數，這裡幾乎都沒有做任何事情
        self.call_hook('before_val_iter')
        # 開始將資料進行向前傳遞
        outputs = self.model.val_step(data_batch, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.val_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_val_iter')
        del self.data_batch
        self._inner_iter += 1

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_iters: Optional[int] = None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        # 已看過
        # data_loaders = list[Dataloader]，list長度等於有多少種不同資料集，通常只會有一種
        # workflow = 整個run的流程，可以看上面的解釋

        # 檢查各項是否符合規定
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            # max_iters需要直接放到self._max_iters當中，舊的寫法已經被淘汰了
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        # 需要設定_max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        # 取出work_dir位置，後面logger需要用的
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        # 在logger當中保存資訊
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        # 呼叫before_run的所有鉤子函數
        self.call_hook('before_run')

        # 將data_loaders的每一個放入到IterLoader class當中進行初始化
        # iter_loaders = IterLoader的實例對象
        iter_loaders = [IterLoader(x) for x in data_loaders]

        # 呼叫before_epoch的所有鉤子函數
        self.call_hook('before_epoch')

        # 開始訓練一個epoch，總共會迭代max_iters次
        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                # 將內部的iter設定成0，這個是用來計算batch的iter的
                self._inner_iter = 0
                # 將mode以及iters拆開來
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    # 檢查格式是否正確且self當中也要有該mode
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                # 獲取迭代器
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    # 將iter_loaders當中對應到的index的實例化對象傳入
                    iter_runner(iter_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        # 呼叫訓練完所有epoch後的鉤子函數，很多都是直接pass
        self.call_hook('after_epoch')
        # 呼叫最後的鉤子函數，很多都是直接pass
        self.call_hook('after_run')

    @no_type_check
    def resume(self,
               checkpoint: str,
               resume_optimizer: bool = True,
               map_location: Union[str, Callable] = 'default') -> None:
        """Resume model from checkpoint.

        Args:
            checkpoint (str): Checkpoint to resume from.
            resume_optimizer (bool, optional): Whether resume the optimizer(s)
                if the checkpoint file includes optimizer(s). Default to True.
            map_location (str, optional): Same as :func:`torch.load`.
                Default to 'default'.
        """
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        self._inner_iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        self.logger.info(f'resumed from epoch: {self.epoch}, iter {self.iter}')

    def save_checkpoint(  # type: ignore
            self,
            out_dir: str,
            filename_tmpl: str = 'iter_{}.pth',
            meta: Optional[Dict] = None,
            save_optimizer: bool = True,
            create_symlink: bool = True) -> None:
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None,
                                custom_hooks_config=None):
        """Register default hooks for iter-based training.

        Checkpoint hook, optimizer stepper hook and logger hooks will be set to
        `by_epoch=False` by default.

        Default hooks include:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | MomentumUpdaterHook  | HIGH (30)               |
        +----------------------+-------------------------+
        | OptimizerStepperHook | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointSaverHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | LoggerHook(s)        | VERY_LOW (90)           |
        +----------------------+-------------------------+
        | CustomHook(s)        | defaults to NORMAL (50) |
        +----------------------+-------------------------+

        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """

        """
        :param lr_config: 學習率調整相關的設定資料 
        :param optimizer_config: 優化器設定相關資料
        :param checkpoint_config: 保存點相關資料
        :param log_config: log檔相關設定資料
        :param momentum_config: momentum相關資料
        :param custom_hooks_config: 自定義鉤子函數
        :return: 
        """
        # 已看過

        # 將一下的幾個部分都添加上by_epoch且value都設定成False
        if checkpoint_config is not None:
            checkpoint_config.setdefault('by_epoch', False)  # type: ignore
        if lr_config is not None:
            lr_config.setdefault('by_epoch', False)  # type: ignore
        if log_config is not None:
            for info in log_config['hooks']:
                info.setdefault('by_epoch', False)
        super().register_training_hooks(
            lr_config=lr_config,
            momentum_config=momentum_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
            log_config=log_config,
            # 跟時間相關的鉤子函數
            timer_config=IterTimerHook(),
            custom_hooks_config=custom_hooks_config)
