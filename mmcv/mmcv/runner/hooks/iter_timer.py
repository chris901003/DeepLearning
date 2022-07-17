# Copyright (c) OpenMMLab. All rights reserved.
import time

from .hook import HOOKS, Hook


@HOOKS.register_module()
class IterTimerHook(Hook):

    def before_epoch(self, runner):
        # 已看過
        # 記錄下當下時間，這樣可以在結束的時候知道過程花費多少時間
        self.t = time.time()

    def before_iter(self, runner):
        # 已看過
        # 記錄下載入訓練資料花費時間
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        # 已看過
        # 更新一個batch花費時間
        runner.log_buffer.update({'time': time.time() - self.t})
        self.t = time.time()
