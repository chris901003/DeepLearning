# Copyright (c) OpenMMLab. All rights reserved.
import functools
import warnings

import torch


class OutputHook:
    """Output feature map of some layers.

    Args:
        module (nn.Module): The whole module to get layers.
        outputs (tuple[str] | list[str]): Layer name to output. Default: None.
        as_tensor (bool): Determine to return a tensor or a numpy array.
            Default: False.
    """

    def __init__(self, module, outputs=None, as_tensor=False):
        """ 已看過，可以將正向傳播時中間的特徵層提取出來
        Args:
            module: 模型本身，會是nn.Module類型
            outputs: 需要輸出的layer名稱
            as_tensor: 如果是True輸出的就會是tensor格式，如過是False輸出的就會是ndarray格式
        """
        # 將傳入資料進行保存
        self.outputs = outputs
        self.as_tensor = as_tensor
        # 構建layer_outputs字典，保存層輸出的
        self.layer_outputs = {}
        # 構建handles的list，用來保存鉤子的
        self.handles = []
        # 呼叫register函數並且將module放入
        self.register(module)

    def register(self, module):
        # 已看過，傳入模型本身

        def hook_wrapper(name):
            # name = 層結構名稱

            def hook(model, input, output):
                if not isinstance(output, torch.Tensor):
                    # 如果output不是tensor格式就會跳出警告
                    warnings.warn(f'Directly return the output from {name}, '
                                  f'since it is not a tensor')
                    # 因為不是tensor格式，所以不管指定的是tensor或是ndarray就直接保存
                    self.layer_outputs[name] = output
                elif self.as_tensor:
                    # 如果有指定是tensor就直接保存
                    self.layer_outputs[name] = output
                else:
                    # 如果有指定需要是ndarray就會將tensor轉成ndarray後保存
                    self.layer_outputs[name] = output.detach().cpu().numpy()

            # 將hook函數進行返回
            return hook

        if isinstance(self.outputs, (list, tuple)):
            # 如果output是list或是tuple就會進來，表示有指的輸出層結構
            # 遍歷需要輸出的層結構
            for name in self.outputs:
                try:
                    # 嘗試獲取層結構實例化對象
                    layer = rgetattr(module, name)
                    # 在層結構當中註冊鉤子函數，將hook_wrapper註冊進去
                    h = layer.register_forward_hook(hook_wrapper(name))
                except AttributeError:
                    # 如果沒有找到對應層結構名稱就會報錯
                    raise AttributeError(f'Module {name} not found')
                # 將h保存在handles
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
