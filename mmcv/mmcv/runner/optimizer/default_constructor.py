# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn import GroupNorm, LayerNorm

from mmcv.utils import _BatchNorm, _InstanceNorm, build_from_cfg, is_list_of
from mmcv.utils.ext_loader import check_ops_exist
from .builder import OPTIMIZER_BUILDERS, OPTIMIZERS


@OPTIMIZER_BUILDERS.register_module()
class DefaultOptimizerConstructor:
    """Default constructor for optimizers.

    By default each parameter share the same optimizer settings, and we
    provide an argument ``paramwise_cfg`` to specify parameter-wise settings.
    It is a dict and may contain the following fields:

    - ``custom_keys`` (dict): Specified parameters-wise settings by keys. If
      one of the keys in ``custom_keys`` is a substring of the name of one
      parameter, then the setting of the parameter will be specified by
      ``custom_keys[key]`` and other setting like ``bias_lr_mult`` etc. will
      be ignored. It should be noted that the aforementioned ``key`` is the
      longest key that is a substring of the name of the parameter. If there
      are multiple matched keys with the same length, then the key with lower
      alphabet order will be chosen.
      ``custom_keys[key]`` should be a dict and may contain fields ``lr_mult``
      and ``decay_mult``. See Example 2 below.
    - ``bias_lr_mult`` (float): It will be multiplied to the learning
      rate for all bias parameters (except for those in normalization
      layers and offset layers of DCN).
    - ``bias_decay_mult`` (float): It will be multiplied to the weight
      decay for all bias parameters (except for those in
      normalization layers, depthwise conv layers, offset layers of DCN).
    - ``norm_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of normalization
      layers.
    - ``dwconv_decay_mult`` (float): It will be multiplied to the weight
      decay for all weight and bias parameters of depthwise conv
      layers.
    - ``dcn_offset_lr_mult`` (float): It will be multiplied to the learning
      rate for parameters of offset layer in the deformable convs
      of a model.
    - ``bypass_duplicate`` (bool): If true, the duplicate parameters
      would not be added into optimizer. Default: False.

    Note:

        1. If the option ``dcn_offset_lr_mult`` is used, the constructor will
        override the effect of ``bias_lr_mult`` in the bias of offset layer.
        So be careful when using both ``bias_lr_mult`` and
        ``dcn_offset_lr_mult``. If you wish to apply both of them to the offset
        layer in deformable convs, set ``dcn_offset_lr_mult`` to the original
        ``dcn_offset_lr_mult`` * ``bias_lr_mult``.

        2. If the option ``dcn_offset_lr_mult`` is used, the constructor will
        apply it to all the DCN layers in the model. So be careful when the
        model contains multiple DCN layers in places other than backbone.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are

                - `type`: class name of the optimizer.

            Optional fields are

                - any arguments of the corresponding optimizer type, e.g.,
                  lr, weight_decay, momentum, etc.
        paramwise_cfg (dict, optional): Parameter-wise options.

    Example 1:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> paramwise_cfg = dict(norm_decay_mult=0.)
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)

    Example 2:
        >>> # assume model have attribute model.backbone and model.cls_head
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, weight_decay=0.95)
        >>> paramwise_cfg = dict(custom_keys={
                'backbone': dict(lr_mult=0.1, decay_mult=0.9)})
        >>> optim_builder = DefaultOptimizerConstructor(
        >>>     optimizer_cfg, paramwise_cfg)
        >>> optimizer = optim_builder(model)
        >>> # Then the `lr` and `weight_decay` for model.backbone is
        >>> # (0.01 * 0.1, 0.95 * 0.9). `lr` and `weight_decay` for
        >>> # model.cls_head is (0.01, 0.95).
    """

    def __init__(self,
                 optimizer_cfg: Dict,
                 paramwise_cfg: Optional[Dict] = None):
        # 已看過
        # optimizer_cfg = 優化器的設定，包含指定使用哪種優化器
        # paramwise_cfg = 對於某些特定地方會有特別的權重值

        if not isinstance(optimizer_cfg, dict):
            # 如果optimizer_cfg不是dict就會報錯
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optimizer_cfg)}')
        # 將優化器配置文件保存
        self.optimizer_cfg = optimizer_cfg
        # 保存paramwise設定
        self.paramwise_cfg = {} if paramwise_cfg is None else paramwise_cfg
        # 獲取基礎學習率
        self.base_lr = optimizer_cfg.get('lr', None)
        # 獲取weight_decay
        self.base_wd = optimizer_cfg.get('weight_decay', None)
        # 用來檢查cfg當中有沒有格式錯誤的地方
        self._validate_cfg()

    def _validate_cfg(self) -> None:
        # 用來檢查cfg當中有沒有格式錯誤的地方
        if not isinstance(self.paramwise_cfg, dict):
            raise TypeError('paramwise_cfg should be None or a dict, '
                            f'but got {type(self.paramwise_cfg)}')

        if 'custom_keys' in self.paramwise_cfg:
            if not isinstance(self.paramwise_cfg['custom_keys'], dict):
                raise TypeError(
                    'If specified, custom_keys must be a dict, '
                    f'but got {type(self.paramwise_cfg["custom_keys"])}')
            if self.base_wd is None:
                for key in self.paramwise_cfg['custom_keys']:
                    if 'decay_mult' in self.paramwise_cfg['custom_keys'][key]:
                        raise ValueError('base_wd should not be None')

        # get base lr and weight decay
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in self.paramwise_cfg
                or 'norm_decay_mult' in self.paramwise_cfg
                or 'dwconv_decay_mult' in self.paramwise_cfg):
            if self.base_wd is None:
                raise ValueError('base_wd should not be None')

    def _is_in(self, param_group: Dict, param_group_list: List) -> bool:
        assert is_list_of(param_group_list, dict)
        param = set(param_group['params'])
        param_set = set()
        for group in param_group_list:
            param_set.update(set(group['params']))

        return not param.isdisjoint(param_set)

    def add_params(self,
                   params: List[Dict],
                   module: nn.Module,
                   prefix: str = '',
                   is_dcn_module: Union[int, float, None] = None) -> None:
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module
            is_dcn_module (int|float|None): If the current module is a
                submodule of DCN, `is_dcn_module` will be passed to
                control conv_offset layer's learning rate. Defaults to None.
        """
        # 已看過，讀出模型當中所有的參數
        # parmas: 最後要將模型當中的參數往這裏面放
        # module: 模型本身
        # prefix: 模型的前綴
        # is_dcn_module: 如果是DCN模型就會進行特殊處理

        # get param-wise options
        # 獲取自定義的key值
        custom_keys = self.paramwise_cfg.get('custom_keys', {})
        # first sort with alphabet order and then sort with reversed len of str
        # 我們先對key的英文字母進行排序，再對字串長度進行反向排序
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        # 從paramwise_cfg中取出一些資訊，大部分是獲取學習率
        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.)
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.)
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.)
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.)
        bypass_duplicate = self.paramwise_cfg.get('bypass_duplicate', False)
        dcn_offset_lr_mult = self.paramwise_cfg.get('dcn_offset_lr_mult', 1.)

        # special rules for norm layers and depth-wise conv layers
        # 如果module傳入的是標準化層的類型這裡就會是True
        is_norm = isinstance(module,
                             (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
        # 這裡如果傳入的module是DW卷積就會是True
        is_dwconv = (
            isinstance(module, torch.nn.Conv2d)
            and module.in_channels == module.groups)

        # 遍歷module當中的所有層結構，這裡recurse=False表示只會遍歷一層結構，不會進行遞迴往下
        for name, param in module.named_parameters(recurse=False):
            # 這裡會構建一個param_group的字典，key就是固定的params且value就是當前遍歷到的層結構
            param_group = {'params': [param]}
            if not param.requires_grad:
                # 如果不是需要訓練的層結構就會到這裡來，添加到parmas當中就直接下一個
                params.append(param_group)
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                # 如果有遇到重複的層結構且有設定如過遇到就直接跳過就會到這裡來
                # 這裡會給出警告表示有被跳過的部分
                warnings.warn(f'{prefix} is duplicate. It is skipped since '
                              f'bypass_duplicate={bypass_duplicate}')
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            # 這裡我們會去遍歷所有自定義的部分，如果有對應上的我們就會捨棄其他的設定，直接套用自定義的設定
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    # 如果有配對上就會到這裡來，將is_custom設定成True
                    is_custom = True
                    # 獲取自定義指定的學習率，如果沒有找到就默認使用1
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    # 將學習率設定成，基礎學習率乘上自定義學習率倍率
                    param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        # 如果有設定基礎weight_decay就會到這裡來
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                    break

            if not is_custom:
                # 如果在自定義當中都沒有找到就會到這裡來
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if name == 'bias' and not (is_norm or is_dcn_module):
                    # 如果是bias參數就會到這裡來
                    param_group['lr'] = self.base_lr * bias_lr_mult

                if (prefix.find('conv_offset') != -1 and is_dcn_module
                        and isinstance(module, torch.nn.Conv2d)):
                    # deal with both dcn_offset's bias & weight
                    param_group['lr'] = self.base_lr * dcn_offset_lr_mult

                # apply weight decay policies
                if self.base_wd is not None:
                    # 設定weight_decay的部分
                    # norm decay
                    if is_norm:
                        param_group[
                            'weight_decay'] = self.base_wd * norm_decay_mult
                    # depth-wise conv
                    elif is_dwconv:
                        param_group[
                            'weight_decay'] = self.base_wd * dwconv_decay_mult
                    # bias lr and decay
                    elif name == 'bias' and not is_dcn_module:
                        # TODO: current bias_decay_mult will have affect on DCN
                        param_group[
                            'weight_decay'] = self.base_wd * bias_decay_mult
            # 將結果保存到params當中，如果當中沒有設定lr就表示該參數不會參與學習當中
            params.append(param_group)

        if check_ops_exist():
            from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
            is_dcn_module = isinstance(module,
                                       (DeformConv2d, ModulatedDeformConv2d))
        else:
            is_dcn_module = False
        # 這裡會遍歷整個module的層結構，這裡只會遍歷子模塊的迭代器
        for child_name, child_mod in module.named_children():
            # 給模塊部分給一個前綴名
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            # 這裡我們透過遞迴將所有的學習參數放到params當中
            self.add_params(
                params,
                child_mod,
                prefix=child_prefix,
                is_dcn_module=is_dcn_module)

    def __call__(self, model: nn.Module):
        # 已看過
        # 我們會將模型傳入進行優化器實例化

        if hasattr(model, 'module'):
            # 因為我們需要的model是在傳入的model當中，所以需要取出來
            model = model.module

        # 將優化器設定資料取出來，這裡使用copy方法
        optimizer_cfg = self.optimizer_cfg.copy()
        # if no paramwise option is specified, just use the global setting
        if not self.paramwise_cfg:
            # 如果沒有指定哪些部分要使用多少學習率，這裡就統一設定
            # 獲取需要學習的層結構的參數
            optimizer_cfg['params'] = model.parameters()
            # 將設定檔以及註冊器傳入，獲得指定優化器的實例對象
            return build_from_cfg(optimizer_cfg, OPTIMIZERS)

        # 有特別設定不同區塊的學習率就會到這裡進行設定
        # set param-wise lr and weight decay recursively
        params: List[Dict] = []
        self.add_params(params, model)
        # 將從add_params獲得到的params放到優化器的config當中
        optimizer_cfg['params'] = params

        # 透過config進行優化器構建
        return build_from_cfg(optimizer_cfg, OPTIMIZERS)
