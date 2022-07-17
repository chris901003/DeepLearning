# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmseg import digit_version

dp_factory = {'cuda': MMDataParallel, 'cpu': MMDataParallel}

ddp_factory = {'cuda': MMDistributedDataParallel}


def build_dp(model, device='cuda', dim=0, *args, **kwargs):
    """build DataParallel module by device type.

    if device is cuda, return a MMDataParallel module; if device is mlu,
    return a MLUDataParallel module.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, cuda, cpu or mlu. Defaults to cuda.
        dim (int): Dimension used to scatter the data. Defaults to 0.

    Returns:
        :class:`nn.Module`: parallelized module.
    """
    # 已看過

    if device == 'cuda':
        # 將模型放到cuda上進行訓練
        model = model.cuda()
    elif device == 'mlu':
        # 使用mlu需要將mmcv的版本升級到1.5.0以上
        assert digit_version(mmcv.__version__) >= digit_version('1.5.0'), \
                'Please use MMCV >= 1.5.0 for MLU training!'
        from mmcv.device.mlu import MLUDataParallel
        dp_factory['mlu'] = MLUDataParallel
        model = model.mlu()

    # 將model再放到dp_factory當中進行加工
    return dp_factory[device](model, dim=dim, *args, **kwargs)


def build_ddp(model, device='cuda', *args, **kwargs):
    """Build DistributedDataParallel module by device type.

    If device is cuda, return a MMDistributedDataParallel module;
    if device is mlu, return a MLUDistributedDataParallel module.

    Args:
        model (:class:`nn.Module`): module to be parallelized.
        device (str): device type, mlu or cuda.

    Returns:
        :class:`nn.Module`: parallelized module.

    References:
        .. [1] https://pytorch.org/docs/stable/generated/torch.nn.parallel.
                     DistributedDataParallel.html
    """
    assert device in ['cuda', 'mlu'], 'Only available for cuda or mlu devices.'
    if device == 'cuda':
        model = model.cuda()
    elif device == 'mlu':
        assert digit_version(mmcv.__version__) >= digit_version('1.5.0'), \
            'Please use MMCV >= 1.5.0 for MLU training!'
        from mmcv.device.mlu import MLUDistributedDataParallel
        ddp_factory['mlu'] = MLUDistributedDataParallel
        model = model.mlu()

    return ddp_factory[device](model, *args, **kwargs)


def is_mlu_available():
    """Returns a bool indicating if MLU is currently available."""
    # 已看過
    # 返回是否可以使用mlu進行訓練
    return hasattr(torch, 'is_mlu_available') and torch.is_mlu_available()


def get_device():
    """Returns an available device, cpu, cuda or mlu."""
    # 已看過
    # 設定模型將會在哪種設備上進行訓練，這裡只有支援[cpu, cuda, mlu]
    is_device_available = {
        'cuda': torch.cuda.is_available(),
        'mlu': is_mlu_available()
    }
    # device_list = 看哪些是可以使用的
    device_list = [k for k, v in is_device_available.items() if v]
    # 如果cuda以及mlu都沒有辦法使用就會使用cpu進行訓練
    return device_list[0] if len(device_list) == 1 else 'cpu'
